"""Frozen four-arm comparison of measured, requirement-conditioned controllers."""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from torch import nn

from agg.consolidation import Proposal, trial
from agg.controller.config import ControllerConfig, MetricGuard
from agg.controller.core import Controller
from agg.controller.events import EventLog
from agg.controller.resource_training import MeasuredTrainingExecutor
from agg.controller.resources import ResourceProfile, measure_resources
from agg.controller.training import TrainingObserver, TrainingPolicy
from agg.evaluation import Constraints
from agg.evaluation.model import make_evaluator
from agg.ledger import Ledger
from agg.tasks import TaskData
from agg.tasks.protocol import make_protocol
from agg.training import TrainConfig, evaluate, train

from .blocks import _paired
from .checkpoints import load_model, save_model
from .independent import CandidateRecorder, file_hash, freeze_run, validate_frozen_run


@dataclass(frozen=True)
class ResourceConfig:
    tasks: tuple[str, ...] = ("retrieval", "modular")
    seeds: tuple[int, ...] = (17, 19)
    data_seed: int = 137
    samples: int = 32
    initial_steps: int = 2
    intervention_steps: int = 3
    batch_size: int = 16
    width: int = 8
    repeats: int = 9
    profile: ResourceProfile = field(
        default_factory=lambda: ResourceProfile(
            min_id=0.05,
            min_ood=0.05,
            max_latency_ms=50,
            max_model_bytes=100000,
            max_parameters=25000,
            max_training_updates=3,
        )
    )

    def __post_init__(self) -> None:
        if not self.tasks or not set(self.tasks) <= {"retrieval", "modular", "hierarchy"}:
            raise ValueError("known task families required")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or len(set(self.tasks)) != len(self.tasks)
        ):
            raise ValueError("nonempty unique seeds and tasks required")
        if any(type(v) is not int or not 0 <= v < 2**32 - 4 for v in (*self.seeds, self.data_seed)):
            raise ValueError("valid independent seeds required")
        if any(
            type(v) is not int or v < 1
            for v in (
                self.samples,
                self.initial_steps,
                self.intervention_steps,
                self.batch_size,
                self.width,
                self.repeats,
            )
        ):
            raise ValueError("positive integer budgets required")
        if self.width % 2 or self.repeats < 3 or self.samples < 10:
            raise ValueError("even width, repeats>=3 and samples>=10 required")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResourceConfig":
        raw = dict(values)
        for key in ("tasks", "seeds"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if "profile" in raw:
            raw["profile"] = ResourceProfile(**raw["profile"])
        return cls(**raw)


def choose_experience(records: list[dict[str, Any]], task: str) -> dict[str, Any] | None:
    eligible = [
        r
        for r in records
        if r["task"] == task
        and r["role"] == "selection"
        and r["status"] == "committed"
        and r["joint_feasible"]
        and type(r["objective"]) in (int, float)
        and math.isfinite(r["objective"])
    ]
    return min(eligible, key=lambda r: r["objective"]) if eligible else None


class _GuidedPolicy(TrainingPolicy):
    def __init__(self, config: ControllerConfig, source: dict[str, Any] | None):
        super().__init__(config)
        self.source = source

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        proposal = super().propose(*args, **kwargs)
        if self.source is not None and proposal.action.value == "adjust_learning_rate":
            proposal = replace(proposal, parameters={"fraction": self.source["fraction"]})
        return proposal


def _scores(model: nn.Module, data: TaskData) -> dict[str, dict[str, float]]:
    return {
        name: evaluate(model, split)
        for name, split in (("train", data.train), ("id", data.id), ("ood", data.ood))
    }


def _capability(scores: dict[str, dict[str, float]], metric: str) -> dict[str, float]:
    return {
        "performance.validation_score": scores["id"][metric],
        "performance.ood_score": scores["ood"][metric],
    }


def _protocol(config: ResourceConfig, task: str) -> Any:
    return make_protocol(
        task,
        seed=config.data_seed,
        samples=config.samples,
        length=8,
        distance=2,
        modulus=17,
        depth=4,
    )


def run_resources(config: ResourceConfig, output: Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resource-config.json").write_text(json.dumps(asdict(config), indent=2))
    cells: list[dict[str, Any]] = []
    experience: list[dict[str, Any]] = []
    partitions = {}
    for task in config.tasks:
        protocol = _protocol(config, task)
        partitions[task] = protocol.manifest()
        data = protocol.development()
        metric = "balanced_accuracy" if task == "hierarchy" else "accuracy"
        for seed in config.seeds:
            cell_index = len(cells)
            folder = output / f"cell-{cell_index:03d}"
            folder.mkdir()
            training = TrainConfig(
                steps=config.initial_steps,
                eval_every=1,
                width=config.width,
                heads=2,
                layers=1,
                seed=seed,
                batch_size=config.batch_size,
            )
            initial = train(data, training, folder / "initial").model
            save_model(initial, folder / "initial.pt")
            recorder = CandidateRecorder(folder, parent_checkpoint=folder / "initial.pt")
            evaluator = make_evaluator(initial, data)
            constraints = Constraints()
            anchor = evaluator(initial)
            baseline = _scores(initial, data)
            row: dict[str, Any] = {
                "index": cell_index,
                "task": task,
                "seed": seed,
                "data_seed": config.data_seed,
                "initial_steps": config.initial_steps,
                "initial_training_tokens": config.initial_steps
                * min(config.batch_size, len(data.train.y))
                * data.train.x.shape[1],
                "counts": {n: len(getattr(data, n).y) for n in ("train", "id", "ood")},
                "arms": {},
            }
            candidate_rows: list[dict[str, Any]] = []
            recorded_indices: set[int] = set()
            for name in ("fixed", "existing", "joint", "experience"):
                start = time.perf_counter()
                source = choose_experience(experience, task) if name == "experience" else None
                if name == "fixed":
                    continuation = replace(training, steps=config.intervention_steps)
                    completed = 0

                    def apply(
                        candidate: nn.Module, data=data, continuation=continuation, folder=folder
                    ) -> nn.Module:
                        nonlocal completed

                        def record(step: int, model: nn.Module, metrics: dict[str, Any]) -> None:
                            nonlocal completed
                            completed = step

                        return train(
                            data,
                            continuation,
                            folder / "fixed-training",
                            model=candidate,
                            callback=record,
                        ).model

                    checked = trial(
                        initial,
                        Proposal(
                            "fixed_continuation",
                            "training",
                            {
                                "steps": config.intervention_steps,
                                "optimizer_artifact": "fixed-training",
                            },
                            apply,
                        ),
                        evaluator,
                        constraints,
                        Ledger(folder / "fixed-trials.jsonl"),
                        run_id="resource-fixed",
                        step=0,
                        reference=anchor,
                        recorder=recorder,
                    )
                    retained = checked.model
                    status = "committed" if checked.accepted else "rolled_back"
                    reasons = [checked.reason]
                    fraction = None
                    budget = {
                        "planned_updates": config.intervention_steps,
                        "executed_updates": completed
                        if completed == config.intervention_steps
                        else None,
                        "completed_updates_lower_bound": completed,
                        "tokens_per_update": min(config.batch_size, len(data.train.y))
                        * data.train.x.shape[1],
                    }
                else:
                    controller_config = ControllerConfig(
                        evaluation_window=config.intervention_steps,
                        evaluation_samples=config.intervention_steps,
                        minimum_gain=1e-5,
                        protected_metrics=(
                            MetricGuard("performance.validation_score", 0.02),
                            MetricGuard("performance.ood_score", 0.02),
                        ),
                        resources=config.profile if name in ("joint", "experience") else None,
                    )
                    controller = Controller(
                        controller_config,
                        EventLog(Ledger(folder / f"{name}-events.jsonl"), name),
                        policy=_GuidedPolicy(controller_config, source),
                    )
                    observer = TrainingObserver(controller, score=metric)
                    # Actual unchanged checkpoint observations establish a plateau;
                    # these are observation ticks, not additional optimizer updates.
                    for tick in range(10):
                        observer(tick, initial, baseline)
                    assert observer.latest is not None
                    provider = MeasuredTrainingExecutor(
                        copy.deepcopy(initial),
                        data,
                        training,
                        evaluator=evaluator,
                        constraints=constraints,
                        ledger=Ledger(folder / f"{name}-trials.jsonl"),
                        output=folder / name,
                        score=metric,
                        recorder=recorder,
                        measurement_repeats=config.repeats,
                    )
                    outcome = controller.start(observer.latest, provider)
                    if outcome.status == "pending":
                        outcome = controller.finish()
                    retained, status, reasons = (
                        provider.model,
                        outcome.status,
                        list(outcome.reasons),
                    )
                    fraction = observer.latest.parameters.get("fraction")
                    budget = provider.last_budget or {
                        "planned_updates": config.intervention_steps,
                        "executed_updates": 0,
                        "completed_updates_lower_bound": 0,
                        "tokens_per_update": min(config.batch_size, len(data.train.y))
                        * data.train.x.shape[1],
                    }
                checkpoint = folder / f"{name}-retained.pt"
                save_model(retained, checkpoint)
                scores = _scores(retained, data)
                resource = measure_resources(
                    retained, data.id.x[: config.batch_size], repeats=config.repeats
                )
                measured = {**resource["metrics"], **_capability(scores, metric)}
                if budget["executed_updates"] is not None:
                    measured.update(
                        {
                            "resources.training_updates": budget["executed_updates"],
                            "resources.training_tokens": budget["executed_updates"]
                            * budget["tokens_per_update"],
                        }
                    )
                feasibility = config.profile.assess(measured)
                row["arms"][name] = {
                    "status": status,
                    "reasons": reasons,
                    "fraction": fraction,
                    "checkpoint": checkpoint.relative_to(output).as_posix(),
                    "selection": scores,
                    "retained_resources": resource,
                    "retained_feasibility": feasibility,
                    "budget": budget,
                    "wall_seconds": time.perf_counter() - start,
                    "experience_source": source,
                    "target_remeasured": True,
                }
                for attempt in recorder.records():
                    if (
                        attempt["event"] != "completed"
                        or attempt["candidate_index"] in recorded_indices
                    ):
                        continue
                    recorded_indices.add(attempt["candidate_index"])
                    attempt.update(
                        {
                            "arm": name,
                            "budget": budget,
                            "controller_status": status,
                            "selection_resources": None,
                            "measurement_missing_reason": None,
                        }
                    )
                    if attempt["checkpoint"] is not None:
                        try:
                            candidate = load_model(folder / attempt["checkpoint"])
                            probe = measure_resources(
                                candidate, data.id.x[: config.batch_size], repeats=config.repeats
                            )
                            candidate_metrics = {
                                **probe["metrics"],
                                **_capability(_scores(candidate, data), metric),
                            }
                            if budget["executed_updates"] is not None:
                                candidate_metrics.update(
                                    {
                                        "resources.training_updates": budget["executed_updates"],
                                        "resources.training_tokens": budget["executed_updates"]
                                        * budget["tokens_per_update"],
                                    }
                                )
                            attempt["selection_resources"] = probe
                            attempt["resource_feasibility"] = config.profile.assess(
                                candidate_metrics
                            )
                        except (RuntimeError, ValueError, FloatingPointError) as exc:
                            attempt["measurement_missing_reason"] = str(exc)
                    else:
                        attempt["measurement_missing_reason"] = "candidate has no checkpoint"
                    candidate_rows.append(attempt)
            row["feasible_resource_ranking"] = sorted(
                [
                    name
                    for name, arm in row["arms"].items()
                    if arm["status"] == "committed"
                    and arm["retained_feasibility"]["joint_feasible"]
                ],
                key=lambda name: row["arms"][name]["retained_resources"]["metrics"][
                    "resources.latency_ms"
                ],
            )
            joint = row["arms"]["joint"]
            experience.append(
                {
                    "source_cell": cell_index,
                    "task": task,
                    "seed": seed,
                    "role": "selection",
                    "profile": asdict(config.profile),
                    "hardware": joint["retained_resources"]["hardware"],
                    "status": joint["status"],
                    "joint_feasible": joint["retained_feasibility"]["joint_feasible"],
                    "fraction": joint["fraction"],
                    "objective": joint["retained_resources"]["metrics"]["resources.latency_ms"],
                    "verification_required": "fresh target measurements and all original gates",
                }
            )
            row["candidates"] = candidate_rows
            cells.append(row)
    result = {
        "schema_version": "agg.resource-experiment/1",
        "cells": cells,
        "experience": experience,
        "resource_objective": "median forward batch latency_ms",
        "unsupported": [
            "depth changes",
            "geometry routing",
            "precision execution",
            "reasoning budgets",
        ],
        "scientific_status": "bounded development comparison; audit pending",
    }
    (output / "resource-results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    freeze_run(output, partitions)
    return result


def audit_resources(output: Path, *, seed: int = 0) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**32 - 4:
        raise ValueError("valid audit seed required")
    output = Path(output)
    if (output / "resource-audit-attempt.json").exists():
        raise ValueError("resource audit already attempted")
    manifest = validate_frozen_run(output)
    config = ResourceConfig.from_dict(json.loads((output / "resource-config.json").read_text()))
    if manifest["partitions"] != {
        task: _protocol(config, task).manifest() for task in config.tasks
    }:
        raise ValueError("resource partitions differ from frozen manifest")
    with (output / "resource-audit-attempt.json").open("x") as stream:
        json.dump(
            {
                "role": "audit",
                "seed": seed,
                "manifest_sha256": file_hash(output / "selection-frozen.json"),
            },
            stream,
        )
    try:
        training = json.loads((output / "resource-results.json").read_text())
        rows = []
        for cell in training["cells"]:
            protocol = _protocol(config, cell["task"])
            private = protocol.evaluation("audit")
            baseline = load_model(output / cell["arms"]["fixed"]["checkpoint"])
            metric = "balanced_accuracy" if cell["task"] == "hierarchy" else "accuracy"

            def measure(
                path: str | None,
                *,
                private=private,
                metric=metric,
                baseline=baseline,
                protocol=protocol,
                budget=None,
            ) -> dict[str, Any]:
                if path is None:
                    return {"measurement": None, "missing_reason": "candidate has no checkpoint"}
                if path not in manifest["artifacts"]:
                    raise ValueError("model absent from frozen artifacts")
                model = load_model(output / path)
                scores = {
                    name: {
                        "score": evaluate(model, getattr(private, name))[metric],
                        "samples": len(getattr(private, name).y),
                        "metric": metric,
                        "paired_vs_fixed": _paired(
                            baseline,
                            model,
                            getattr(private, name),
                            getattr(private, name + "_clusters"),
                            protocol,
                            seed,
                        ),
                    }
                    for name in ("id", "ood")
                }
                resource = measure_resources(
                    model, private.id.x[: config.batch_size], repeats=config.repeats
                )
                measured = {
                    **resource["metrics"],
                    "performance.validation_score": scores["id"]["score"],
                    "performance.ood_score": scores["ood"]["score"],
                }
                if budget is not None and budget["executed_updates"] is not None:
                    measured.update(
                        {
                            "resources.training_updates": budget["executed_updates"],
                            "resources.training_tokens": budget["executed_updates"]
                            * budget["tokens_per_update"],
                        }
                    )
                return {
                    "scores": scores,
                    "resources": resource,
                    "joint_feasibility": config.profile.assess(measured),
                }

            candidates = {}
            for candidate in cell["candidates"]:
                path = candidate["checkpoint"]
                try:
                    candidates[str(candidate["candidate_index"])] = measure(
                        f"cell-{cell['index']:03d}/{path}" if path else None,
                        budget=candidate["budget"],
                    )
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    candidates[str(candidate["candidate_index"])] = {
                        "measurement": None,
                        "missing_reason": str(exc),
                    }
            rows.append(
                {
                    "index": cell["index"],
                    "task": cell["task"],
                    "seed": cell["seed"],
                    "arms": {
                        name: measure(arm["checkpoint"], budget=arm["budget"])
                        for name, arm in cell["arms"].items()
                    },
                    "candidates": candidates,
                }
            )
        result = {
            "schema_version": "agg.resource-audit/1",
            "cells": rows,
            "final_test": None,
            "interpretation": "independent audit; timing is descriptive on this hardware",
        }
        (output / "resource-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    except Exception as exc:
        (output / "resource-audit-failure.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        )
        raise
