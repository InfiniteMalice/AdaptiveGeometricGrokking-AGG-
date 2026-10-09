"""Explicit curriculum benchmark: task-rule summaries, matched copies, lagged curation."""

import copy
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agg.consolidation import Proposal, trial
from agg.controller.curriculum import CurriculumGate, RetentionTracker
from agg.controller.events import EventLog
from agg.controller.memory import AbstractionRegistry, TransferEvidence
from agg.controller.routing import (
    AuditedEvidenceRouter,
    EvidenceValidity,
    ReferenceRouter,
    RoutingConfig,
)
from agg.evaluation import Constraints, Evaluation
from agg.evaluation.independent import paired_cluster_comparison
from agg.evaluation.model import make_evaluator
from agg.ledger import Ledger
from agg.models import TinyTransformer
from agg.tasks import Split, TaskData
from agg.tasks.lessons import CONTROLS, OPERATIONS, extract_lessons, lesson_examples
from agg.tasks.protocol import EvaluationProtocol, make_protocol
from agg.telemetry.controller import Observation
from agg.training import TrainConfig, evaluate, seed_all

from .candidates import fit_candidate
from .checkpoints import load_model, save_model
from .config import ExperimentConfig, Features
from .independent import CandidateRecorder, file_hash, freeze_run, validate_frozen_run


@dataclass(frozen=True)
class Block:
    task: str = "retrieval"
    length: int = 8
    distance: int = 2
    density: float = 0.5
    depth: int = 4
    modulus: int = 17


@dataclass(frozen=True)
class BlockConfig:
    blocks: tuple[Block, ...] = (
        Block(density=0.2),
        Block(length=10, distance=3, density=0.8),
        Block(length=12, distance=4),
        Block("modular"),
    )
    data_seed: int = 101
    model_seed: int = 17
    samples: int = 16
    steps: int = 2
    batch_size: int = 16
    width: int = 8
    heads: int = 2
    bootstrap_seed: int = 31

    def __post_init__(self) -> None:
        if (
            not self.blocks
            or len(self.blocks) > 12
            or any(
                type(v) is not int or v < 1
                for v in (self.samples, self.steps, self.batch_size, self.width, self.heads)
            )
            or self.width % self.heads
            or self.samples < 10
        ):
            raise ValueError(
                "bounded blocks and positive compatible model/training budgets required"
            )
        if any(
            type(v) is not int or not 0 <= v < 2**32 - 4
            for v in (self.data_seed, self.model_seed, self.bootstrap_seed)
        ):
            raise ValueError("block seeds must be integers in [0,2**32-4)")
        moduli = {b.modulus for b in self.blocks if b.task == "modular"}
        if len(moduli) > 1:
            raise ValueError("keep modulus fixed to preserve modular structural roles")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BlockConfig":
        raw = dict(values)
        if "blocks" in raw:
            raw["blocks"] = tuple(Block(**b) for b in raw["blocks"])
        return cls(**raw)


def _protocols(config: BlockConfig) -> list[EvaluationProtocol]:
    return [
        make_protocol(
            b.task,
            seed=config.data_seed,
            modulus=b.modulus,
            depth=b.depth,
            samples=config.samples,
            length=b.length,
            distance=b.distance,
            density=b.density,
        )
        for b in config.blocks
    ]


def _encoding(
    protocols: list[EvaluationProtocol],
) -> tuple[dict[str, int], int, int, int, dict[str, int]]:
    sizes: dict[str, int] = {}
    for p in protocols:
        task = p.metadata["task"]
        sizes[task] = max(sizes.get(task, 0), p.vocab_size)
    offsets = {}
    total = 1  # Token zero is only global left padding.
    for task, size in sorted(sizes.items()):
        offsets[task] = total
        total += size
    length = max(p.development().train.x.shape[1] for p in protocols)
    queries = (
        {"hierarchy": offsets["hierarchy"] + sizes["hierarchy"] - 1} if "hierarchy" in sizes else {}
    )
    return offsets, total, max(p.classes for p in protocols), length, queries


def _encode(split: Split, offset: int, length: int, *, query_token: int | None = None) -> Split:
    padding = torch.zeros((len(split.y), length - split.x.shape[1]), dtype=torch.long)
    tokens = split.x + offset
    if query_token is not None:
        tokens[:, -1] = query_token
    return Split(torch.cat((padding, tokens), 1), split.y.clone())


def _data(
    protocol: EvaluationProtocol,
    offset: int,
    length: int,
    vocab: int,
    classes: int,
    query_token: int | None = None,
) -> TaskData:
    raw = protocol.development()
    return TaskData(
        _encode(raw.train, offset, length, query_token=query_token),
        _encode(raw.id, offset, length, query_token=query_token),
        _encode(raw.ood, offset, length, query_token=query_token),
        vocab,
        classes,
        dict(raw.metadata),
    )


def _correct(model: nn.Module, split: Split) -> list[int]:
    model.eval()
    with torch.no_grad():
        logits = model(split.x)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("nonfinite block predictions")
        return (logits.argmax(-1) == split.y).int().tolist()


def _paired(
    before: nn.Module,
    after: nn.Module,
    split: Split,
    clusters: tuple[str, ...],
    protocol: EvaluationProtocol,
    seed: int,
) -> dict[str, Any]:
    hierarchy = protocol.metadata["task"] == "hierarchy"
    return paired_cluster_comparison(
        _correct(before, split),
        _correct(after, split),
        list(clusters),
        seed=seed,
        labels=split.y.tolist() if hierarchy else None,
        expected_classes=list(range(protocol.classes)) if hierarchy else None,
    )


@dataclass(frozen=True)
class _RetentionConstraints(Constraints):
    additional: Callable[[], list[str]] = field(default=lambda: [], compare=False, repr=False)

    def violations(self, baseline: Evaluation, candidate: Evaluation) -> list[str]:
        return super().violations(baseline, candidate) + self.additional()


@dataclass
class _BlockExecution:
    index: int
    config: BlockConfig
    parent: dict[str, nn.Module]
    data: TaskData
    raw: TaskData
    offset: int
    query_token: int | None
    length: int
    anchors: dict[str, list[tuple[Callable[[nn.Module], Evaluation], Evaluation]]]
    recorder: CandidateRecorder
    ledger: Ledger
    output: Path
    raw_history: list[TaskData]
    models: dict[str, nn.Module]
    attempted: dict[str, nn.Module]
    attempt_rows: dict[str, dict[str, Any]]

    def execute(
        self,
        name: str,
        lineage: str,
        ops: list[str],
        *,
        control: str | None = None,
        replay: bool = False,
        shadow: bool = True,
    ) -> dict[str, Any]:
        source_model = self.parent[lineage]
        current_evaluator = make_evaluator(source_model, self.data)
        extra: list[str] = []

        def evaluator(candidate: nn.Module) -> Evaluation:
            extra.clear()
            for scope, (observe, reference) in enumerate(self.anchors[lineage]):
                extra.extend(
                    f"retention block {scope}: {reason}"
                    for reason in Constraints().violations(reference, observe(candidate))
                )
            return current_evaluator(candidate)

        details: list[dict[str, Any]] = []
        executed_updates = 0
        fit_seconds = None

        def optimizer_observer(payload):
            nonlocal executed_updates
            states = payload["optimizer"]["state"].values()
            executed_updates = max((int(s["step"]) for s in states if "step" in s), default=0)
            self.recorder.record_optimizer(payload)

        def apply(student: nn.Module) -> nn.Module:
            nonlocal fit_seconds
            train_split = self.raw.train
            for operation in ops:
                train_split, detail = lesson_examples(
                    train_split,
                    self.raw.metadata,
                    operation,
                    seed=self.config.model_seed + self.index,
                    control=control,
                )
                details.append(detail)
            encoded = _encode(train_split, self.offset, self.length, query_token=self.query_token)
            past_examples = {
                tuple(row)
                for previous in self.raw_history
                for row in torch.cat((previous.train.x, previous.train.y[:, None]), 1).tolist()
            }
            generated = torch.cat((encoded.x, encoded.y[:, None]), 1).tolist()
            distinctiveness = (
                sum(tuple(row) not in past_examples for row in generated) / len(generated)
                if past_examples
                else None
            )
            for detail in details:
                detail["information_preservation_proxy"] = detail["oracle_agreement"]
                detail["contextual_distinctiveness"] = distinctiveness
                detail["distinctiveness_definition"] = (
                    "fraction of input-label rows absent from earlier training blocks"
                )
                detail["distinctiveness_missing_reason"] = (
                    None if past_examples else "no earlier block"
                )
            if replay and self.raw_history:
                past = Split(
                    torch.cat([d.train.x for d in self.raw_history]),
                    torch.cat([d.train.y for d in self.raw_history]),
                )
                count = len(encoded.y) // 2
                draw = torch.arange(count) % len(past.y)
                encoded.x[:count], encoded.y[:count] = past.x[draw], past.y[draw]
            fitting = replace(self.data, train=encoded)
            cfg = ExperimentConfig(
                training=TrainConfig(
                    steps=self.config.steps,
                    width=self.config.width,
                    heads=self.config.heads,
                    layers=1,
                    batch_size=self.config.batch_size,
                    seed=self.config.model_seed + self.index,
                ),
                features=Features.baseline(),
                dimensions=(self.config.width,),
                candidate_steps=self.config.steps,
            )
            started = time.perf_counter()
            try:
                fitted = fit_candidate(
                    student, source_model, fitting, cfg, False, optimizer_observer
                )
            finally:
                fit_seconds = time.perf_counter() - started
            self.attempted[name] = fitted
            return fitted

        result = trial(
            source_model,
            Proposal(
                "block_" + name.split(":")[0],
                "model",
                {
                    "steps": self.config.steps,
                    "block": self.index,
                    "name": name,
                    "shadow": shadow,
                    "operations": ops,
                    "control": control,
                    "replay": replay,
                },
                apply,
            ),
            evaluator,
            _RetentionConstraints(additional=lambda: list(extra)),
            self.ledger,
            run_id="block-learning",
            step=self.index,
            recorder=self.recorder,
        )
        completed = self.recorder.records()[-1]
        outcome = {
            "candidate_index": completed["candidate_index"],
            "status": completed["status"],
            "reason": result.reason,
            "selection_before": asdict(result.before),
            "selection_candidate": asdict(result.after) if result.after else None,
            "selection_retained": asdict(
                result.after if result.accepted and result.after else result.before
            ),
            "checkpoint": completed.get("checkpoint"),
            "feasible": result.accepted,
            "shadow": shadow,
            "operations": ops,
            "knowledge_eligible": control is None,
            "transformations": details,
            "fit_and_snapshot_seconds": fit_seconds,
            "budget": {
                "planned_updates": self.config.steps,
                "executed_updates": executed_updates,
                "batch_size": min(self.config.batch_size, len(self.data.train.y)),
                "tokens_per_update": min(self.config.batch_size, len(self.data.train.y))
                * self.length,
            },
            "oracle_valid": len(details) == len(ops)
            and all(d["oracle_agreement"] == 1 for d in details),
        }
        if not shadow:
            self.models[lineage] = result.model
            checkpoint = f"block-{self.index}-{lineage}.pt"
            save_model(result.model, self.output / checkpoint)
            outcome["retained_checkpoint"] = checkpoint
        self.attempt_rows[name] = outcome
        return outcome


def run_blocks(config: BlockConfig, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output)
    protocols = _protocols(config)
    offsets, vocab, classes, length, queries = _encoding(protocols)
    output.mkdir(parents=True, exist_ok=False)
    (output / "block-config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    seed_all(config.model_seed)
    initial = TinyTransformer(
        vocab, classes, width=config.width, heads=config.heads, layers=1, max_length=length
    )
    models: dict[str, nn.Module] = {
        name: copy.deepcopy(initial) for name in ("baseline", "summary", "replay")
    }
    events = EventLog(Ledger(output / "events.jsonl"), "block-learning")
    registry = AbstractionRegistry(events=events)
    router = AuditedEvidenceRouter(
        registry, ReferenceRouter(RoutingConfig(mode="full_history")), events
    )
    gate = CurriculumGate(dict.fromkeys(range(len(protocols)), 0.8), events)
    retention = {name: RetentionTracker() for name in models}
    anchors: dict[str, list[tuple[Callable[[nn.Module], Evaluation], Evaluation]]] = {
        n: [] for n in models
    }
    recorder = CandidateRecorder(output, parent_checkpoint=None)
    ledger = Ledger(output / "ledger.jsonl")
    raw_history: list[TaskData] = []
    rows: list[dict[str, Any]] = []

    for index, protocol in enumerate(protocols):
        task = protocol.metadata["task"]
        offset = offsets[task]
        data = _data(protocol, offset, length, vocab, classes, queries.get(task))
        raw = protocol.development()
        selection = protocol.evaluation("selection")
        parent = {name: copy.deepcopy(model) for name, model in models.items()}
        eligible: list[str] = []
        invalid: list[str] = []
        profiles: dict[str, dict[str, float]] = {}
        for lesson in registry.snapshot():
            if lesson.block != task or lesson.status != "active":
                continue
            _, validation = lesson_examples(
                raw.train,
                raw.metadata,
                lesson.provenance["operation"],
                seed=config.model_seed + index,
            )
            (eligible if validation["oracle_agreement"] == 1 else invalid).append(lesson.id)
        decision = router.route(
            Observation(index),
            block=task,
            context={"task": task},
            validity=EvidenceValidity(
                tuple(eligible), tuple(invalid), source="current-block exact oracle"
            ),
            profiles=profiles,
        )
        active_ids = list(decision.selected_ids)
        for lesson in registry.snapshot():
            registry.record_applicability(
                lesson.id,
                should_apply=lesson.block == task,
                selected=lesson.id in active_ids,
                step=index,
            )
        operations = list(
            dict.fromkeys(registry.get(key).provenance["operation"] for key in active_ids)
        )
        block: dict[str, Any] = {
            "index": index,
            "task": task,
            "active_lesson_ids": active_ids,
            "redundant_activations": len(active_ids) - len(operations),
            "invalid_lesson_ids": invalid,
            "lineages": {},
            "controls": {},
            "transfer": [],
            "scope_switch": index > 0 and protocols[index - 1].metadata["task"] != task,
        }
        attempted: dict[str, nn.Module] = {}
        attempt_rows: dict[str, dict[str, Any]] = {}

        execute = _BlockExecution(
            index=index,
            config=config,
            parent=parent,
            data=data,
            raw=raw,
            offset=offset,
            query_token=queries.get(task),
            length=length,
            anchors=anchors,
            recorder=recorder,
            ledger=ledger,
            output=output,
            raw_history=raw_history,
            models=models,
            attempted=attempted,
            attempt_rows=attempt_rows,
        ).execute

        for name in models:
            block["lineages"][name] = execute(
                name,
                name,
                operations if name == "summary" else [],
                replay=name == "replay",
                shadow=False,
            )
        # Same-parent references are separate from the independently evolving no-summary lineage.
        for name, outcome in block["lineages"].items():
            before = outcome["selection_before"]["ood_accuracy"]
            after = outcome["selection_retained"]["ood_accuracy"]
            gain = after - before
            previous = rows[-1]["lineages"][name] if rows else None
            outcome["learning_gain"] = gain
            outcome["change_in_learning_gain"] = (
                gain - previous["learning_gain"] if previous else None
            )
            outcome["block_accuracy_change"] = (
                after - previous["selection_retained"]["ood_accuracy"] if previous else None
            )
            outcome["headroom_normalized_gain"] = gain / (1 - before) if before < 1 else None
            outcome["curve_note"] = (
                "contexts differ across blocks; changes are not significance claims"
            )
        execute("no_lessons", "summary", [])
        pending = [
            a
            for a in registry.snapshot()
            if a.block == task
            and a.status in {"active", "experimental"}
            and a.provenance["source_block"] < index
        ]
        for lesson in pending:
            operation = lesson.provenance["operation"]
            alone_name, marginal_name = f"alone:{lesson.id}", f"marginal:{lesson.id}"
            alone = execute(alone_name, "summary", [operation])
            marginal_reference = "summary"
            other_operations = operations
            if lesson.id in active_ids:
                marginal_reference = f"without:{lesson.id}"
                other_operations = [op for op in operations if op != operation]
                execute(marginal_reference, "summary", other_operations)
            marginal = execute(
                marginal_name, "summary", list(dict.fromkeys([*other_operations, operation]))
            )
            standalone_stats = marginal_stats = None
            if all(
                name in attempted
                for name in (alone_name, marginal_name, "no_lessons", marginal_reference)
            ):
                standalone_stats = _paired(
                    attempted["no_lessons"],
                    attempted[alone_name],
                    data.ood,
                    selection.ood_clusters,
                    protocol,
                    config.bootstrap_seed,
                )
                marginal_stats = _paired(
                    attempted[marginal_reference],
                    attempted[marginal_name],
                    data.ood,
                    selection.ood_clusters,
                    protocol,
                    config.bootstrap_seed,
                )
            evidence = TransferEvidence(
                index,
                standalone_stats["effect"] if standalone_stats else None,
                marginal_stats["effect"] if marginal_stats else None,
                marginal_stats["interval"][0]
                if marginal_stats and marginal_stats["interval"]
                else None,
                f"block-{index}/selection/{lesson.id}",
                alone["feasible"] and marginal["feasible"],
                alone["oracle_valid"] and marginal["oracle_valid"],
            )
            registry.record_transfer(lesson.id, evidence)
            metric = "balanced_accuracy" if task == "hierarchy" else "accuracy"
            baseline_score = evaluate(attempted.get("no_lessons", parent["summary"]), data.ood)[
                metric
            ]
            headroom = 1 - baseline_score
            block["transfer"].append(
                {
                    "lesson_id": lesson.id,
                    "operation": operation,
                    "marginal_reference": marginal_reference,
                    "standalone": standalone_stats,
                    "marginal": marginal_stats,
                    "standalone_attempt": alone_name,
                    "marginal_attempt": marginal_name,
                    "headroom": headroom,
                    "headroom_normalized_gain": evidence.standalone / headroom
                    if evidence.standalone is not None and headroom > 0
                    else None,
                    "normalized_missing_reason": None
                    if evidence.standalone is not None and headroom > 0
                    else "missing transfer or saturated reference",
                }
            )
        for control in CONTROLS:
            control_operation = OPERATIONS[task][0]
            if control == "out_of_scope":
                control_operation = OPERATIONS["modular" if task != "modular" else "retrieval"][0]
            block["controls"][control] = execute(
                "control_" + control, "summary", [control_operation], control=control
            )
        scores = {}
        for name, model in models.items():
            all_data = [*raw_history, data]
            for scope, previous in enumerate(all_data):
                metric = (
                    "balanced_accuracy" if previous.metadata["task"] == "hierarchy" else "accuracy"
                )
                score = evaluate(model, previous.ood)[metric]
                retention[name].record(str(scope), index, score)
                if name == "summary":
                    scores[scope] = score
            observe = make_evaluator(model, data)
            anchors[name].append((observe, observe(model)))
        block["curriculum_recommendation"] = asdict(gate.evaluate(index, scores, step=index))
        block["schedule_policy"] = (
            "fixed preregistered comparison; gate recommendation is diagnostic"
        )
        with torch.no_grad():
            models["summary"].eval()
            predictions = models["summary"](data.train.x).argmax(-1).tolist()
        prefix = f"block-{index}"
        for i, (x, y, prediction) in enumerate(
            zip(raw.train.x.tolist(), raw.train.y.tolist(), predictions, strict=True)
        ):
            Ledger(output / "source-episodes.jsonl").append(
                {
                    "id": f"{prefix}:episode:{i}",
                    "block": index,
                    "task": task,
                    "input": x,
                    "label": y,
                    "prediction": prediction,
                    "correct": prediction == y,
                    "oracle": "agg.tasks.causal/1",
                }
            )
        new_lessons = extract_lessons(
            raw.train,
            raw.metadata,
            source_block=index,
            episode_prefix=prefix,
            predictions=predictions,
            seed=config.model_seed + index,
        )
        for lesson in new_lessons:
            redundant = [
                a.id
                for a in registry.snapshot()
                if a.provenance["signature"] == lesson.provenance["signature"]
            ]
            registry.add(
                replace(lesson, provenance={**lesson.provenance, "redundant_with": redundant}),
                step=index,
            )
        block["attempts"] = attempt_rows
        block["new_lesson_ids"] = [a.id for a in new_lessons]
        block["retention"] = {
            name: {scope: tracker.summary(scope) for scope in tracker.curves}
            for name, tracker in retention.items()
        }
        rows.append(block)
        raw_history.append(data)  # Replay comparator only; knowledge remains in the registry.
    save_model(models["summary"], output / "selected-state.pt")
    (output / "abstractions.json").write_text(
        json.dumps(registry.to_dict(), indent=2, allow_nan=False)
    )
    attempts = [attempt for row in rows for attempt in row["attempts"].values()]
    result = {
        "schema_version": "agg.block-learning/1",
        "blocks": rows,
        "encoding": {
            "offsets": offsets,
            "query_tokens": queries,
            "vocab_size": vocab,
            "classes": classes,
            "padded_length": length,
        },
        "candidate_count": sum(r["event"] == "completed" for r in recorder.records()),
        "executed_training_updates": sum(a["budget"]["executed_updates"] for a in attempts),
        "executed_training_tokens": sum(
            a["budget"]["executed_updates"] * a["budget"]["tokens_per_update"] for a in attempts
        ),
        "attempt_status_counts": {
            status: sum(a["status"] == status for a in attempts)
            for status in ("accepted", "rejected", "failed", "reference_failed")
        },
        "run_wall_seconds": time.perf_counter() - started,
        "wall_scope": "curriculum through summaries, excluding freeze and audit",
        "registry_serialized_bytes": (output / "abstractions.json").stat().st_size,
        "summary_model_tokens": 0,
        "summary_encoding": "executable templates; no language model",
        "compute_flops": None,
        "compute_missing_reason": "not instrumented",
        "scientific_status": "development transfer; independent audit pending",
    }
    (output / "block-learning.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    freeze_run(output, {str(i): p.manifest() for i, p in enumerate(protocols)})
    return result


def audit_blocks(output: Path, *, seed: int = 0) -> dict[str, Any]:
    output = Path(output)
    if type(seed) is not int or not 0 <= seed < 2**32 - 4:
        raise ValueError("audit seed must be an integer in [0,2**32-4)")
    if (output / "block-audit-attempt.json").exists() or (output / "block-audit.json").exists():
        raise ValueError("block audit already attempted")
    manifest = validate_frozen_run(output)
    config = BlockConfig.from_dict(json.loads((output / "block-config.json").read_text()))
    protocols = _protocols(config)
    if manifest["partitions"] != {str(i): p.manifest() for i, p in enumerate(protocols)}:
        raise ValueError("block partitions differ from frozen selection")
    training = json.loads((output / "block-learning.json").read_text())
    offsets, _, _, length, queries = _encoding(protocols)
    intent = {
        "schema_version": "agg.block-audit-attempt/1",
        "role": "audit",
        "seed": seed,
        "manifest_sha256": file_hash(output / "selection-frozen.json"),
    }
    with (output / "block-audit-attempt.json").open("x", encoding="utf-8") as stream:
        json.dump(intent, stream, indent=2)
    try:
        results = []
        for row, protocol in zip(training["blocks"], protocols, strict=True):
            private = protocol.evaluation("audit")
            task = protocol.metadata["task"]
            splits = {
                name: _encode(
                    getattr(private, name), offsets[task], length, query_token=queries.get(task)
                )
                for name in ("id", "ood")
            }
            baseline = load_model(output / row["lineages"]["baseline"]["retained_checkpoint"])

            def measure(
                checkpoint: str | None,
                *,
                splits=splits,
                task=task,
                baseline=baseline,
                private=private,
                protocol=protocol,
            ) -> dict[str, Any]:
                if checkpoint is None:
                    return {"measurement": None, "missing_reason": "candidate has no checkpoint"}
                if checkpoint not in manifest["artifacts"]:
                    raise ValueError("audit model not frozen")
                model = load_model(output / checkpoint)
                scores = {}
                for name, split in splits.items():
                    metric = "balanced_accuracy" if task == "hierarchy" else "accuracy"
                    value = evaluate(model, split)[metric]
                    scores[name] = {
                        "score": value,
                        "metric": metric,
                        "samples": len(split.y),
                        "paired_vs_baseline": _paired(
                            baseline,
                            model,
                            split,
                            getattr(private, name + "_clusters"),
                            protocol,
                            seed,
                        ),
                    }
                return scores

            candidates = {}
            for name, attempt in row["attempts"].items():
                try:
                    candidates[name] = measure(attempt["checkpoint"])
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    candidates[name] = {
                        "measurement": None,
                        "missing_reason": f"{type(exc).__name__}: {exc}",
                    }
            retained = {}
            for scope, earlier in enumerate(protocols[: row["index"] + 1]):
                old_private = earlier.evaluation("audit")
                old_task = earlier.metadata["task"]
                old_splits = {
                    name: _encode(
                        getattr(old_private, name),
                        offsets[old_task],
                        length,
                        query_token=queries.get(old_task),
                    )
                    for name in ("id", "ood")
                }
                retained[str(scope)] = {
                    name: measure(
                        value["retained_checkpoint"],
                        splits=old_splits,
                        task=old_task,
                        private=old_private,
                        protocol=earlier,
                    )
                    for name, value in row["lineages"].items()
                }
            results.append(
                {
                    "index": row["index"],
                    "task": task,
                    "lineages": {
                        name: measure(value["retained_checkpoint"])
                        for name, value in row["lineages"].items()
                    },
                    "candidates": candidates,
                    "retention": retained,
                    "selection_transfer": row["transfer"],
                }
            )
        result = {
            **intent,
            "schema_version": "agg.block-audit/1",
            "blocks": results,
            "candidate_count": training["candidate_count"],
            "final_test": None,
            "scientific_status": "independent measurements; no scientific conclusion from smoke",
        }
        with (output / "block-audit.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result
    except Exception as exc:
        Ledger(output / "block-audit-failures.jsonl").append(
            {**intent, "reason": f"{type(exc).__name__}: {exc}"}
        )
        raise
