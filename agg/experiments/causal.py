"""Independent causal trajectories over frozen inference artifacts; no selection API."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agg.evaluation.causal import intervention_metrics
from agg.evaluation.independent import paired_cluster_comparison
from agg.ledger import Ledger
from agg.tasks.causal import InterventionPairs, intervention_pairs

from .checkpoints import load_model
from .config import ExperimentConfig
from .independent import file_hash, validate_frozen_run


def phase_checkpoints(steps: list[int], crossing: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"crossing": crossing, "proxy": True}
    onset = crossing.get("stable_crossing") if crossing else None
    confirmed = crossing.get("confirmation_step") if crossing else None
    for phase in ("before", "during", "after"):
        eligible: list[int] = []
        if onset is not None and confirmed is not None:
            if phase == "before":
                eligible = [s for s in steps if s < onset][-1:]
            elif phase == "during":
                eligible = [s for s in steps if onset <= s <= confirmed][:1]
            else:
                eligible = [s for s in steps if s > confirmed][:1]
        result[phase] = {
            "step": eligible[0] if eligible else None,
            "missing_reason": None
            if eligible
            else "no confirmed crossing or no checkpoint in this phase",
        }
    return result


@torch.no_grad()
def _predictions(model: nn.Module, x: torch.Tensor) -> list[int]:
    model.eval()
    logits = model(x)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite causal predictions")
    return logits.argmax(-1).tolist()


def _measure(model: nn.Module, pairs: InterventionPairs, seed: int) -> dict[str, Any]:
    metrics = None
    observations = None
    if len(pairs.original.y):
        before = _predictions(model, pairs.original.x)
        after = _predictions(model, pairs.transformed.x)
        original, transformed = pairs.original.y.tolist(), pairs.transformed.y.tolist()
        metrics = intervention_metrics(
            before, after, original, transformed, list(pairs.clusters), kind=pairs.kind, seed=seed
        )
        observations = {
            "source_indices": pairs.source_indices,
            "clusters": list(pairs.clusters),
            "original_labels": original,
            "transformed_labels": transformed,
            "original_predictions": before,
            "transformed_predictions": after,
        }
    return {
        "transformation": pairs.transformation,
        "definition_version": "agg.causal-pairs/1",
        "attempted": pairs.attempted,
        "missing": pairs.missing,
        "metrics": metrics,
        "observations": observations,
        "missing_reason": None if metrics else "no valid nontrivial transformations",
    }


def causal_report(output: Path, *, seed: int = 0) -> dict[str, Any]:
    from .runner import evaluation_protocol

    if type(seed) is not int or seed < 0:
        raise ValueError("causal seed must be a nonnegative integer")
    output = Path(output)
    intent = output / "causal-attempt.json"
    if intent.exists() or (output / "causal-report.json").exists():
        raise ValueError("causal audit already attempted; use a fresh preregistered run")
    manifest = validate_frozen_run(output)
    config = ExperimentConfig.from_dict(json.loads((output / "experiment.json").read_text()))
    protocol = evaluation_protocol(config)
    if protocol.manifest() != manifest["partitions"]:
        raise ValueError("causal partitions differ from frozen selection")
    attempt = {
        "schema_version": "agg.causal-attempt/1",
        "role": "audit",
        "seed": seed,
        "manifest_sha256": file_hash(output / "selection-frozen.json"),
    }
    with intent.open("x", encoding="utf-8") as stream:
        json.dump(attempt, stream, indent=2)
    try:
        audit = protocol.evaluation("audit")
        pairs = {
            name: {
                kind: intervention_pairs(
                    getattr(audit, name),
                    getattr(audit, name + "_clusters"),
                    protocol.metadata,
                    kind=kind,
                    seed=seed,
                )
                for kind in ("invariant", "decisive")
            }
            for name in ("id", "ood")
        }

        def measure(filename: str) -> dict[str, Any]:
            if filename not in manifest["artifacts"]:
                raise ValueError("checkpoint not frozen")
            model = load_model(output / filename)
            measured: dict[str, Any] = {}
            for name, cases in pairs.items():
                row = {kind: _measure(model, pair, seed) for kind, pair in cases.items()}
                inv, dec = row["invariant"]["metrics"], row["decisive"]["metrics"]
                correct = (
                    (_predictions_tensor(model, getattr(audit, name).x) == getattr(audit, name).y)
                    .int()
                    .tolist()
                )
                balanced = (
                    paired_cluster_comparison(
                        [0] * len(correct),
                        correct,
                        list(getattr(audit, name + "_clusters")),
                        seed=seed,
                        labels=getattr(audit, name).y.tolist(),
                        expected_classes=list(range(protocol.classes)),
                    )
                    if config.task == "hierarchy"
                    else None
                )
                measured[name] = {
                    **row,
                    "structural_accuracy": {
                        "value": sum(correct) / len(correct),
                        "numerator": sum(correct),
                        "denominator": len(correct),
                        "missing_reason": None,
                        "balanced_value": balanced["effect"] if balanced else None,
                        "balanced_uncertainty": balanced,
                        "balanced_missing_reason": (
                            balanced["unavailable_reason"] if balanced else "not requested for task"
                        ),
                        "uncertainty": paired_cluster_comparison(
                            [0] * len(correct),
                            correct,
                            list(getattr(audit, name + "_clusters")),
                            seed=seed,
                        ),
                    },
                    "relevant_feature_dependence": dec["prediction_flip"] if dec else None,
                    "invariant_input_dependence": inv["prediction_flip"] if inv else None,
                    "distractor_dependence": (
                        inv["prediction_flip"] if inv and config.task == "retrieval" else None
                    ),
                    "distractor_missing_reason": (
                        None
                        if inv and config.task == "retrieval"
                        else "no supported irrelevant-context intervention"
                    ),
                    "causal_sensitivity": {
                        "value": (dec["prediction_flip"]["value"] - inv["prediction_flip"]["value"])
                        if inv and dec
                        else None,
                        "invariant_denominator": inv["samples"] if inv else 0,
                        "decisive_denominator": dec["samples"] if dec else 0,
                        "uncertainty": None,
                        "uncertainty_unavailable_reason": (
                            "joint contrast resampling across differing supports not estimated"
                        ),
                        "missing_reason": None if inv and dec else "one intervention class missing",
                    },
                }
            return measured

        trajectory = []
        checkpoints = sorted(
            (int(Path(name).stem.split("-")[-1]), name)
            for name in manifest["artifacts"]
            if name.startswith("training/inference-") and name.endswith(".pt")
        )
        for step, name in checkpoints:
            measured = None
            reason = None
            try:
                measured = measure(name)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            trajectory.append(
                {
                    "step": step,
                    "checkpoint": name,
                    "measurement": measured,
                    "missing_reason": reason,
                }
            )
        crossing = (
            json.loads((output / "phase.json").read_text())["crossing"]
            if "phase.json" in manifest["artifacts"]
            else None
        )
        report = {
            **attempt,
            "schema_version": "agg.causal-report/1",
            "trajectory": trajectory,
            "trajectory_missing_reason": None if trajectory else "run predates inference snapshots",
            "selected": measure("selected-state.pt"),
            "phase": phase_checkpoints(
                [row["step"] for row in trajectory if row["measurement"] is not None], crossing
            ),
            "final_test": None,
            "interpretation": "behavioral diagnostics; no geometry causation",
            "recovery_definition": "transformed correct conditional on original wrong; no retry",
        }
        with (output / "causal-report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return report
    except Exception as exc:
        Ledger(output / "causal-failures.jsonl").append(
            {**attempt, "reason": f"{type(exc).__name__}: {exc}"}
        )
        raise


def _predictions_tensor(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_predictions(model, x))
