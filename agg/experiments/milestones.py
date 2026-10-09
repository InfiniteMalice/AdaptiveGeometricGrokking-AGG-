"""Compose frozen measurements into diagnostics without training or selection authority."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from agg.ledger import Ledger
from agg.telemetry.milestones import Measurement, milestone_diagnostics

from .checkpoints import load_model
from .independent import file_hash, validate_frozen_run


def _future_validation(points: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for target_name, target in (("accuracy", "ood_accuracy"), ("retrieval", "required_update")):
        comparisons = {}
        for feature in ("raw_loss", *diagnostics["milestones"]):
            pairs, missing = [], []
            for index in range(len(points) - 1):
                first, later = points[index : index + 2]
                source = first["evidence"].get("train_loss", [])
                x = (
                    (source[0].value if source else None)
                    if feature == "raw_loss"
                    else (diagnostics["milestones"][feature]["history"][index]["satisfied"])
                )
                outcome = later["evidence"].get(target, [])
                y = outcome[0].value if outcome and outcome[0].verified else None
                if x is None or y is None:
                    missing.append(
                        {
                            "step": first["step"],
                            "future_step": later["step"],
                            "reason": "feature or future outcome unavailable",
                        }
                    )
                else:
                    pairs.append(
                        {
                            "step": first["step"],
                            "future_step": later["step"],
                            "feature": float(x),
                            "outcome": y,
                        }
                    )
            correlation = None
            reason: str | None = "fewer than three pairs or constant feature/outcome"
            if len(pairs) >= 3:
                x = np.array([p["feature"] for p in pairs])
                y = np.array([p["outcome"] for p in pairs])
                if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                    correlation = float(np.corrcoef(x, y)[0, 1])
                    reason = None
            comparisons[feature] = {
                "pairs": pairs,
                "missing": missing,
                "pearson": correlation,
                "missing_reason": reason,
                "interval": None,
                "interval_reason": "dependent checkpoints; no independent-run uncertainty",
            }
        result[target_name] = comparisons
    return result


def milestone_report(output: Path) -> dict[str, Any]:
    output = Path(output)
    if (output / "milestone-attempt.json").exists() or (output / "milestone-report.json").exists():
        raise ValueError("milestone report already attempted")
    manifest = validate_frozen_run(output)
    manifest_hash = file_hash(output / "selection-frozen.json")
    causal = json.loads((output / "causal-report.json").read_text())
    if (
        causal.get("schema_version") != "agg.causal-report/1"
        or causal.get("manifest_sha256") != manifest_hash
        or causal.get("role") != "audit"
    ):
        raise ValueError("milestones require a matching causal audit report")
    audit = None
    if (output / "audit-report.json").exists():
        audit = json.loads((output / "audit-report.json").read_text())
        if (
            audit.get("schema_version")
            not in {"agg.independent-report/1", "agg.independent-report/2"}
            or audit.get("manifest_sha256") != manifest_hash
            or audit.get("role") != "audit"
        ):
            raise ValueError("milestones require a matching independent audit report")
    source_files = ["causal-report.json"] + (["audit-report.json"] if audit else [])
    sources = {name: file_hash(output / name) for name in source_files}
    attempt = {
        "schema_version": "agg.milestone-attempt/1",
        "manifest_sha256": manifest_hash,
        "sources": sources,
        "role": "audit",
    }
    with (output / "milestone-attempt.json").open("x", encoding="utf-8") as stream:
        json.dump(attempt, stream, indent=2)
    try:
        training = {row["step"]: row for row in Ledger(output / "training/metrics.jsonl").read()}
        config = json.loads((output / "experiment.json").read_text())
        points = []
        for row in causal["trajectory"]:
            step = row["step"]
            evidence: dict[str, list[Measurement]] = {}

            def add(
                name: str,
                value: float | None,
                source: str,
                reason: str | None = None,
                destination=evidence,
            ):
                destination[name] = [
                    Measurement(
                        value,
                        source,
                        True,
                        reason or "required observation unavailable" if value is None else None,
                    )
                ]

            if step in training:
                metrics = training[step]["train"]
                add("train_loss", metrics["loss"], f"training/metrics.jsonl#{step}/loss")
                score = "balanced_accuracy" if config["task"] == "hierarchy" else "accuracy"
                add("train_accuracy", metrics[score], f"training/metrics.jsonl#{step}/{score}")
            else:
                for metric in ("train_loss", "train_accuracy"):
                    add(
                        metric,
                        None,
                        f"training/metrics.jsonl#{step}",
                        "no frozen training metrics at checkpoint",
                    )
            measured = row["measurement"]
            if measured:
                ood = measured["ood"]
                prefix = f"causal-report.json:{sources['causal-report.json']}#{step}/ood"
                add("ood_accuracy", ood["structural_accuracy"]["value"], prefix)
                for kind, mappings in {
                    "invariant": {
                        "correct_invariance": "correct_invariance",
                        "prediction_flip": "irrelevant_flip",
                    },
                    "decisive": {
                        "required_update": "required_update",
                        "prediction_flip": "relevant_flip",
                        "joint_correctness": "decisive_joint",
                    },
                }.items():
                    values = ood[kind]["metrics"]
                    for metric, name in mappings.items():
                        add(
                            name,
                            values[metric]["value"] if values else None,
                            f"{prefix}/{kind}/{metric}",
                            values[metric]["missing_reason"]
                            if values
                            else ood[kind]["missing_reason"],
                        )
                if config["task"] == "hierarchy":
                    # A majority-negative classifier must not pass the OOD accuracy threshold.
                    add(
                        "ood_accuracy",
                        ood["structural_accuracy"].get("balanced_value"),
                        prefix + "/full_stratum_balanced_accuracy",
                        ood["structural_accuracy"].get("balanced_missing_reason")
                        or "full-stratum balanced accuracy unavailable in source report",
                    )
                if config["task"] != "retrieval":
                    add("irrelevant_flip", None, prefix, "task has no distractor-context transform")
            else:
                for metric in (
                    "ood_accuracy",
                    "correct_invariance",
                    "required_update",
                    "irrelevant_flip",
                    "relevant_flip",
                    "decisive_joint",
                ):
                    add(
                        metric,
                        None,
                        f"causal-report.json:{sources['causal-report.json']}#{step}",
                        row["missing_reason"] or "causal checkpoint measurement unavailable",
                    )
            points.append({"step": step, "evidence": evidence})
        if not points:
            raise ValueError("milestones require saved checkpoint measurements")
        if audit:
            evidence = points[-1]["evidence"]
            source = f"audit-report.json:{sources['audit-report.json']}#post_selection"
            for name in ("id", "ood"):
                baseline = audit["baseline"][name]["score"]
                selected = audit["selected"][name]["score"]
                gain = (
                    selected - baseline if selected is not None and baseline is not None else None
                )
                evidence[f"compression_{name}_gain"] = [
                    Measurement(
                        gain,
                        source,
                        True,
                        "independent class score unavailable" if gain is None else None,
                    )
                ]
            baseline_ood = audit["baseline"]["ood"]["score"]
            evidence["baseline_ood"] = [
                Measurement(
                    baseline_ood,
                    source,
                    True,
                    "baseline OOD missing" if baseline_ood is None else None,
                )
            ]
            counts = []
            for filename in ("baseline-state.pt", "selected-state.pt"):
                if filename not in manifest["artifacts"]:
                    raise ValueError("compression checkpoints must be frozen")
                counts.append(sum(p.numel() for p in load_model(output / filename).parameters()))
            evidence["parameters_reduced"] = [
                Measurement(
                    float(counts[1] < counts[0]),
                    "frozen baseline/selected parameter counts; not physical bytes",
                    True,
                )
            ]
        diagnostics = milestone_diagnostics(points)
        result = {
            **attempt,
            **diagnostics,
            "future_validation": _future_validation(points, diagnostics),
            "validation_scope": "next checkpoints on withheld audit structures; descriptive",
            "compression_scope": "post-selection observation at terminal training step",
            "existing_crossing": causal["phase"]["crossing"],
            "final_test": None,
        }
        with (output / "milestone-report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result
    except Exception as exc:
        Ledger(output / "milestone-failures.jsonl").append(
            {**attempt, "reason": f"{type(exc).__name__}: {exc}"}
        )
        raise
