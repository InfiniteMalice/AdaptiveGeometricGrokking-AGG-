"""Append-only candidate accounting and immutable inputs for independent reports."""

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agg.consolidation import Proposal, state_hash
from agg.ledger import Ledger

from .checkpoints import load_model, save_model
from .config import ExperimentConfig


def selection_comparison(config: ExperimentConfig) -> dict[str, ExperimentConfig]:
    """Matched training/search configs; audit reporting is a separate operation.

    The first and third arms intentionally have identical selection. Report audit
    for the third to expose the reporting gap without changing the chosen model.
    Larger selection increases evaluation cost, which must be reported separately.
    """
    from dataclasses import replace

    if not config.independent_evaluation or config.task != "retrieval":
        raise ValueError("comparison requires independent retrieval evaluation")
    base = replace(config, selection_gain_floor=None)
    count = base.selection_samples if base.selection_samples is not None else base.samples
    return {
        "existing_selection": base,
        "larger_selection": replace(base, selection_samples=2 * count),
        "audit_reporting": replace(base),
        "conservative": replace(base, selection_gain_floor=0.01),
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateRecorder:
    def __init__(self, output: Path, *, parent_checkpoint: Path | None):
        self.output = Path(output)
        self.ledger = Ledger(self.output / "candidate-attempts.jsonl")
        if self.ledger.path.exists():
            raise ValueError("candidate accounting refuses to append to an existing run")
        self.parent_checkpoint = parent_checkpoint
        self._attempts: dict[int, dict[str, Any]] = {}
        self._started: dict[int, float] = {}

    def records(self) -> list[dict[str, Any]]:
        return self.ledger.read()

    def record_optimizer(self, payload: dict[str, Any]) -> None:
        if not self._attempts:
            raise ValueError("optimizer state requires an active candidate attempt")
        index = len(self._attempts)
        name = f"candidate-{index:04d}-optimizer.pt"
        torch.save(payload, self.output / name)
        self._attempts[index].update(
            {
                "optimizer_state": name,
                "optimizer_state_sha256": file_hash(self.output / name),
                "optimizer_state_reason": "actual fresh optimizer state after fitting attempt",
            }
        )

    def begin(self, model: nn.Module, proposal: Proposal, *, run_id: str, step: int) -> int:
        index = len(self._attempts) + 1
        parent_name = f"candidate-{index:04d}-parent.pt"
        save_model(model, self.output / parent_name)
        record = {
            "schema_version": "agg.candidate-attempt/1",
            "event": "proposed",
            "candidate_index": index,
            "selection_reuse_cycle": index,
            "run_id": run_id,
            "step": step,
            "family": proposal.kind,
            "component": proposal.component,
            "proposal": copy.deepcopy(proposal.state),
            "parent_state": state_hash(model),
            "parent_checkpoint": parent_name,
            "parent_checkpoint_sha256": file_hash(self.output / parent_name),
            "original_training_checkpoint": (
                str(self.parent_checkpoint) if self.parent_checkpoint else None
            ),
            "original_training_checkpoint_sha256": (
                file_hash(self.parent_checkpoint) if self.parent_checkpoint else None
            ),
            "optimizer": "fresh AdamW" if "steps" in proposal.state else "no optimizer",
            "optimizer_state": None,
            "optimizer_state_reason": "not yet executed; parent optimizer is never mutated",
            "training_updates_budget": proposal.state.get("steps", 0),
            "parameters_before": sum(p.numel() for p in model.parameters()),
            "compute_flops": None,
            "compute_reason": "not instrumented",
        }
        self.ledger.append(record)  # Persist proposal before execution or evaluation.
        self._attempts[index] = record
        self._started[index] = time.perf_counter()
        return index

    def finish(self, attempt: int, record: dict[str, Any], candidate: nn.Module | None) -> None:
        checkpoint = None
        artifact_error = None
        if candidate is not None:
            checkpoint = f"candidate-{attempt:04d}.pt"
            try:
                save_model(candidate, self.output / checkpoint)
            except (ValueError, TypeError) as exc:
                checkpoint = None
                artifact_error = f"{type(exc).__name__}: {exc}"
        status = "accepted" if record["accepted"] else "rejected"
        if record["reason"].startswith("candidate error:"):
            status = "failed"
        self.ledger.append(
            {
                **self._attempts[attempt],
                "event": "completed",
                "status": status,
                "reason": record["reason"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": file_hash(self.output / checkpoint) if checkpoint else None,
                "artifact_error": artifact_error,
                "candidate_state": record["state_candidate"],
                "selection_before": record["evaluation_before"],
                "selection_after": record["evaluation_after"],
                "protected_anchor": record["evaluation_reference"],
                "parameters_after": sum(p.numel() for p in candidate.parameters())
                if candidate
                else None,
                "wall_seconds": time.perf_counter() - self._started[attempt],
                "wall_scope": "trial, telemetry and snapshot serialization; not inference latency",
                "audit": None,
                "final": None,
            }
        )
        if artifact_error and record["accepted"]:
            raise ValueError(f"accepted candidate checkpoint unavailable: {artifact_error}")

    def fail(self, attempt: int, error: Exception) -> None:
        self.ledger.append(
            {
                **self._attempts[attempt],
                "event": "completed",
                "status": "reference_failed",
                "reason": f"{type(error).__name__}: {error}",
                "checkpoint": None,
                "selection_after": None,
                "audit": None,
                "final": None,
                "wall_seconds": time.perf_counter() - self._started[attempt],
            }
        )


def freeze_run(output: Path, partitions: dict[str, Any]) -> dict[str, Any]:
    """Called only after selection. Hash every run artifact before any audit report."""
    path = output / "selection-frozen.json"
    if path.exists():
        raise ValueError("selection is already frozen")
    artifacts = {
        p.relative_to(output).as_posix(): file_hash(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema_version": "agg.frozen-selection/1",
        "artifacts": artifacts,
        "partitions": partitions,
        "selection_closed": True,
        "selected_checkpoint": "selected-state.pt",
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def validate_frozen_run(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "selection-frozen.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "agg.frozen-selection/1" or not manifest.get(
        "selection_closed"
    ):
        raise ValueError("unsupported or unfrozen selection manifest")
    root = output.resolve()
    for name, expected in manifest["artifacts"].items():
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError("missing or external frozen artifact")
        if file_hash(artifact) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    return manifest


def report_run(
    output: Path,
    *,
    role: str = "audit",
    authorization: str | None = None,
    manifest_sha256: str | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate frozen checkpoints. Never return evidence to the selector.

    Final authorization names the research release and binds the manifest hash.
    It is an explicit trusted-host attestation, not an authentication mechanism.
    Each report refuses overwrite; a failed attempt remains visible in the log.
    """
    from agg.evaluation.independent import paired_cluster_comparison
    from agg.training import evaluate

    from .config import ExperimentConfig
    from .runner import evaluation_protocol

    output = Path(output)
    if role not in {"audit", "final"}:
        raise ValueError("independent report role must be audit or final")
    manifest_path = output / "selection-frozen.json"
    if role == "final" and (
        not isinstance(authorization, str) or not authorization.strip() or not manifest_sha256
    ):
        raise ValueError("final evaluation requires authorization and the frozen manifest SHA256")
    actual_hash = file_hash(manifest_path)
    if role == "final" and manifest_sha256 != actual_hash:
        raise ValueError("authorization manifest hash does not match frozen selection")
    destination = output / f"{role}-report.json"
    attempt_path = output / f"{role}-attempt.json"
    if destination.exists() or attempt_path.exists():
        raise ValueError(f"{role} evaluation already attempted; use a fresh preregistered run")
    manifest = validate_frozen_run(output)
    config = ExperimentConfig.from_dict(json.loads((output / "experiment.json").read_text()))
    protocol = evaluation_protocol(config)
    if protocol.manifest() != manifest["partitions"]:
        raise ValueError("reconstructed partitions differ from frozen manifest")
    # Write intent before evaluating protected data, including failed attempts.
    attempt = {
        "role": role,
        "manifest_sha256": actual_hash,
        "authorization": authorization,
        "status": "started",
        "schema_version": "agg.independent-attempt/1",
    }
    with attempt_path.open("x", encoding="utf-8") as stream:
        json.dump(attempt, stream, indent=2)

    def correct(model: nn.Module, split) -> list[int]:
        model.eval()
        with torch.no_grad():
            logits = model(split.x)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite independent predictions")
            return (logits.argmax(-1) == split.y).int().tolist()

    def measure(filename: str) -> dict[str, Any]:
        if filename not in manifest["artifacts"]:
            raise ValueError("checkpoint is not part of frozen selection")
        model = load_model(output / filename)
        measurements = {}
        for name in ("id", "ood"):
            split = getattr(data, name)
            observed: dict[str, Any] = evaluate(model, split)
            selection_score = evaluate(model, getattr(selection, name))[metric]
            missing_classes = metric == "balanced_accuracy" and (
                set(split.y.tolist()) != set(range(protocol.classes))
            )
            if missing_classes:
                observed[metric] = None
            measurements[name] = {
                "score": observed[metric],
                "metrics": observed,
                "metric": metric,
                "selection_score": selection_score,
                "selection_gap": (
                    selection_score - observed[metric] if observed[metric] is not None else None
                ),
                "missing_reason": "required class absent" if missing_classes else None,
                "paired_vs_baseline": paired_cluster_comparison(
                    baseline_correct[name],
                    correct(model, split),
                    getattr(data, name + "_clusters"),
                    seed=bootstrap_seed,
                    labels=split.y.tolist() if metric == "balanced_accuracy" else None,
                    expected_classes=(
                        list(range(protocol.classes)) if metric == "balanced_accuracy" else None
                    ),
                ),
            }
        return measurements

    try:
        data = protocol.evaluation(role)
        selection = protocol.evaluation("selection")
        metric = "balanced_accuracy" if config.task == "hierarchy" else "accuracy"
        baseline = load_model(output / "baseline-state.pt")
        baseline_correct = {name: correct(baseline, getattr(data, name)) for name in ("id", "ood")}
        candidates = []
        for record in Ledger(output / "candidate-attempts.jsonl").read():
            if record["event"] != "completed":
                continue
            measured = None
            error = None
            if record.get("checkpoint"):
                try:
                    measured = measure(record["checkpoint"])
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            candidates.append(
                {
                    "candidate_index": record["candidate_index"],
                    "family": record["family"],
                    "selection_status": record["status"],
                    "selection_reason": record["reason"],
                    "measurement": measured,
                    "missing_reason": error or (record["reason"] if measured is None else None),
                }
            )
        report = {
            "schema_version": "agg.independent-report/1",
            "role": role,
            "manifest_sha256": actual_hash,
            "authorization": authorization,
            "data_seed": config.data_seed,
            "model_seed": config.training.seed,
            "candidates_considered": len(candidates),
            "selection_reuse_cycles": len(candidates),
            "baseline": measure("baseline-state.pt"),
            "selected": measure("selected-state.pt"),
            "candidates": candidates,
            "final_test": "this authorized report" if role == "final" else None,
            "scientific_status": (
                "independent measurements; no generalization conclusion from fixture"
            ),
            "audit_reuse": (
                "report-only after frozen selection; later adaptation requires fresh data"
            ),
        }
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return report
    except Exception as exc:
        Ledger(output / "independent-failures.jsonl").append(
            {**attempt, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        )
        raise
