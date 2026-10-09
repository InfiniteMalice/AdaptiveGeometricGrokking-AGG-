"""Reproducible small integration study, portable evidence, and explicit final release."""

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from agg.models import TinyTransformer
from agg.training import TrainConfig

from .causal import causal_report
from .config import ExperimentConfig, Features
from .independent import file_hash, report_run, validate_frozen_run
from .milestones import milestone_report
from .runner import evaluation_protocol, run_experiment, write_json


@dataclass(frozen=True)
class IntegrationConfig:
    tasks: tuple[str, ...] = ("retrieval", "modular")
    seeds: tuple[int, ...] = (23, 29)
    steps: int = 32
    candidate_steps: int = 8
    samples: int = 64
    width: int = 16
    eval_every: int = 4

    def __post_init__(self) -> None:
        if (
            not self.tasks
            or set(self.tasks) - {"retrieval", "modular"}
            or len(set(self.tasks)) != len(self.tasks)
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(v) is not int or v < 0 for v in self.seeds)
            or any(
                type(v) is not int or v < 1
                for v in (
                    self.steps,
                    self.candidate_steps,
                    self.samples,
                    self.width,
                    self.eval_every,
                )
            )
            or self.samples < 10
            or self.width < 4
            or self.width % 2
        ):
            raise ValueError("Invalid integration study axes or budget")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IntegrationConfig":
        values: dict[str, Any] = dict(raw)
        for key in ("tasks", "seeds"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def run_integration(config: IntegrationConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "integration-config.json", asdict(config))
    rows: list[dict[str, Any]] = []
    for task in config.tasks:
        for seed in config.seeds:
            cfg = ExperimentConfig(
                task=task,
                independent_evaluation=True,
                data_seed=211 if task == "retrieval" else 223,
                samples=config.samples,
                context_length=8,
                distance=2,
                dimensions=(4,),
                geometries=("euclidean",),
                candidate_steps=config.candidate_steps,
                training=TrainConfig(
                    steps=config.steps,
                    seed=seed,
                    width=config.width,
                    layers=1,
                    heads=2,
                    batch_size=16,
                    eval_every=config.eval_every,
                ),
                features=replace(
                    Features.baseline(), consolidation=True, distillation=True, telemetry=True
                ),
                telemetry_mode="task",
            )
            name = f"{task}-{seed}"
            root = output / name
            phase = {"name": "development"}
            calls: dict[str, dict[str, int]] = {}

            def count_forward(module, inputs, result, calls=calls, phase=phase):
                if isinstance(module, TinyTransformer):
                    entry = calls.setdefault(phase["name"], {"calls": 0, "token_inputs": 0})
                    entry["calls"] += 1
                    entry["token_inputs"] += inputs[0].numel()

            hook = torch.nn.modules.module.register_module_forward_hook(count_forward)
            row: dict[str, Any] = {
                "id": name,
                "task": task,
                "seed": seed,
                "data_seed": cfg.data_seed,
                "status": "started",
                "forward_calls": calls,
            }
            rows.append(row)
            start = time.perf_counter()
            try:
                summary = run_experiment(cfg, root)
                phase["name"] = "audit"
                audit = report_run(root, bootstrap_seed=41)
                phase["name"] = "causal"
                causal = causal_report(root, seed=41)
                phase["name"] = "milestones"
                milestones = milestone_report(root)
                data = evaluation_protocol(cfg).development()
                attempts = [
                    json.loads(s)
                    for s in (root / "candidate-attempts.jsonl").read_text().splitlines()
                ]
                completed = [a for a in attempts if a["event"] == "completed"]
                all_executed = all(a["status"] in ("accepted", "rejected") for a in completed)
                updates = cfg.training.steps + cfg.candidate_steps * (1 + len(completed))
                batch = min(cfg.training.batch_size, len(data.train.y))
                row.update(
                    {
                        "status": "complete",
                        "summary": summary,
                        "selected_vs_continued": audit["selected_vs_continued"],
                        "independent": {
                            key: audit[key] for key in ("baseline", "selected", "continued")
                        },
                        "crossing": causal["phase"]["crossing"],
                        "milestones": milestones,
                        "candidate_statuses": [a["status"] for a in completed],
                        "budget": {
                            "training_updates": updates if all_executed else None,
                            "training_tokens": updates * batch * data.train.x.shape[1]
                            if all_executed
                            else None,
                            "training_update_upper_budget": updates,
                            "candidate_count": len(completed),
                            "control_updates": cfg.candidate_steps,
                            "candidate_updates_each": cfg.candidate_steps,
                            "batch_examples_actual": batch,
                            "unknown_reason": None if all_executed else "partial candidate failure",
                            "training_flops": None,
                        },
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "failed_phase": phase["name"],
                        "reason": f"{type(exc).__name__}: {exc}",
                        "budget": None,
                        "budget_missing_reason": "partial run; inspect logs",
                    }
                )
            finally:
                hook.remove()
                row["wall_seconds"] = time.perf_counter() - start
                write_json(output / "integration-progress.json", rows)
    report = {
        "schema_version": "agg.integration-study/1",
        "config": asdict(config),
        "cells": rows,
        "final_test": None,
        "scope": "Small fixed smoke matrix, audit-only; no grokking/acceleration claim",
        "forward_cost_scope": "Actual TinyTransformer forward calls incl teacher/diagnostics; "
        "backward, Python overhead and FLOPs not counted",
    }
    write_json(output / "integration-report.json", report)
    return report


def build_bundle(sources: dict[str, Path], output: Path) -> dict[str, Any]:
    """Copy complete explicitly supplied evidence trees with content hashes.

    Sources may include multiple nested frozen runs. Their immutable selection
    artifacts are checked before copying; post-freeze reports receive bundle hashes.
    """
    if not sources or any(not re.fullmatch(r"[a-z][a-z0-9_-]*", k) for k in sources):
        raise ValueError("Nonempty uniquely named evidence sources required")
    for root in sources.values():
        root = root.resolve()
        if not root.is_dir() or output.resolve().is_relative_to(root):
            raise ValueError("Bundle must be outside each existing source directory")
        for p in root.rglob("*"):
            if p.is_symlink() or not p.resolve().is_relative_to(root):
                raise ValueError("Evidence source contains an external or symbolic path")
        for manifest in root.rglob("selection-frozen.json"):
            validate_frozen_run(manifest.parent)
    output.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, root in sources.items():
        destination = output / "sources" / name
        shutil.copytree(root, destination)
        for manifest in destination.rglob("selection-frozen.json"):
            validate_frozen_run(manifest.parent)
        files = {
            p.relative_to(destination).as_posix(): file_hash(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        }
        entries[name] = {
            "files": files,
            "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
        }
    bundle_manifest = {
        "schema_version": "agg.evidence-bundle/1",
        "sources": entries,
        "integrity": "Trusted-host hashes, not signatures or independent attestation",
        "final_test": "Only explicitly authorized source reports, if any",
    }
    write_json(output / "evidence-manifest.json", bundle_manifest)
    return bundle_manifest


def prepare_release(roots: list[Path], path: Path) -> dict[str, Any]:
    if not roots or len({p.resolve() for p in roots}) != len(roots):
        raise ValueError("Unique nonempty final-release runs required")
    rows = []
    for root in roots:
        validate_frozen_run(root)
        if (root / "final-attempt.json").exists():
            raise ValueError("final evaluation already attempted")
        rows.append(
            {
                "path": os.path.relpath(root.resolve(), path.parent.resolve()),
                "manifest_sha256": file_hash(root / "selection-frozen.json"),
            }
        )
    report = {
        "schema_version": "agg.final-release/1",
        "runs": rows,
        "hypothesis": "Selected self-distillation policy improves independent ID/OOD accuracy "
        "over equal-update continuation from the same initial checkpoint",
        "primary_endpoint": "paired selected-minus-continued OOD accuracy, all listed seeds",
        "secondary_endpoints": "ID accuracy, selection-final gaps, all candidate outcomes",
        "analysis": "Within-stratum whole-structural-cluster paired percentile bootstrap, "
        "seed41; descriptive, no pooled confirmatory p-value",
        "no_adaptation": True,
        "authorization": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


def release_final(path: Path, *, authorization: str, digest: str) -> dict[str, Any]:
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError("Named explicit researcher authorization required")
    if file_hash(path) != digest:
        raise ValueError("Final release digest mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "agg.final-release/1" or not manifest.get("runs"):
        raise ValueError("Unsupported or empty release manifest")
    attempt = path.with_name(path.stem + "-attempt.json")
    if attempt.exists():
        raise ValueError("final release already attempted")
    roots = [(path.parent / row["path"]).resolve() for row in manifest["runs"]]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate release runs")
    for root, row in zip(roots, manifest["runs"], strict=True):
        if file_hash(root / "selection-frozen.json") != row["manifest_sha256"]:
            raise ValueError("Frozen source manifest changed")
        validate_frozen_run(root)
        if (root / "final-attempt.json").exists() or (root / "final-report.json").exists():
            raise ValueError("source final evaluation already attempted")
    result: dict[str, Any] = {
        "schema_version": "agg.final-release-result/1",
        "authorization": authorization,
        "manifest_sha256": digest,
        "runs": [],
    }
    with attempt.open("x") as stream:
        json.dump({**result, "status": "started"}, stream, indent=2)
    for root, row in zip(roots, manifest["runs"], strict=True):
        try:
            report = report_run(
                root,
                role="final",
                authorization=authorization,
                manifest_sha256=row["manifest_sha256"],
                bootstrap_seed=41,
            )
            result["runs"].append({"path": row["path"], "status": "complete", "report": report})
        except Exception as exc:
            result["runs"].append(
                {"path": row["path"], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            )
        write_json(path.with_name(path.stem + "-result.json"), result)
    return result
