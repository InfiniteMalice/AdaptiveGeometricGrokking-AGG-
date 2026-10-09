"""Controlled geometry/parameter/volume grid with frozen extrapolation reporting."""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agg.controller.resources import measure_resources
from agg.evaluation.independent import paired_cluster_comparison
from agg.evaluation.scaling import fit_scaling
from agg.geometry import AdaptedModel, GeometryAdapter
from agg.horizon import effective_horizons
from agg.models import TinyTransformer
from agg.tasks import Split
from agg.tasks.causal import intervention_pairs, oracle
from agg.tasks.protocol import EvaluationProtocol, make_protocol
from agg.training import TrainConfig, evaluate, seed_all, train

from .blocks import _correct, _paired
from .candidates import ActivationPrecisionModel
from .causal import _measure, _predictions
from .checkpoints import load_model, save_model
from .independent import file_hash, freeze_run, validate_frozen_run


@dataclass(frozen=True)
class ScalingConfig:
    fit_widths: tuple[int, ...] = (8, 12, 16, 20)
    extrapolation_widths: tuple[int, ...] = (24,)
    geometries: tuple[str, ...] = ("euclidean", "hyperbolic", "product")
    seeds: tuple[int, ...] = (17, 19)
    volumes: tuple[int, ...] = (16, 64)
    steps: tuple[int, ...] = (4, 8)
    dimension: int = 4
    data_seed: int = 149
    samples: int = 32
    batch_size: int = 16
    bootstrap_draws: int = 200
    measurement_repeats: int = 9

    def __post_init__(self) -> None:
        groups = (
            self.fit_widths,
            self.extrapolation_widths,
            self.geometries,
            self.seeds,
            self.volumes,
            self.steps,
        )
        if any(not g or len(set(g)) != len(g) for g in groups):
            raise ValueError("nonempty unique grid axes required")
        integers = (
            *self.fit_widths,
            *self.extrapolation_widths,
            *self.volumes,
            *self.steps,
            self.dimension,
            self.samples,
            self.batch_size,
            self.bootstrap_draws,
            self.measurement_repeats,
        )
        if any(type(v) is not int or v < 1 for v in integers):
            raise ValueError("positive integer grid values required")
        widths = self.fit_widths + self.extrapolation_widths
        if (
            len(self.fit_widths) < 3
            or min(self.extrapolation_widths) <= max(self.fit_widths)
            or any(w % 2 or w < self.dimension for w in widths)
            or self.dimension < 2
        ):
            raise ValueError("three small even widths and strictly larger held-out widths required")
        if set(self.geometries) - {"euclidean", "hyperbolic", "product"}:
            raise ValueError("unsupported geometry")
        if any(type(s) is not int or not 0 <= s < 2**32 - 4 for s in (*self.seeds, self.data_seed)):
            raise ValueError("valid seeds required")
        if self.samples < 10 or self.bootstrap_draws < 20 or self.measurement_repeats < 3:
            raise ValueError("insufficient measurement counts")
        if (
            len(widths)
            * len(self.geometries)
            * len(self.seeds)
            * len(self.volumes)
            * len(self.steps)
            > 256
        ):
            raise ValueError("bounded CPU grid supports at most 256 cells")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScalingConfig":
        values = dict(raw)
        for name in (
            "fit_widths",
            "extrapolation_widths",
            "geometries",
            "seeds",
            "volumes",
            "steps",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


def _protocol(config: ScalingConfig) -> EvaluationProtocol:
    return make_protocol(
        "retrieval", seed=config.data_seed, samples=config.samples, length=8, distance=2
    )


def run_scaling(config: ScalingConfig, output: Path) -> dict[str, Any]:
    protocol = _protocol(config)
    development = protocol.development()
    if max(config.volumes) > len(development.train.y):
        raise ValueError("training volume exceeds available independent training rows")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "scaling-config.json").write_text(json.dumps(asdict(config), indent=2))
    order = torch.randperm(
        len(development.train.y), generator=torch.Generator().manual_seed(config.data_seed + 77)
    )
    cells = []
    for width, geometry, seed, volume, steps in product(
        config.fit_widths + config.extrapolation_widths,
        config.geometries,
        config.seeds,
        config.volumes,
        config.steps,
    ):
        identity = f"w{width}-{geometry}-s{seed}-v{volume}-u{steps}"
        indices = order[:volume]
        data = replace(
            development, train=Split(development.train.x[indices], development.train.y[indices])
        )
        completed = 0
        last_step = 0
        started = time.perf_counter()
        checkpoint = None
        row: dict[str, Any] = {
            "id": identity,
            "width": width,
            "geometry": geometry,
            "seed": seed,
            "volume": volume,
            "steps": steps,
            "fit_role": "fit" if width in config.fit_widths else "extrapolation",
            "parameters": None,
            "trainable_parameters": None,
            "training_rows_available": volume,
            "training_unique_sequences": len(torch.unique(data.train.x, dim=0)),
            "training_flops": None,
            "flops_missing_reason": "operator-complete training FLOPs not instrumented",
            "missing_reason": None,
        }
        # Reproduce the trainer's independent batch-index generator to count exposure.
        generator = torch.Generator().manual_seed(seed + 1)
        seen: set[int] = set()
        planned = []
        for _ in range(steps):
            planned.append(
                torch.randperm(volume, generator=generator)[: config.batch_size].tolist()
            )

        def observe(step: int, model: nn.Module, metrics: dict[str, Any]) -> None:
            nonlocal completed, last_step
            completed = step
            last_step = step

        training_started = False
        try:
            seed_all(seed)
            base = TinyTransformer(
                protocol.vocab_size, protocol.classes, width=width, heads=2, layers=1, max_length=8
            )
            model = AdaptedModel(
                base,
                GeometryAdapter(
                    width, config.dimension, geometry, gate=0.5, learned_curvature=True
                ),
            )
            row["parameters"] = sum(p.numel() for p in model.parameters())
            row["trainable_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            training_started = True
            trained = train(
                data,
                TrainConfig(
                    steps=steps,
                    eval_every=1,
                    width=width,
                    heads=2,
                    layers=1,
                    seed=seed,
                    batch_size=config.batch_size,
                ),
                output / identity,
                model=model,
                callback=observe,
            )
            checkpoint = f"{identity}/inference.pt"
            save_model(trained.model, output / checkpoint)
            row["selection"] = trained.history[-1]
            row["resources"] = _attempt(
                measure_resources,
                model,
                data.id.x[: config.batch_size],
                repeats=config.measurement_repeats,
            )
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["missing_reason"] = f"{type(exc).__name__}: {exc}"
        for batch in planned[:last_step]:
            seen.update(batch)
        row.update(
            {
                "checkpoint": checkpoint,
                "training_unique_rows_observed_lower_bound": len(seen),
                "budget": {
                    "updates": completed if completed == steps or not training_started else None,
                    "completed_updates_lower_bound": completed,
                    "tokens": completed * min(config.batch_size, volume) * 8
                    if completed == steps or not training_started
                    else None,
                    "tokens_lower_bound": completed * min(config.batch_size, volume) * 8,
                },
                "wall_seconds": time.perf_counter() - started,
            }
        )
        cells.append(row)
        (output / "scaling-progress.json").write_text(json.dumps(cells, indent=2, allow_nan=False))
    result = {
        "schema_version": "agg.scaling-grid/1",
        "cells": cells,
        "parameter_control": "same adapter dimension, prototypes, gate and learned scalar count",
        "resource_control": (
            "equal updates and actual tokens within each volume/budget; FLOPs not equated"
        ),
    }
    (output / "scaling-grid.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    freeze_run(output, protocol.manifest())
    return result


def _horizon(
    model: nn.Module, split: Split, clusters: tuple[str, ...], metadata: dict[str, Any], seed: int
) -> dict[str, Any]:
    x = split.x
    query = x[:, -1] - 65
    relevant = (x[:, :-1] > 0) & ((x[:, :-1] - 1) // 4 == query[:, None])
    positions = relevant.long().argmax(1)
    if not bool((relevant.sum(1) == 1).all()) or not torch.equal(oracle(x, metadata), split.y):
        raise ValueError("horizon source failed oracle validation")
    distances = [1, 2, 4, 7]
    baseline = [int(p == y) for p, y in zip(_predictions(model, x), split.y.tolist(), strict=True)]
    accuracy, curves = [], []
    for distance in distances:
        target = x.shape[1] - 1 - distance
        moved = x.clone()
        rows = torch.arange(len(x))
        moved[rows, positions] = x[:, target]
        moved[:, target] = x[rows, positions]
        if not torch.equal(oracle(moved, metadata), split.y):
            raise ValueError("horizon move changed task semantics")
        correct = [
            int(p == y) for p, y in zip(_predictions(model, moved), split.y.tolist(), strict=True)
        ]
        accuracy.append(sum(correct) / len(correct))
        curves.append(paired_cluster_comparison(baseline, correct, clusters, seed=seed))
    result = effective_horizons(distances, accuracy, 7, threshold=0.8)
    return {
        **result,
        "distances": distances,
        "accuracy": accuracy,
        "paired_vs_original": curves,
        "oracle_valid": True,
        "missing_reason": "no sampled distance reached threshold"
        if result["retrieval_horizon"] is None
        else None,
    }


def _attempt(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        return {"measurement": None, "missing_reason": f"{type(exc).__name__}: {exc}"}


def _compression(
    model: nn.Module,
    private: Any,
    protocol: EvaluationProtocol,
    seed: int,
    precision: str,
    scores: dict[str, Any],
) -> dict[str, Any]:
    altered = ActivationPrecisionModel(model, precision)
    comparisons = {
        name: _paired(
            model,
            altered,
            getattr(private, name),
            getattr(private, name + "_clusters"),
            protocol,
            seed,
        )
        for name in ("id", "ood")
    }
    return {
        "paired_accuracy": comparisons,
        "within_accuracy_tolerance": all(c["effect"] >= -0.02 for c in comparisons.values()),
        "baseline_capability_established": all(
            s.get("accuracy", -1) >= 0.8 for s in scores.values()
        ),
        "physical_savings": None,
        "physical_savings_reason": "simulated activation precision, unchanged tensor storage",
    }


def audit_scaling(output: Path, *, seed: int = 0) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**32 - 4:
        raise ValueError("valid audit seed required")
    output = Path(output)
    if (output / "scaling-audit-attempt.json").exists():
        raise ValueError("scaling audit already attempted")
    manifest = validate_frozen_run(output)
    config = ScalingConfig.from_dict(json.loads((output / "scaling-config.json").read_text()))
    protocol = _protocol(config)
    if manifest["partitions"] != protocol.manifest():
        raise ValueError("scaling partitions differ from frozen selection")
    with (output / "scaling-audit-attempt.json").open("x") as stream:
        json.dump(
            {
                "role": "audit",
                "seed": seed,
                "manifest_sha256": file_hash(output / "selection-frozen.json"),
            },
            stream,
        )
    try:
        private = protocol.evaluation("audit")
        grid = json.loads((output / "scaling-grid.json").read_text())
        pairs = {
            kind: _attempt(
                intervention_pairs,
                private.ood,
                private.ood_clusters,
                protocol.metadata,
                kind=kind,
                seed=seed,
            )
            for kind in ("invariant", "decisive")
        }
        rows = []
        for cell in grid["cells"]:
            row = {**cell, "measurement": None, "error": None}
            counter = {"forward_calls": 0, "token_inputs": 0}
            audit_started = time.perf_counter()
            hook = None
            try:
                if cell["checkpoint"] is None or cell["status"] != "completed":
                    raise ValueError(cell["missing_reason"] or "training checkpoint unavailable")
                if cell["checkpoint"] not in manifest["artifacts"]:
                    raise ValueError("checkpoint not frozen")
                model = load_model(output / cell["checkpoint"])

                def count_forward(module, inputs, counter=counter):
                    counter["forward_calls"] += 1
                    counter["token_inputs"] += inputs[0].numel()

                hook = model.register_forward_pre_hook(count_forward)
                scores = {
                    name: _attempt(evaluate, model, getattr(private, name))
                    for name in ("id", "ood")
                }
                row["measurement"] = {"scores": scores}
                if "accuracy" in scores["ood"]:
                    row["error"] = 1 - scores["ood"]["accuracy"]
                else:
                    row["missing_reason"] = scores["ood"]["missing_reason"]
                # Primary response is retained before any optional diagnostic runs.
                row["measurement"].update(
                    {
                        "correct_observations": {
                            name: _attempt(_correct, model, getattr(private, name))
                            for name in ("id", "ood")
                        },
                        "compression": {
                            precision: _attempt(
                                _compression, model, private, protocol, seed, precision, scores
                            )
                            for precision in ("FP16", "INT8")
                        },
                        "horizon": _attempt(
                            _horizon,
                            model,
                            private.id,
                            private.id_clusters,
                            protocol.metadata,
                            seed,
                        ),
                        "causal": {
                            kind: _attempt(_measure, model, pair, seed)
                            if not isinstance(pair, dict)
                            else pair
                            for kind, pair in pairs.items()
                        },
                    }
                )
            except Exception as exc:
                row["missing_reason"] = f"{type(exc).__name__}: {exc}"
            finally:
                if hook is not None:
                    hook.remove()
                row["audit_budget"] = {
                    **counter,
                    "wall_seconds": time.perf_counter() - audit_started,
                    "scope": "all scoring, compression, horizon and causal forwards for this cell",
                }
            rows.append(row)
        geometry_contrasts = []
        for row in rows:
            if row["geometry"] == "euclidean":
                continue
            baseline = next(
                (
                    r
                    for r in rows
                    if r["geometry"] == "euclidean"
                    and all(r[k] == row[k] for k in ("width", "seed", "volume", "steps"))
                ),
                None,
            )
            available = (
                baseline is not None
                and baseline["measurement"] is not None
                and row["measurement"] is not None
            )
            geometry_contrasts.append(
                {
                    "id": row["id"],
                    "reference": baseline["id"] if baseline else None,
                    "parameter_matched": baseline is not None
                    and baseline["trainable_parameters"] == row["trainable_parameters"],
                    "budget_matched": baseline is not None and baseline["budget"] == row["budget"],
                    "paired_accuracy": {
                        name: _attempt(
                            paired_cluster_comparison,
                            baseline["measurement"]["correct_observations"][name],
                            row["measurement"]["correct_observations"][name],
                            getattr(private, name + "_clusters"),
                            seed=seed,
                        )
                        for name in ("id", "ood")
                    }
                    if available and baseline is not None
                    else None,
                    "missing_reason": None
                    if available
                    else "matching completed Euclidean/reference measurements unavailable",
                }
            )
        groups = []
        for geometry, volume, steps in product(config.geometries, config.volumes, config.steps):
            selected = [
                r
                for r in rows
                if r["geometry"] == geometry and r["volume"] == volume and r["steps"] == steps
            ]
            groups.append(
                {
                    "geometry": geometry,
                    "volume": volume,
                    "steps": steps,
                    "response": "independent audit OOD error",
                    "fit": fit_scaling(selected, seed=seed, draws=config.bootstrap_draws),
                }
            )
        volume_effects = []
        for geometry, width, steps in product(
            config.geometries, config.fit_widths + config.extrapolation_widths, config.steps
        ):
            by_volume = {
                v: {
                    r["seed"]: r
                    for r in rows
                    if r["geometry"] == geometry
                    and r["width"] == width
                    and r["steps"] == steps
                    and r["volume"] == v
                    and r["error"] is not None
                }
                for v in config.volumes
            }
            volumes = sorted(config.volumes)
            for left, right in zip(volumes, volumes[1:], strict=False):
                common = sorted(by_volume[left].keys() & by_volume[right].keys())
                differences = [
                    by_volume[left][s]["error"] - by_volume[right][s]["error"] for s in common
                ]
                interval = None
                if len(differences) >= 2:
                    rng = np.random.default_rng(seed)
                    interval = np.quantile(
                        [
                            np.mean(rng.choice(differences, len(differences), replace=True))
                            for _ in range(config.bootstrap_draws)
                        ],
                        [0.025, 0.975],
                    ).tolist()
                volume_effects.append(
                    {
                        "geometry": geometry,
                        "width": width,
                        "steps": steps,
                        "smaller_volume": left,
                        "larger_volume": right,
                        "paired_seeds": common,
                        "accuracy_gain": float(np.mean(differences)) if differences else None,
                        "interval": interval,
                        "interpretation": (
                            "descriptive seed-paired volume effect; "
                            "near-zero does not establish saturation"
                        ),
                    }
                )
        result = {
            "schema_version": "agg.scaling-audit/1",
            "cells": rows,
            "scaling_groups": groups,
            "volume_effects": volume_effects,
            "geometry_contrasts": geometry_contrasts,
            "final_test": None,
            "scientific_status": (
                "controlled CPU measurements; no established scaling law or robotics claim"
            ),
        }
        (output / "scaling-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    except Exception as exc:
        (output / "scaling-audit-failure.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        )
        raise
