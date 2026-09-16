"""Exploratory analyses; candidate changes are not established phase transitions."""

import math
import statistics
from typing import Any

import numpy as np


def candidate_changes(
    steps: list[int], values: list[float], *, window: int = 3, threshold: float = 0.2
) -> list[dict[str, Any]]:
    if (
        len(steps) != len(values)
        or window < 1
        or not math.isfinite(threshold)
        or threshold < 0
        or any(not math.isfinite(v) for v in values)
        or any(b <= a for a, b in zip(steps, steps[1:], strict=False))
    ):
        raise ValueError("need aligned increasing steps, finite values and valid window/threshold")
    changes = []
    for index in range(window, len(values) - window + 1):
        before = statistics.mean(values[index - window : index])
        after = statistics.mean(values[index : index + window])
        if abs(after - before) >= threshold:
            changes.append(
                {
                    "step": steps[index],
                    "before_mean": before,
                    "after_mean": after,
                    "delta": after - before,
                    "label": "candidate_change_not_confirmed_transition",
                }
            )
    return changes


def telemetry_redundancy(columns: dict[str, list[float]]) -> dict[str, Any]:
    if not columns or len({len(v) for v in columns.values()}) != 1:
        raise ValueError("need aligned named telemetry columns")
    matrix = np.asarray(list(columns.values()), dtype=float)
    if matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("need at least two finite observations per column")
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    correlation: list[list[float | None]] = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            row.append(
                float(np.clip(centered[i] @ centered[j] / (norms[i] * norms[j]), -1, 1))
                if norms[i] > 0 and norms[j] > 0
                else None
            )
        correlation.append(row)
    return {
        "components": list(columns),
        "correlation": correlation,
        "observations": matrix.shape[1],
        "method": "Pearson; constant columns unavailable",
    }


def phase_surface(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    factors = ("gate", "distance", "density", "context_length")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[f] for f in factors), []).append(row)
    surface = []
    for key, group in grouped.items():
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("duplicate seed in the same phase cell")
        onset = {str(row["seed"]): row["crossing"]["stable_crossing"] for row in group}
        observed = [value for value in onset.values() if value is not None]
        surface.append(
            {
                **dict(zip(factors, key, strict=True)),
                "seed_steps": onset,
                "censored_seeds": len(group) - len(observed),
                "observed_median_step": statistics.median(observed) if observed else None,
                "observed_range": [min(observed), max(observed)] if observed else None,
                "censor_steps": {
                    str(r["seed"]): r["crossing"]["last_step"]
                    for r in group
                    if r["crossing"]["stable_crossing"] is None
                },
                "inference": "observed-only summary; no confidence interval or survival estimate",
            }
        )
    return surface
