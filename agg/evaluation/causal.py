"""Correctness-aware causal-pair measurements with explicit denominators."""

from typing import Any

import numpy as np

from .independent import paired_cluster_comparison


def intervention_metrics(
    before: list[int],
    after: list[int],
    original: list[int],
    transformed: list[int],
    clusters: list[str],
    *,
    kind: str,
    seed: int,
) -> dict[str, Any]:
    if (
        kind not in {"invariant", "decisive"}
        or not original
        or len({len(v) for v in (before, after, original, transformed, clusters)}) != 1
    ):
        raise ValueError("nonempty aligned intervention observations required")
    arrays = [np.asarray(v) for v in (before, after, original, transformed)]
    if any(a.dtype.kind not in "iu" or np.any(a < 0) for a in arrays):
        raise ValueError("predictions and labels must be nonnegative integer classes")
    p, q, y, z = arrays
    if not bool(np.all(y == z) if kind == "invariant" else np.all(y != z)):
        raise ValueError("labels do not satisfy the intervention class")
    correct_before, correct_after = p == y, q == z
    flip = p != q

    def rate(values, mask=None):
        mask = np.ones(len(y), dtype=bool) if mask is None else mask
        selected = values[mask].astype(int).tolist()
        selected_clusters = np.asarray(clusters)[mask].tolist()
        stats = (
            paired_cluster_comparison([0] * len(selected), selected, selected_clusters, seed=seed)
            if selected
            else None
        )
        return {
            "value": float(np.mean(selected)) if selected else None,
            "numerator": sum(selected),
            "denominator": len(selected),
            "uncertainty": stats,
            "missing_reason": None if selected else "no eligible observations",
        }

    return {
        "samples": len(y),
        "original_accuracy": rate(correct_before),
        "transformed_accuracy": rate(correct_after),
        "joint_correctness": rate(correct_before & correct_after),
        "prediction_flip": rate(flip),
        "correct_invariance": rate(correct_before & correct_after) if kind == "invariant" else None,
        "spurious_flip": rate(flip) if kind == "invariant" else None,
        "required_update": rate(correct_after, correct_before) if kind == "decisive" else None,
        "failure_recovery": rate(correct_after, ~correct_before),
        "paired_accuracy_change": paired_cluster_comparison(
            correct_before.astype(int).tolist(),
            correct_after.astype(int).tolist(),
            clusters,
            seed=seed,
        ),
    }
