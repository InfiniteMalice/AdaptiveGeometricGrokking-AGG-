"""Paired structural-cluster uncertainty, separate from adaptive acceptance."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def paired_cluster_comparison(
    before: Sequence[int],
    after: Sequence[int],
    clusters: Sequence[str],
    *,
    seed: int,
    draws: int = 1000,
    labels: Sequence[int] | None = None,
    expected_classes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Bootstrap whole paired clusters; labels requests macro class recall.

    These descriptive intervals do not correct adaptive selection or repeated
    audit reuse. ID and OOD strata must be passed separately by the caller.
    """
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if (
        a.ndim != 1
        or b.shape != a.shape
        or len(a) == 0
        or len(clusters) != len(a)
        or not np.isin(a, [0, 1]).all()
        or not np.isin(b, [0, 1]).all()
        or any(not isinstance(c, str) or not c for c in clusters)
        or type(draws) is not int
        or not 100 <= draws <= 10000
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("need aligned nonempty binary pairs, named clusters and valid seed/draws")
    y = None if labels is None else np.asarray(labels)
    if y is not None and (y.shape != a.shape or not np.isfinite(y).all()):
        raise ValueError("balanced comparisons require aligned finite labels")
    if y is not None and (
        expected_classes is None
        or len(set(expected_classes)) < 2
        or any(type(c) is not int for c in expected_classes)
    ):
        raise ValueError("balanced comparisons require the complete expected class universe")
    classes = np.asarray(sorted(set(expected_classes or ()))) if y is not None else None
    if y is not None:
        assert classes is not None
        if not np.isin(y, classes).all():
            raise ValueError("labels lie outside the expected class universe")
    names = sorted(set(clusters))
    rows = [np.asarray([i for i, c in enumerate(clusters) if c == name]) for name in names]

    def effect(indices: np.ndarray) -> float | None:
        delta = b[indices] - a[indices]
        if y is None:
            return float(delta.mean())
        assert classes is not None
        if any(not np.any(y[indices] == c) for c in classes):
            return None
        return float(np.mean([delta[y[indices] == c].mean() for c in classes]))

    samples = []
    if len(rows) >= 2:
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            selected = rng.integers(len(rows), size=len(rows))
            value = effect(np.concatenate([rows[i] for i in selected]))
            if value is not None:
                samples.append(value)
    interval = np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) == draws else None
    return {
        "effect": effect(np.arange(len(a))),
        "interval": interval,
        "confidence": 0.95,
        "clusters": len(rows),
        "samples": len(a),
        "seed": seed,
        "bootstrap_draws": draws,
        "valid_draws": len(samples),
        "metric": "accuracy" if y is None else "balanced_accuracy",
        "method": "paired whole-cluster percentile bootstrap within one stratum",
        "unavailable_reason": (
            "fewer than two clusters or resampled classes missing" if interval is None else None
        ),
        "interpretation": "descriptive; not adjusted for selection or repeated audit reuse",
    }
