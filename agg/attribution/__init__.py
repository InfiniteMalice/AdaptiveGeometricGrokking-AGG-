"""Interventional score changes and centered linear representation similarity."""

import math

from torch import Tensor

from agg.dimension import observation_matrix


def intervention_attribution(
    baseline: float, intervened: float, higher_is_better: bool = True
) -> float:
    """Signed performance loss due to an explicitly executed intervention."""
    if not all(math.isfinite(v) for v in (baseline, intervened)):
        raise ValueError("Intervention scores must be finite")
    return (baseline - intervened) * (1 if higher_is_better else -1)


def linear_cka(x: Tensor, y: Tensor) -> float:
    x, y = observation_matrix(x), observation_matrix(y)
    if len(x) != len(y):
        raise ValueError("CKA needs matching aligned observation counts")
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = (x.T @ x).norm() * (y.T @ y).norm()
    if denominator == 0:
        return 0.0  # Constant representations provide no preservation evidence.
    return float(((x.T @ y).square().sum() / denominator).clamp(0, 1))
