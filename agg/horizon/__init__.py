"""Empirical horizons from independently measured distance-response sweeps."""

import math
from collections.abc import Sequence


def effective_horizons(
    distances: Sequence[float],
    retrieval: Sequence[float],
    nominal: float,
    threshold: float = 0.8,
    attribution: Sequence[float] | None = None,
    causal: Sequence[float] | None = None,
) -> dict[str, float | str | None]:
    if (
        not distances
        or not math.isfinite(nominal)
        or nominal < 0
        or not math.isfinite(threshold)
        or any(not math.isfinite(d) or d < 0 or d > nominal for d in distances)
    ):
        raise ValueError("Distances must be finite, nonnegative, and within nominal horizon")

    def estimate(values: Sequence[float] | None) -> float | None:
        if values is None:
            return None
        if len(values) != len(distances) or any(not math.isfinite(v) for v in values):
            raise ValueError("Each response must be finite and aligned with distances")
        accepted = [d for d, value in zip(distances, values, strict=True) if value >= threshold]
        return max(accepted) if accepted else None

    return {
        "nominal_horizon": nominal,
        "retrieval_horizon": estimate(retrieval),
        "attribution_horizon": estimate(attribution),
        "causal_horizon": estimate(causal),
        "threshold": threshold,
        "method": "maximum sampled qualifying distance; no interpolation",
    }
