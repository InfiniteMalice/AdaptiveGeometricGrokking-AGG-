"""Independent operational evidence scores and sustained-crossing summaries."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProbeEvidence:
    memorization: float
    retrieval: float
    generalization: float
    proxy: bool = True

    @classmethod
    def from_predictions(
        cls,
        predictions: Tensor,
        nearest_neighbor_predictions: Tensor,
        ablated_predictions: Tensor,
        labels: Tensor,
        ood_correct: Tensor,
    ) -> "ProbeEvidence":
        """kNN agreement, relevant-ablation accuracy loss, and independent OOD accuracy.

        kNN agreement is associative evidence, not proof of memorization. Relevant
        ablations must be executed by the caller; OOD correctness is a separate set.
        """
        values = (
            predictions,
            nearest_neighbor_predictions,
            ablated_predictions,
            labels,
            ood_correct,
        )
        if not all(bool(torch.isfinite(v).all()) for v in values):
            raise ValueError("Probe predictions and labels must be finite")
        if (
            predictions.numel() == 0
            or ood_correct.numel() == 0
            or any(
                v.shape != predictions.shape
                for v in (nearest_neighbor_predictions, ablated_predictions, labels)
            )
        ):
            raise ValueError("Require aligned nonempty predictions and independent OOD correctness")
        if not all(bool(((v == 0) | (v == 1)).all()) for v in (ood_correct,)):
            raise ValueError("ood_correct must be binary")
        memorization = float((predictions == nearest_neighbor_predictions).float().mean())
        retrieval = float(
            (predictions == labels).float().mean() - (ablated_predictions == labels).float().mean()
        )
        return cls(memorization, retrieval, float(ood_correct.float().mean()))


@dataclass(frozen=True)
class Crossing:
    first_crossing: int | None
    stable_crossing: int | None
    confirmation_step: int | None
    interval: tuple[int | None, int] | None
    last_step: int
    censored: bool


def retrieval_crossing(
    steps: Sequence[int],
    memorization: Sequence[float],
    retrieval: Sequence[float],
    margin: float = 0.0,
    sustain: int = 3,
) -> Crossing:
    """Interval brackets sampled onset, not a statistical confidence interval."""
    if (
        not steps
        or len(steps) != len(memorization)
        or len(steps) != len(retrieval)
        or sustain < 1
        or not math.isfinite(margin)
        or margin < 0
        or any(b <= a for a, b in zip(steps, steps[1:], strict=False))
    ):
        raise ValueError("Need increasing aligned steps, finite margin, and positive sustain")
    first = stable = confirmed = None
    interval = None
    run = 0
    for i, (step, m, r) in enumerate(zip(steps, memorization, retrieval, strict=True)):
        crossed = math.isfinite(m) and math.isfinite(r) and r > m + margin
        if crossed and first is None:
            first = step
        run = run + 1 if crossed else 0
        if run >= sustain and stable is None:
            start = i - sustain + 1
            stable, confirmed = steps[start], step
            interval = (steps[start - 1] if start else None, stable)
    return Crossing(first, stable, confirmed, interval, steps[-1], stable is None)


def summarize_crossings(crossings: Sequence[Crossing]) -> dict[str, Any]:
    observed = [c.stable_crossing for c in crossings if c.stable_crossing is not None]
    return {
        "seeds": len(crossings),
        "censored_seeds": sum(c.censored for c in crossings),
        "observed_median_step": statistics.median(observed) if observed else None,
        "observed_min_step": min(observed) if observed else None,
        "observed_max_step": max(observed) if observed else None,
        "censor_steps": [c.last_step for c in crossings if c.censored],
        "note": "Observed-only summary; censored seeds excluded, no survival inference",
    }
