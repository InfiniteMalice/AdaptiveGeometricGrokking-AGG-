"""Constrained acceptance: target improvement never buys protected regression."""

import math
from dataclasses import dataclass

from .config import MetricGuard


@dataclass(frozen=True)
class EvaluationWindow:
    start_step: int
    end_step: int
    metrics: dict[str, float]
    samples: int

    def __post_init__(self) -> None:
        if (
            any(type(n) is not int for n in (self.start_step, self.end_step, self.samples))
            or self.start_step < 0
            or self.end_step < self.start_step
            or self.samples < 1
        ):
            raise ValueError("evaluation requires ordered nonnegative steps and positive samples")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in self.metrics.values()):
            raise ValueError("evaluation metrics must be finite numbers")


def regression_reasons(
    baseline: dict[str, float],
    candidate: dict[str, float],
    anchor: dict[str, float],
    guards: tuple[MetricGuard, ...],
    target: str,
    higher_is_better: bool,
    minimum_gain: float,
    *,
    check_target: bool = True,
) -> list[str]:
    reasons = []

    def finite(metrics: dict[str, float], name: str) -> bool:
        value = metrics.get(name)
        return type(value) in (float, int) and math.isfinite(value)  # type: ignore[arg-type]

    if check_target:
        if not finite(baseline, target) or not finite(candidate, target):
            reasons.append(f"missing/nonfinite target: {target}")
        else:
            gain = (candidate[target] - baseline[target]) * (1 if higher_is_better else -1)
            if gain <= minimum_gain:
                reasons.append(f"target gain {gain:g} does not exceed {minimum_gain:g}: {target}")
    for guard in guards:
        if not finite(candidate, guard.name) or not finite(anchor, guard.name):
            reasons.append(f"missing/nonfinite protected metric: {guard.name}")
            continue
        references = [anchor[guard.name]]
        if finite(baseline, guard.name):
            references.append(baseline[guard.name])
        for reference in references:
            regression = (reference - candidate[guard.name]) * (1 if guard.higher_is_better else -1)
            if regression > guard.tolerance + 1e-12:
                reasons.append(f"protected regression exceeds {guard.tolerance:g}: {guard.name}")
                break
    return reasons
