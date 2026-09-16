"""Independent feasibility constraints; diagnostic metrics are not loss terms."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Evaluation:
    id_accuracy: float
    ood_accuracy: float
    mechanism_score: float
    cost: float = 0.0
    execution_score: float | None = None
    stable: bool = True
    reproducible: bool = True


@dataclass(frozen=True)
class Constraints:
    id_tolerance: float = 0.02
    ood_tolerance: float = 0.02
    mechanism_min: float = 0.9
    execution_tolerance: float = 0.0
    require_execution: bool = False
    require_cost_reduction: bool = False

    def __post_init__(self) -> None:
        values = [
            self.id_tolerance,
            self.ood_tolerance,
            self.mechanism_min,
            self.execution_tolerance,
        ]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("constraint thresholds must be finite and in [0,1]")

    def violations(self, baseline: Evaluation, candidate: Evaluation) -> list[str]:
        reasons = []
        for label, e in (("reference", baseline), ("candidate", candidate)):
            values = [e.id_accuracy, e.ood_accuracy, e.mechanism_score, e.cost]
            if e.execution_score is not None:
                values.append(e.execution_score)
            if any(not math.isfinite(v) for v in values):
                reasons.append(f"{label} contains nonfinite evidence")
            if not e.stable:
                reasons.append(f"{label} numerical stability failed")
            if not e.reproducible:
                reasons.append(f"{label} reproducibility failed")
        if baseline.id_accuracy - candidate.id_accuracy > self.id_tolerance + 1e-12:
            reasons.append("ID capability degradation exceeds tolerance")
        if baseline.ood_accuracy - candidate.ood_accuracy > self.ood_tolerance + 1e-12:
            reasons.append("OOD capability degradation exceeds tolerance")
        if candidate.mechanism_score < self.mechanism_min:
            reasons.append("mechanism preservation below threshold")
        if self.require_execution:
            if baseline.execution_score is None or candidate.execution_score is None:
                reasons.append("required execution evidence unavailable")
            elif baseline.execution_score - candidate.execution_score > self.execution_tolerance:
                reasons.append("execution behavior degradation exceeds tolerance")
        if self.require_cost_reduction and candidate.cost >= baseline.cost:
            reasons.append("candidate does not reduce configured cost")
        return reasons
