"""Nonmonotonic curriculum gates and separately scoped retention curves."""

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .events import EventLog, EventType


@dataclass(frozen=True)
class CurriculumDecision:
    action: str
    stage: int
    failing_stages: tuple[int, ...]
    missing_stages: tuple[int, ...]


class CurriculumGate:
    def __init__(self, thresholds: dict[int, float], events: EventLog | None = None):
        if not thresholds or any(
            type(k) is not int or k < 0 or not math.isfinite(v) or not 0 <= v <= 1
            for k, v in thresholds.items()
        ):
            raise ValueError("stage thresholds must be nonnegative indices and finite scores [0,1]")
        self.thresholds = dict(thresholds)
        self.events = events

    def evaluate(
        self, current: int, scores: dict[int, float], *, step: int = 0
    ) -> CurriculumDecision:
        if current not in self.thresholds:
            raise ValueError("current stage requires a configured threshold")
        selected = sorted(s for s in self.thresholds if s <= current)
        missing = tuple(s for s in selected if s not in scores or not math.isfinite(scores[s]))
        failed = tuple(s for s in selected if s not in missing and scores[s] < self.thresholds[s])
        if missing:
            decision = CurriculumDecision("hold", current, failed, missing)
        elif failed:
            decision = CurriculumDecision("backtrack", min(failed), failed, missing)
        else:
            future = sorted(s for s in self.thresholds if s > current)
            decision = CurriculumDecision(
                "advance" if future else "complete", future[0] if future else current, (), ()
            )
        if self.events:
            self.events.emit(EventType.CURRICULUM, step, asdict(decision))
        return decision


class RetentionTracker:
    """Curve fit is descriptive exponential decay, not a causal lifetime estimate."""

    def __init__(self, min_samples: int = 5):
        if min_samples < 3:
            raise ValueError("retention fit requires at least three observations")
        self.min_samples = min_samples
        self.curves: dict[str, list[tuple[int, float]]] = {}

    def record(self, scope: str, step: int, accuracy: float) -> None:
        if not scope or not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or step < 0:
            raise ValueError("retention needs named scope, nonnegative step and accuracy [0,1]")
        curve = self.curves.setdefault(scope, [])
        if curve and step <= curve[-1][0]:
            raise ValueError("retention steps must increase")
        curve.append((step, accuracy))

    def summary(self, scope: str) -> dict[str, float | int | None]:
        curve = self.curves.get(scope, [])
        if not curve:
            return {
                "samples": 0,
                "retained_accuracy": None,
                "forgetting": None,
                "slope": None,
                "half_life": None,
            }
        t, y = np.array(curve, dtype=float).T
        slope = half_life = None
        if len(y) >= self.min_samples:
            t = t - t[-1]
            slope = float(np.polyfit(t, y, 1)[0])
            if np.all(y > 0) and y[-1] < y[0]:
                decay = float(np.polyfit(t, np.log(y), 1)[0])
                half_life = math.log(2) / -decay if decay < 0 else None
        return {
            "samples": len(curve),
            "retained_accuracy": float(y[-1]),
            "forgetting": float(y.max() - y[-1]),
            "slope": slope,
            "half_life": half_life,
        }


class ContinualLearningRepairProvider(Protocol):
    """Future post-hoc repair must still pass the controller's protected gates."""

    def detect_regression(self, retained: dict[str, float]) -> list[str]: ...
    def propose_repair(self, regressions: list[str]) -> dict[str, object]: ...
    def evaluate_repair(self, proposal: dict[str, object]) -> dict[str, float]: ...
