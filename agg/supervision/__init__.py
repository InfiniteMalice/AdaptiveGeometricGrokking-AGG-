"""Immutable observations from an actually executed deterministic integer environment."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

SOURCES = (
    "external_outcome",
    "environment_state",
    "execution_log",
    "task_verifier",
    "ground_truth",
    "teacher",
    "process",
    "self_report",
)


@dataclass(frozen=True)
class Signal:
    source: str
    value: float
    confidence: float
    position: int
    provenance: str

    def __post_init__(self) -> None:
        if (
            self.source not in SOURCES
            or not math.isfinite(self.value)
            or not -1 <= self.value <= 1
            or not 0 <= self.confidence <= 1
            or self.position < 0
            or not self.provenance
        ):
            raise ValueError("Invalid evidence source, value, confidence, position or provenance")


@dataclass(frozen=True)
class EvidencePolicy:
    hierarchy: tuple[str, ...] = SOURCES
    high_confidence: float = 0.9
    auxiliary_bound: float = 0.2

    def __post_init__(self) -> None:
        if set(self.hierarchy) != set(SOURCES) or len(self.hierarchy) != len(SOURCES):
            raise ValueError("Hierarchy must contain every source exactly once")
        if not 0 <= self.high_confidence <= 1 or not 0 <= self.auxiliary_bound < 0.5:
            raise ValueError("Invalid confidence threshold or auxiliary mixing bound")


@dataclass(frozen=True)
class CombinedEvidence:
    value: float
    primary_source: str | None
    conflicts: tuple[str, ...]
    signals: tuple[Signal, ...]


def combine_evidence(
    signals: list[Signal], policy: EvidencePolicy | None = None
) -> CombinedEvidence:
    """Select strongest source, bounding *total* auxiliary influence, not each signal.

    High-confidence independently observed outcomes have precedence even when the
    configurable ordinary hierarchy puts teacher signals first.
    """
    policy = policy or EvidencePolicy()
    usable = [s for s in signals if s.confidence > 0]
    if not usable:
        return CombinedEvidence(0.0, None, (), tuple(signals))
    strong = [
        s
        for s in usable
        if s.source in ("external_outcome", "environment_state")
        and s.confidence >= policy.high_confidence
    ]
    candidates = strong or usable
    primary = min(candidates, key=lambda s: (policy.hierarchy.index(s.source), -s.confidence))
    base = primary.value * primary.confidence
    auxiliary = [s for s in usable if s is not primary]
    conflicts = tuple(s.source for s in auxiliary if s.value * primary.value < 0)
    if auxiliary:
        average = sum(s.value * s.confidence for s in auxiliary) / sum(
            s.confidence for s in auxiliary
        )
        # Relative cap prevents arbitrarily weak verified values being sign-flipped.
        bound = policy.auxiliary_bound * abs(base)
        base += max(-bound, min(bound, average))
    return CombinedEvidence(max(-1.0, min(1.0, base)), primary.source, conflicts, tuple(signals))


@dataclass(frozen=True)
class ExecutionEvidence:
    order: int
    intent: str
    action: str
    argument: int
    pre_state: int
    post_state: int
    result: int | None
    error: str | None
    expected_state: int
    verified: bool
    verifier_confidence: float
    verifier_provenance: str
    pre_hash: str
    post_hash: str

    def signal(self) -> Signal:
        return Signal(
            "environment_state",
            1.0 if self.verified else -1.0,
            self.verifier_confidence,
            self.order,
            self.verifier_provenance,
        )


class ToyEnvironment:
    """Integer register: add, multiply, set, and exact integer division.

    Failed actions retain the prior state. Intent is diagnostic text only. Order
    is the deterministic logical timestamp; expected state is an explicit verifier target.
    """

    def __init__(self, initial_state: int = 0) -> None:
        if type(initial_state) is not int:
            raise ValueError("Initial state must be an integer")
        self._state = initial_state
        self._trace: list[ExecutionEvidence] = []

    @property
    def state(self) -> int:
        return self._state

    @property
    def trace(self) -> tuple[ExecutionEvidence, ...]:
        return tuple(self._trace)

    def execute(
        self, action: str, argument: int, *, intent: str, expected_state: int
    ) -> ExecutionEvidence:
        if type(argument) is not int or type(expected_state) is not int:
            raise ValueError("Argument and expected state must be integers")
        before = self._state
        result: int | None = None
        error: str | None = None
        try:
            if action == "add":
                result = before + argument
            elif action == "multiply":
                result = before * argument
            elif action == "set":
                result = argument
            elif action == "divide":
                if argument == 0 or before % argument:
                    raise ValueError("Division must be nonzero and exact")
                result = before // argument
            else:
                raise ValueError(f"Unknown action: {action}")
            self._state = result
        except ValueError as exc:
            error = str(exc)
        evidence = ExecutionEvidence(
            len(self._trace),
            intent,
            action,
            argument,
            before,
            self._state,
            result,
            error,
            expected_state,
            error is None and self._state == expected_state,
            1.0,
            "toy-integer-register/exact-state/v1",
            _state_hash(before),
            _state_hash(self._state),
        )
        self._trace.append(evidence)
        return evidence


def _state_hash(state: int) -> str:
    return hashlib.sha256(str(state).encode("ascii")).hexdigest()
