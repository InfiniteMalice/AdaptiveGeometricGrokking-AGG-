"""Auditable discounted return-to-go over verified and bounded auxiliary evidence."""

import math
from dataclasses import dataclass

from agg.supervision import CombinedEvidence, EvidencePolicy, Signal, combine_evidence


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    if not 0 <= gamma <= 1 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("Finite rewards and gamma in [0,1] required")
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        if not math.isfinite(running):
            raise ValueError("Return overflow")
        returns[index] = running
    return returns


@dataclass(frozen=True)
class CreditRecord:
    position: int
    gamma: float
    credit: float
    local_evidence: CombinedEvidence
    future_evidence: tuple[CombinedEvidence, ...]


def assign_credit(
    signals: list[Signal], length: int, gamma: float = 0.9, policy: EvidencePolicy | None = None
) -> list[CreditRecord]:
    if length < 0 or any(s.position >= length for s in signals):
        raise ValueError("Signals must lie within the requested sequence")
    combined = [
        combine_evidence([s for s in signals if s.position == i], policy) for i in range(length)
    ]
    returns = discounted_returns([c.value for c in combined], gamma)
    return [
        CreditRecord(i, gamma, returns[i], combined[i], tuple(combined[i:])) for i in range(length)
    ]
