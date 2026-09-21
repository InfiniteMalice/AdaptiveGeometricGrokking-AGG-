"""Evidence activation anomalies are evaluation prompts, never deception labels."""

import math
from dataclasses import dataclass

from agg.telemetry.controller import Observation, Reasoning

from .routing import RoutingDecision


@dataclass(frozen=True)
class RoutingDiagnostics:
    active_evidence_fraction: float | None
    evidence_turnover: float | None
    reactivated_ids: tuple[str, ...]
    findings: tuple[str, ...]

    def telemetry(self, state: Reasoning) -> Reasoning:
        from dataclasses import replace

        return replace(
            state,
            active_evidence_fraction=self.active_evidence_fraction,
            evidence_turnover=self.evidence_turnover,
        )


def diagnose_routing(
    current: RoutingDecision,
    previous: RoutingDecision | None = None,
    earlier: RoutingDecision | None = None,
    *,
    low_entropy: float = 0.2,
    narrow_fraction: float = 0.25,
    semantic_continuity: float | None = None,
    stopping: bool = False,
) -> RoutingDiagnostics:
    """Turnover is Jaccard distance; first sample/empty eligible fraction is None.

    Empty-to-empty sets have measured turnover zero. Reactivation uses a
    three-sample window. Entropy threshold is in producer units (default nats).
    Semantic continuity, if externally supplied, is a similarity in [-1,1].
    """
    if not math.isfinite(low_entropy) or low_entropy < 0 or not 0 <= narrow_fraction <= 1:
        raise ValueError("invalid routing diagnostic thresholds")
    if semantic_continuity is not None and (
        not math.isfinite(semantic_continuity) or not -1 <= semantic_continuity <= 1
    ):
        raise ValueError("semantic continuity must be finite in [-1,1]")
    if earlier is not None and previous is None:
        raise ValueError("earlier routing requires previous routing")
    history = [d for d in (earlier, previous, current) if d is not None]
    if any(
        a.component != b.component or a.step >= b.step
        for a, b in zip(history, history[1:], strict=False)
    ):
        raise ValueError("routing history requires increasing steps within one component")
    active = set(current.selected_ids)
    fraction = len(active) / len(current.eligible_ids) if current.eligible_ids else None
    now = Observation.from_dict(current.to_dict()["observation"]).reasoning
    findings: list[str] = []
    turnover = None
    reactivated: tuple[str, ...] = ()
    narrow = fraction is not None and fraction < narrow_fraction
    if narrow and now.predictive_entropy is not None and now.predictive_entropy <= low_entropy:
        findings.append("low_entropy_narrow_support")
    if previous is not None:
        old_active = set(previous.selected_ids)
        union = active | old_active
        turnover = len(active ^ old_active) / len(union) if union else 0.0
        removed = old_active - active
        before = Observation.from_dict(previous.to_dict()["observation"]).reasoning
        if (
            len(active) < len(old_active)
            and now.predictive_entropy is not None
            and before.predictive_entropy is not None
            and now.predictive_entropy > before.predictive_entropy
        ):
            findings.append("uncertainty_rising_support_shrinking")
        if removed and now.trajectory_cosine is not None and now.trajectory_cosine < 0:
            findings.append("trajectory_redirection_with_removal")
        if earlier is not None:
            reactivated = tuple(sorted(active & set(earlier.selected_ids) - old_active))
            if reactivated:
                findings.append("retrieval_oscillation")
    if (
        semantic_continuity is not None
        and now.trajectory_cosine is not None
        and semantic_continuity * now.trajectory_cosine < 0
    ):
        findings.append("semantic_internal_continuity_disagreement")
    if stopping and (
        not active or narrow or (now.reasoning_progress is not None and now.reasoning_progress > 0)
    ):
        findings.append("possible_premature_stopping")
    return RoutingDiagnostics(fraction, turnover, reactivated, tuple(findings))
