"""Diagnostic evidence is separate from control. Confidence is a heuristic score."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

from agg.telemetry.controller import Observation

from .config import ControllerConfig
from .temporal import TemporalSummary


class DiagnosisType(StrEnum):
    HEALTHY_PROGRESS = "healthy_progress"
    PLATEAU = "plateau"
    UNSTABLE_TRAINING = "unstable_training"
    REPRESENTATION_DRIFT = "representation_drift"
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"
    CAPABILITY_REGRESSION = "capability_regression"
    MEMORIZATION_DOMINANCE = "memorization_dominance"
    RETRIEVAL_FAILURE = "retrieval_failure"
    REPLAY_INSUFFICIENCY = "replay_insufficiency"
    ABSTRACTION_FAILURE = "abstraction_failure"
    ABSTRACTION_OVERGENERALIZATION = "abstraction_overgeneralization"
    ABSTRACTION_UNDERCOVERAGE = "abstraction_undercoverage"
    SKILL_COMPOSITION_FAILURE = "skill_composition_failure"
    CAPACITY_SATURATION = "capacity_saturation"
    GATE_SATURATION = "gate_saturation"
    EXCESSIVE_REASONING = "excessive_reasoning"
    INSUFFICIENT_REASONING = "insufficient_reasoning"
    EXPLORATION_COLLAPSE = "exploration_collapse"
    EXCESSIVE_EXPLORATION = "excessive_exploration"
    POSSIBLE_SHORTCUT_LEARNING = "possible_shortcut_learning"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNKNOWN_ANOMALY = "unknown_anomaly"


@dataclass(frozen=True)
class Diagnosis:
    kind: DiagnosisType
    confidence: float
    supporting_signals: dict[str, float | str]
    contradicting_signals: dict[str, float | str]
    step: int
    suggested_families: tuple[str, ...]
    severity: float
    reversibility: str = "stage privately; retain the accepted state until evaluation passes"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Diagnosis":
        fields = dict(raw)
        fields["kind"] = DiagnosisType(fields["kind"])
        fields["suggested_families"] = tuple(fields["suggested_families"])
        return cls(**fields)


class DiagnosticProvider(Protocol):
    """Statistical/learned/JEV providers return evidence, never execute changes."""

    def diagnose(
        self, observation: Observation, temporal: dict[str, TemporalSummary]
    ) -> list[Diagnosis]: ...


class RuleDiagnosis:
    def __init__(self, config: ControllerConfig):
        self.config = config

    def diagnose(
        self, observation: Observation, temporal: dict[str, TemporalSummary]
    ) -> list[Diagnosis]:
        c = self.config
        D = DiagnosisType
        results: list[Diagnosis] = []
        if observation.out_of_distribution or observation.contradictory:
            return [
                Diagnosis(
                    D.UNKNOWN_ANOMALY,
                    0.0,
                    {"quality": "out-of-distribution or contradictory observations"},
                    {},
                    observation.step,
                    ("observe", "escalation"),
                    0.5,
                )
            ]

        def emit(
            kind: DiagnosisType,
            names: tuple[str, ...],
            families: tuple[str, ...],
            severity: float = 0.5,
            contradictions: dict[str, float | str] | None = None,
        ) -> None:
            summaries = [temporal[name] for name in names if name in temporal]
            if len(summaries) != len(names) or any(
                s.samples < c.temporal.min_samples for s in summaries
            ):
                return
            confidence = min(s.confidence for s in summaries)
            if any(name in observation.proxy_metrics for name in names):
                confidence = min(confidence, c.proxy_confidence_cap)
            if contradictions:
                confidence = min(confidence, c.proxy_confidence_cap)
            evidence: dict[str, float | str] = {}
            for name in names:
                summary = temporal[name]
                evidence[name] = summary.current
                evidence[f"{name}.trend"] = summary.trend
                if summary.smoothed_derivative is not None:
                    evidence[f"{name}.slope"] = summary.smoothed_derivative
            results.append(
                Diagnosis(
                    kind,
                    confidence,
                    evidence,
                    contradictions or {},
                    observation.step,
                    families,
                    severity,
                )
            )

        performance = "performance.task_accuracy"
        progress = temporal.get(performance)
        if progress:
            if progress.trend in {"improving", "improving_but_decelerating"}:
                emit(D.HEALTHY_PROGRESS, (performance,), ("observe",), 0.0)
            elif progress.trend == "plateauing":
                emit(D.PLATEAU, (performance,), ("compute", "optimization"))
            elif progress.trend == "unstable":
                emit(D.UNSTABLE_TRAINING, (performance,), ("optimization", "stability"), 0.8)
            elif progress.trend == "degrading":
                emit(D.CAPABILITY_REGRESSION, (performance,), ("stability",), 0.9)
        for name in ("continual.retained_performance", "continual.exact_retention"):
            signal = temporal.get(name)
            if signal and signal.smoothed_derivative is not None:
                if signal.smoothed_derivative < -c.regression_slope:
                    emit(D.CATASTROPHIC_FORGETTING, (name,), ("replay", "curriculum"), 0.9)

        # Each rule requires repeated evidence. Multiple independent diagnoses coexist.
        rules = (
            (
                D.REPRESENTATION_DRIFT,
                "geometry.representation_drift",
                c.drift_threshold,
                True,
                ("stability",),
            ),
            (
                D.CAPABILITY_REGRESSION,
                "continual.capability_regression",
                c.error_rate_threshold,
                True,
                ("stability",),
            ),
            (
                D.RETRIEVAL_FAILURE,
                "abstraction.retrieval_miss_rate",
                c.error_rate_threshold,
                True,
                ("memory",),
            ),
            (
                D.REPLAY_INSUFFICIENCY,
                "continual.replay_diversity",
                c.diversity_threshold,
                False,
                ("replay",),
            ),
            (
                D.ABSTRACTION_FAILURE,
                "abstraction.contradiction_rate",
                c.error_rate_threshold,
                True,
                ("abstraction",),
            ),
            (
                D.ABSTRACTION_OVERGENERALIZATION,
                "abstraction.false_applicability_rate",
                c.error_rate_threshold,
                True,
                ("abstraction",),
            ),
            (
                D.ABSTRACTION_UNDERCOVERAGE,
                "abstraction.coverage",
                c.coverage_threshold,
                False,
                ("abstraction",),
            ),
            (
                D.SKILL_COMPOSITION_FAILURE,
                "abstraction.composition_failure_rate",
                c.error_rate_threshold,
                True,
                ("curriculum", "abstraction"),
            ),
            (
                D.CAPACITY_SATURATION,
                "resources.adapter_utilization",
                c.saturation_threshold,
                True,
                ("capacity",),
            ),
            (
                D.GATE_SATURATION,
                "gating.saturation",
                c.saturation_threshold,
                True,
                ("optimization",),
            ),
            (
                D.EXPLORATION_COLLAPSE,
                "search.branch_diversity",
                c.diversity_threshold,
                False,
                ("compute",),
            ),
        )
        for kind, name, threshold, high, families in rules:
            signal = temporal.get(name)
            if signal and (
                (signal.rolling_mean > threshold) if high else (signal.rolling_mean < threshold)
            ):
                emit(kind, (name,), families)

        metrics = observation.metrics()
        think = metrics.get("gating.think_frequency")
        gain = metrics.get("search.marginal_gain_per_compute")
        if think is not None and gain is not None:
            if think > c.saturation_threshold and gain < c.marginal_gain_threshold:
                emit(
                    D.EXCESSIVE_REASONING,
                    ("gating.think_frequency", "search.marginal_gain_per_compute"),
                    ("compute",),
                )
            if think < c.diversity_threshold and progress and progress.trend == "plateauing":
                emit(
                    D.INSUFFICIENT_REASONING, ("gating.think_frequency", performance), ("compute",)
                )
        ratio = metrics.get("search.exploit_explore_ratio")
        if ratio is not None and gain is not None:
            if ratio < c.diversity_threshold and gain < c.marginal_gain_threshold:
                emit(
                    D.EXCESSIVE_EXPLORATION,
                    ("search.exploit_explore_ratio", "search.marginal_gain_per_compute"),
                    ("compute",),
                )
        memorization = metrics.get("performance.memorization_score")
        validation = metrics.get("performance.validation_score")
        if memorization is not None and validation is not None:
            if memorization - validation > c.error_rate_threshold:
                emit(
                    D.MEMORIZATION_DOMINANCE,
                    ("performance.memorization_score", "performance.validation_score"),
                    ("curriculum",),
                )
        concentration = metrics.get("attribution.concentration")
        ood = metrics.get("performance.ood_score")
        accuracy = metrics.get(performance)
        if concentration is not None and ood is not None and accuracy is not None:
            if concentration > c.saturation_threshold and accuracy - ood > c.error_rate_threshold:
                emit(
                    D.POSSIBLE_SHORTCUT_LEARNING,
                    ("attribution.concentration", performance, "performance.ood_score"),
                    ("observe", "curriculum"),
                    contradictions={
                        "causality": "concentration and generalization gap do not establish cause"
                    },
                )
        if not results or not any(d.confidence >= c.confidence_threshold for d in results):
            results.append(
                Diagnosis(
                    D.INSUFFICIENT_EVIDENCE,
                    0.0,
                    {"support": "no sufficiently supported actionable pattern"},
                    {},
                    observation.step,
                    ("observe",),
                    0.0,
                )
            )
        return results
