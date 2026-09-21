"""Bounded recommendations contain no model, evaluator or executable callable."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from agg.telemetry.controller import Observation

from .config import ControllerConfig, MetricGuard
from .diagnosis import Diagnosis, DiagnosisType
from .temporal import TemporalSummary


class Action(StrEnum):
    NO_OP = "no_op"
    OBSERVE_MORE = "observe_more"
    INCREASE_EVALUATION_FREQUENCY = "increase_evaluation_frequency"
    INCREASE_REPLAY = "increase_replay"
    DECREASE_REPLAY = "decrease_replay"
    INCREASE_REPLAY_DIVERSITY = "increase_replay_diversity"
    REFRESH_MEMORY_INDEX = "refresh_memory_index"
    RETRIEVE_ALTERNATIVE_MEMORIES = "retrieve_alternative_memories"
    ADVANCE_CURRICULUM = "advance_curriculum"
    HOLD_CURRICULUM = "hold_curriculum"
    BACKTRACK_CURRICULUM = "backtrack_curriculum"
    INCREASE_COMPOSITIONAL_EXAMPLES = "increase_compositional_examples"
    INCREASE_OOD_EXAMPLES = "increase_OOD_examples"
    PROPOSE_ADDITIONAL_ABSTRACTIONS = "propose_additional_abstractions"
    SPECIALIZE_ABSTRACTION = "specialize_abstraction"
    MERGE_ABSTRACTIONS = "merge_abstractions"
    WEAKEN_ABSTRACTION = "weaken_abstraction"
    RETIRE_ABSTRACTION = "retire_abstraction"
    TEST_ABSTRACTION_COMPOSITION = "test_abstraction_composition"
    WITHHOLD_ABSTRACTION = "withhold_abstraction"
    INCREASE_REASONING_BUDGET = "increase_reasoning_budget"
    DECREASE_REASONING_BUDGET = "decrease_reasoning_budget"
    INCREASE_EXPLORATION_BUDGET = "increase_exploration_budget"
    DECREASE_EXPLORATION_BUDGET = "decrease_exploration_budget"
    INCREASE_PARALLELISM = "increase_parallelism"
    DECREASE_PARALLELISM = "decrease_parallelism"
    EXPAND_ADAPTER_CAPACITY = "expand_adapter_capacity"
    ALLOCATE_FRESH_ADAPTER = "allocate_fresh_adapter"
    MERGE_ADAPTER = "merge_adapter"
    REDUCE_ACTIVE_CAPACITY = "reduce_active_capacity"
    ADJUST_LEARNING_RATE = "adjust_learning_rate"
    ADJUST_REGULARIZATION = "adjust_regularization"
    ADJUST_GRN_STRENGTH = "adjust_GRN_strength"
    ADJUST_CLIPPING = "adjust_clipping"
    ADJUST_BATCH_MIX = "adjust_batch_mix"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    TRIGGER_REGRESSION_SUITE = "trigger_regression_suite"
    TRIGGER_CONTINUAL_LEARNING_REPAIR = "trigger_continual_learning_repair"
    REQUEST_HUMAN_REVIEW = "request_human_review"


# Exhaustive action names are not execution authority. Only these numeric targets
# may be staged by the initial controller; all others require a reviewed extension.
MUTABLE_TARGETS = {
    Action.INCREASE_REPLAY: "memory.replay_fraction",
    Action.DECREASE_REPLAY: "memory.replay_fraction",
    Action.INCREASE_REPLAY_DIVERSITY: "memory.replay_diversity",
    Action.INCREASE_REASONING_BUDGET: "runtime.reasoning_budget",
    Action.DECREASE_REASONING_BUDGET: "runtime.reasoning_budget",
    Action.INCREASE_EXPLORATION_BUDGET: "runtime.exploration_budget",
    Action.DECREASE_EXPLORATION_BUDGET: "runtime.exploration_budget",
    Action.ADJUST_LEARNING_RATE: "training.learning_rate",
    Action.ADJUST_REGULARIZATION: "training.weight_decay",
}
OBSERVATIONAL = {Action.NO_OP, Action.OBSERVE_MORE, Action.REQUEST_HUMAN_REVIEW}
EXPENSIVE = {
    Action.EXPAND_ADAPTER_CAPACITY,
    Action.ALLOCATE_FRESH_ADAPTER,
    Action.MERGE_ADAPTER,
    Action.INCREASE_PARALLELISM,
}


@dataclass(frozen=True)
class InterventionProposal:
    id: str
    diagnosis: Diagnosis
    target_component: str
    action: Action
    parameters: dict[str, float]
    expected_effect: str
    risk: str
    reversibility: str
    baseline_metrics: dict[str, float]
    protected_metrics: tuple[MetricGuard, ...]
    evaluation_window: int
    target_metric: str
    higher_is_better: bool
    step: int
    evaluation_samples: int = 2
    baseline_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterventionProposal":
        raw = dict(value)
        raw["action"] = Action(raw["action"])
        raw["diagnosis"] = Diagnosis.from_dict(raw["diagnosis"])
        raw["protected_metrics"] = tuple(MetricGuard(**g) for g in raw["protected_metrics"])
        return cls(**raw)


class ComputeAllocationPolicy(Protocol):
    def recommend(self, observation: Observation, progress: TemporalSummary | None) -> Action: ...


class SlopeComputePolicy:
    def __init__(self, config: ControllerConfig):
        self.config = config

    def recommend(self, observation: Observation, progress: TemporalSummary | None) -> Action:
        search = observation.search
        if progress is None or progress.confidence < self.config.confidence_threshold:
            return Action.OBSERVE_MORE
        if search.risk is not None and search.risk >= self.config.confidence_threshold:
            return Action.REQUEST_HUMAN_REVIEW
        if progress.trend in {"degrading", "unstable"}:
            return Action.OBSERVE_MORE
        if search.uncertainty is not None and search.uncertainty > self.config.confidence_threshold:
            return Action.OBSERVE_MORE
        if (
            search.marginal_gain_per_compute is not None
            and search.marginal_gain_per_compute < self.config.marginal_gain_threshold
            and search.branch_diversity is not None
            and search.branch_diversity > self.config.diversity_threshold
        ):
            return Action.DECREASE_EXPLORATION_BUDGET
        if (
            search.difficulty is not None
            and search.difficulty > self.config.coverage_threshold
            and progress.trend == "plateauing"
        ):
            return Action.INCREASE_REASONING_BUDGET
        if progress.trend in {"plateauing", "improving_but_decelerating"}:
            return Action.INCREASE_EXPLORATION_BUDGET
        return Action.DECREASE_EXPLORATION_BUDGET


class InterventionPolicy:
    def __init__(self, config: ControllerConfig, compute: ComputeAllocationPolicy | None = None):
        self.config = config
        self.compute = compute or SlopeComputePolicy(config)
        self.failed_actions: dict[DiagnosisType, set[Action]] = {}

    def feedback(self, proposal: InterventionProposal, status: str) -> None:
        if status == "rolled_back":
            self.failed_actions.setdefault(proposal.diagnosis.kind, set()).add(proposal.action)

    def propose(
        self,
        observation: Observation,
        temporal: dict[str, TemporalSummary],
        diagnoses: list[Diagnosis],
        *,
        sequence: int,
    ) -> InterventionProposal:
        c = self.config
        credible = [d for d in diagnoses if d.confidence >= c.confidence_threshold]
        diagnosis = max(credible or diagnoses, key=lambda d: (d.severity, d.confidence))
        action = Action.OBSERVE_MORE
        target = "performance.task_accuracy"
        higher = True
        D = DiagnosisType
        choices = {
            D.CATASTROPHIC_FORGETTING: (Action.INCREASE_REPLAY, "continual.retained_performance"),
            D.REPLAY_INSUFFICIENCY: (
                Action.INCREASE_REPLAY_DIVERSITY,
                "continual.retained_performance",
            ),
            D.RETRIEVAL_FAILURE: (Action.REFRESH_MEMORY_INDEX, "abstraction.retrieval_miss_rate"),
            D.ABSTRACTION_OVERGENERALIZATION: (
                Action.SPECIALIZE_ABSTRACTION,
                "abstraction.false_applicability_rate",
            ),
            D.ABSTRACTION_UNDERCOVERAGE: (
                Action.PROPOSE_ADDITIONAL_ABSTRACTIONS,
                "abstraction.coverage",
            ),
            D.ABSTRACTION_FAILURE: (Action.WITHHOLD_ABSTRACTION, "abstraction.contradiction_rate"),
            D.SKILL_COMPOSITION_FAILURE: (
                Action.TEST_ABSTRACTION_COMPOSITION,
                "abstraction.composition_failure_rate",
            ),
            D.CAPACITY_SATURATION: (Action.EXPAND_ADAPTER_CAPACITY, "performance.task_accuracy"),
            D.UNSTABLE_TRAINING: (Action.ADJUST_LEARNING_RATE, "performance.task_loss"),
            D.CAPABILITY_REGRESSION: (Action.TRIGGER_REGRESSION_SUITE, "performance.task_accuracy"),
            D.REPRESENTATION_DRIFT: (
                Action.TRIGGER_REGRESSION_SUITE,
                "geometry.representation_drift",
            ),
            D.EXCESSIVE_REASONING: (Action.DECREASE_REASONING_BUDGET, "performance.task_accuracy"),
            D.INSUFFICIENT_REASONING: (
                Action.INCREASE_REASONING_BUDGET,
                "performance.task_accuracy",
            ),
            D.EXPLORATION_COLLAPSE: (
                Action.INCREASE_EXPLORATION_BUDGET,
                "performance.task_accuracy",
            ),
            D.EXCESSIVE_EXPLORATION: (
                Action.DECREASE_EXPLORATION_BUDGET,
                "performance.task_accuracy",
            ),
            D.MEMORIZATION_DOMINANCE: (Action.INCREASE_OOD_EXAMPLES, "performance.ood_score"),
        }
        if credible:
            if diagnosis.kind in choices:
                action, target = choices[diagnosis.kind]
            elif diagnosis.kind == D.PLATEAU:
                action = self.compute.recommend(observation, temporal.get(target))
            elif diagnosis.kind == D.HEALTHY_PROGRESS:
                action = Action.NO_OP
        if action in EXPENSIVE and diagnosis.confidence < c.expensive_confidence:
            action = Action.OBSERVE_MORE
        failed = self.failed_actions.get(diagnosis.kind, set())
        if action in failed:
            action = (
                Action.INCREASE_REPLAY_DIVERSITY
                if action == Action.INCREASE_REPLAY
                and Action.INCREASE_REPLAY_DIVERSITY not in failed
                else Action.OBSERVE_MORE
            )
        if target in c.minimize_metrics:
            higher = False
        if target == "continual.retained_performance" and target not in observation.metrics():
            target = "continual.exact_retention"
        fraction = c.max_fraction * diagnosis.confidence
        if action == Action.ADJUST_LEARNING_RATE or action.value.startswith("decrease_"):
            fraction = -fraction
        return InterventionProposal(
            f"agg-{observation.step}-{sequence}",
            diagnosis,
            MUTABLE_TARGETS.get(action, action.value),
            action,
            {"fraction": fraction} if action in MUTABLE_TARGETS else {},
            f"improve {target}; verify protected metrics independently",
            "high" if action in EXPENSIVE else "bounded",
            "discard private candidate; accepted state remains available",
            observation.metrics(),
            c.protected_metrics,
            c.evaluation_window,
            target,
            higher,
            observation.step,
            evaluation_samples=c.evaluation_samples,
            baseline_provenance=observation.provenance,
        )
