"""Single-owner, one-candidate controller with explicit audit and trust boundaries.

Providers are trusted host code. They must stage private state, retain a rollback
snapshot until outcome logging succeeds, and never modify evaluator, reward or security
policy. Python interfaces alone cannot sandbox a malicious provider.
"""

import copy
import math
from dataclasses import asdict, dataclass
from typing import Protocol

from agg.telemetry.controller import Observation

from .config import ControllerConfig
from .diagnosis import DiagnosticProvider, RuleDiagnosis
from .evaluation import EvaluationWindow, regression_reasons
from .events import EventLog, EventType
from .policy import (
    EXPENSIVE,
    MUTABLE_TARGETS,
    OBSERVATIONAL,
    Action,
    InterventionPolicy,
    InterventionProposal,
)
from .resources import ResourceProfile
from .temporal import TemporalTelemetry


class ExecutionProvider(Protocol):
    @property
    def supported_actions(self) -> set[Action]: ...
    def stage(self, proposal: InterventionProposal) -> None: ...
    def evaluate(self) -> EvaluationWindow: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def finalize(self) -> None:
        """Release rollback state after the controller records a successful commit."""
        ...


@dataclass(frozen=True)
class Outcome:
    proposal_id: str
    status: str
    reasons: tuple[str, ...] = ()


@dataclass
class Pending:
    proposal: InterventionProposal
    provider: ExecutionProvider


class Controller:
    def __init__(
        self,
        config: ControllerConfig,
        events: EventLog,
        diagnostic: DiagnosticProvider | None = None,
        policy: InterventionPolicy | None = None,
    ):
        self.config, self.events = config, events
        self.diagnostic = diagnostic or RuleDiagnosis(config)
        self.policy = policy or InterventionPolicy(config)
        self.temporal = TemporalTelemetry(config.temporal)
        self._issued: InterventionProposal | None = None
        self._anchor: dict[str, float] = {}
        self._anchor_locked = False
        self._pending: Pending | None = None
        self._sequence = 0
        self._clock = -1
        self._last_intervention: int | None = None
        self.blocked = False

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def observe(self, observation: Observation) -> InterventionProposal:
        if observation.step <= self._clock:
            raise ValueError("observation must follow all prior observation/evaluation steps")
        features = self.temporal.update(observation)
        self._clock = observation.step
        self.events.emit(
            EventType.TELEMETRY,
            observation.step,
            {
                "observation": observation.to_dict(),
                "temporal": {name: asdict(summary) for name, summary in features.items()},
            },
        )
        diagnoses = self.diagnostic.diagnose(observation, features)
        if not diagnoses:
            raise ValueError("diagnostic providers must return evidence or an explicit unknown")
        for diagnosis in diagnoses:
            if (
                not math.isfinite(diagnosis.confidence)
                or not 0 <= diagnosis.confidence <= 1
                or not math.isfinite(diagnosis.severity)
                or not 0 <= diagnosis.severity <= 1
                or diagnosis.step != observation.step
            ):
                raise ValueError("invalid diagnostic confidence, severity or step")
            self.events.emit(EventType.DIAGNOSIS, observation.step, diagnosis.to_dict())
        metrics = observation.metrics()
        # Capture capability at first intervention, not an early-training weak state.
        # Candidate observations and subsequent trials cannot lower this anchor.
        if (
            not self._anchor_locked
            and not observation.out_of_distribution
            and not observation.contradictory
        ):
            for guard in self.config.protected_metrics:
                if guard.name in metrics:
                    self._anchor[guard.name] = metrics[guard.name]
        self._sequence += 1
        proposal = self.policy.propose(observation, features, diagnoses, sequence=self._sequence)
        self.events.emit(EventType.PROPOSAL, observation.step, proposal.to_dict(), proposal.id)
        if (
            proposal.baseline_metrics != metrics
            or proposal.baseline_provenance != observation.provenance
            or proposal.step != observation.step
            or proposal.diagnosis not in diagnoses
        ):
            self._issued = None
            self._outcome(
                proposal, "rejected", ["policy baseline or diagnosis differs from evidence"]
            )
            raise ValueError("policy baseline or diagnosis differs from observed evidence")
        if proposal.higher_is_better != (
            proposal.target_metric not in self.config.minimize_metrics
        ):
            self._issued = None
            self._outcome(proposal, "rejected", ["policy cannot redefine metric direction"])
            raise ValueError("policy cannot redefine metric direction")
        self._issued = copy.deepcopy(proposal)
        return proposal

    def _outcome(
        self, proposal: InterventionProposal, status: str, reasons: list[str] | tuple[str, ...] = ()
    ) -> Outcome:
        outcome = Outcome(proposal.id, status, tuple(reasons))
        self.events.emit(EventType.OUTCOME, self._clock, asdict(outcome), proposal.id)
        return outcome

    def start(self, proposal: InterventionProposal, provider: ExecutionProvider) -> Outcome:
        # Record even forged/rejected attempts, without granting their fields authority.
        self.events.emit(
            EventType.EXECUTION,
            self._clock,
            {"status": "validation", "proposal": proposal.to_dict()},
            proposal.id,
        )
        if self.blocked:
            return self._outcome(
                proposal, "blocked", ["human review required after transaction failure"]
            )
        if self.pending:
            return self._outcome(proposal, "rejected", ["one intervention is already pending"])
        if self._issued is None or proposal != self._issued or proposal.step != self._clock:
            return self._outcome(
                proposal, "rejected", ["proposal is stale, altered or not issued here"]
            )
        # Consume this issuance even if validation fails; no replay of an old attempt.
        self._issued = None
        if proposal.action in OBSERVATIONAL:
            return self._outcome(
                proposal,
                "human_review" if proposal.action == Action.REQUEST_HUMAN_REVIEW else "observed",
            )
        c = self.config
        if (
            self._last_intervention is not None
            and self._clock - self._last_intervention < c.cooldown
        ):
            return self._outcome(proposal, "rejected", ["intervention cooldown is active"])
        threshold = (
            c.expensive_confidence if proposal.action in EXPENSIVE else c.confidence_threshold
        )
        if proposal.diagnosis.confidence < threshold:
            return self._outcome(proposal, "rejected", ["insufficient evidence for action cost"])
        if (
            proposal.action not in MUTABLE_TARGETS
            or proposal.action not in provider.supported_actions
        ):
            return self._outcome(
                proposal, "unsupported", ["no authorized execution mechanism for action"]
            )
        fraction = proposal.parameters.get("fraction")
        if (
            proposal.target_component != MUTABLE_TARGETS[proposal.action]
            or set(proposal.parameters) != {"fraction"}
            or not isinstance(fraction, (float, int))
            or isinstance(fraction, bool)
            or not math.isfinite(fraction)
            or not 0 < abs(fraction) <= c.max_fraction
        ):
            return self._outcome(
                proposal, "rejected", ["target or parameters violate mutation allow-list"]
            )
        if proposal.action.value.startswith("increase_") and fraction < 0:
            return self._outcome(
                proposal, "rejected", ["increase action requires positive fraction"]
            )
        if proposal.action.value.startswith("decrease_") and fraction > 0:
            return self._outcome(
                proposal, "rejected", ["decrease action requires negative fraction"]
            )
        if (
            proposal.protected_metrics != c.protected_metrics
            or proposal.evaluation_window != c.evaluation_window
            or proposal.evaluation_samples != c.evaluation_samples
        ):
            return self._outcome(
                proposal, "rejected", ["proposal cannot change acceptance configuration"]
            )
        required = {proposal.target_metric} | {g.name for g in c.protected_metrics}
        if required - proposal.baseline_metrics.keys() or any(
            g.name not in self._anchor for g in c.protected_metrics
        ):
            return self._outcome(
                proposal, "rejected", ["baseline or protected evidence unavailable"]
            )
        # Write intent before any side effect. An unavailable audit sink prevents stage.
        if c.resources is not None and (
            c.resources.max_training_updates is not None
            or c.resources.max_training_tokens is not None
        ):
            limits = ResourceProfile(
                max_training_updates=c.resources.max_training_updates,
                max_training_tokens=c.resources.max_training_tokens,
            )
            preflight = getattr(provider, "preflight_resources", None)
            try:
                assessment = limits.assess(preflight(proposal) if callable(preflight) else {})
            except Exception as exc:
                return self._outcome(proposal, "rejected", [f"resource preflight failed: {exc}"])
            self.events.emit(
                EventType.EXECUTION,
                self._clock,
                {"status": "resource_preflight", "assessment": assessment},
                proposal.id,
            )
            if not assessment["joint_feasible"]:
                return self._outcome(proposal, "rejected", ["training budget preflight failed"])
        self.events.emit(
            EventType.EXECUTION,
            self._clock,
            {"status": "staging", "parameters": proposal.parameters},
            proposal.id,
        )
        self._pending = Pending(copy.deepcopy(proposal), provider)
        self._last_intervention = self._clock
        self._anchor_locked = True
        try:
            provider.stage(copy.deepcopy(proposal))
            self.events.emit(EventType.EXECUTION, self._clock, {"status": "staged"}, proposal.id)
        except Exception as exc:
            return self._rollback([f"execution failed: {type(exc).__name__}: {exc}"])
        return Outcome(proposal.id, "pending")

    def finish(self) -> Outcome:
        if self._pending is None:
            raise ValueError("no intervention is pending")
        pending = self._pending
        proposal, c = pending.proposal, self.config
        try:
            window = pending.provider.evaluate()
            # Revalidate mutable nested payloads returned by a provider.
            window = EvaluationWindow(**asdict(window))
            if window.start_step != proposal.step or window.end_step < self._clock:
                raise ValueError("evaluation window does not align with the pending intervention")
            self._clock = window.end_step
            self.events.emit(
                EventType.EXECUTION,
                self._clock,
                {"status": "evaluated", "window": asdict(window)},
                proposal.id,
            )
            protected = regression_reasons(
                proposal.baseline_metrics,
                window.metrics,
                self._anchor,
                c.protected_metrics,
                proposal.target_metric,
                proposal.higher_is_better,
                c.minimum_gain,
                check_target=False,
            )
            # Missing evidence in an unfinished window warrants observation; a measured
            # regression beyond tolerance is sufficient for early rollback.
            measured_regressions = [r for r in protected if r.startswith("protected regression")]
            target_before = proposal.baseline_metrics[proposal.target_metric]
            target_after = window.metrics.get(proposal.target_metric)
            if target_after is not None:
                delta = (target_after - target_before) * (1 if proposal.higher_is_better else -1)
                if delta < -c.rollback_threshold:
                    measured_regressions.append("target regression exceeds rollback threshold")
            if measured_regressions:
                return self._rollback(measured_regressions, regression=True)
            if (
                window.end_step - window.start_step < c.evaluation_window
                or window.samples < c.evaluation_samples
            ):
                return Outcome(proposal.id, "pending", ("collect more evaluation evidence",))
            reasons = regression_reasons(
                proposal.baseline_metrics,
                window.metrics,
                self._anchor,
                c.protected_metrics,
                proposal.target_metric,
                proposal.higher_is_better,
                c.minimum_gain,
            )
            if c.resources is not None:
                assessment = c.resources.assess(window.metrics)
                self.events.emit(
                    EventType.EXECUTION,
                    self._clock,
                    {"status": "resource_evaluated", "assessment": assessment},
                    proposal.id,
                )
                reasons.extend(
                    f"resource requirement failed: {name}"
                    for name, requirement in assessment["requirements"].items()
                    if not requirement["passed"]
                )
            if reasons:
                return self._rollback(reasons, regression=True)
            self.events.emit(
                EventType.EXECUTION, self._clock, {"status": "committing"}, proposal.id
            )
            pending.provider.commit()
            result = self._outcome(proposal, "committed")
            self._pending = None
            self._last_intervention = self._clock
            try:
                pending.provider.finalize()
            except Exception:
                # The audited commit is already accepted. Stop reuse if cleanup fails.
                self.blocked = True
                raise
            return result
        except Exception as exc:
            if self._pending is None:
                raise
            return self._rollback([f"evaluation/commit failed: {type(exc).__name__}: {exc}"])

    def cancel(self, reason: str = "cancelled by host") -> Outcome:
        return self._rollback([reason])

    def _rollback(self, reasons: list[str], *, regression: bool = False) -> Outcome:
        if self._pending is None:
            raise ValueError("no intervention is pending")
        pending = self._pending
        status = "rolled_back"
        try:
            pending.provider.rollback()
        except Exception as exc:
            status = "blocked"
            self.blocked = True
            reasons.append(f"rollback failed; human review required: {type(exc).__name__}: {exc}")
        self._pending = None
        self._last_intervention = self._clock
        self.policy.feedback(pending.proposal, status)
        try:
            if regression:
                self.events.emit(
                    EventType.REGRESSION, self._clock, {"reasons": reasons}, pending.proposal.id
                )
            self.events.emit(
                EventType.ROLLBACK,
                self._clock,
                {"status": status, "reasons": reasons},
                pending.proposal.id,
            )
            return self._outcome(pending.proposal, status, reasons)
        except Exception:
            # An audit failure cannot be reported as a successful logged transaction.
            self.blocked = True
            raise
