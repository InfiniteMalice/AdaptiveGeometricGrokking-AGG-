import copy
from dataclasses import replace

import pytest


class BudgetBackend:
    """Test-only transactional memory budget, not a replay training implementation."""

    def __init__(self, proposal, metrics, *, elapsed=20, fail=None):
        from agg.controller.evaluation import EvaluationWindow
        from agg.controller.policy import Action

        self.supported_actions = {Action.INCREASE_REPLAY}
        self.accepted = {"replay": 0.2}
        self.candidate = None
        self.before = copy.deepcopy(self.accepted)
        self.window = EvaluationWindow(proposal.step, proposal.step + elapsed, metrics, 2)
        self.fail = fail
        self.calls = []

    def stage(self, proposal):
        self.calls.append("stage")
        self.candidate = {"replay": self.accepted["replay"] * (1 + proposal.parameters["fraction"])}
        if self.fail == "stage":
            raise RuntimeError("partial stage")

    def evaluate(self):
        if self.fail == "evaluate":
            raise RuntimeError("missing evaluator")
        return self.window

    def commit(self):
        self.calls.append("commit")
        self.accepted = self.candidate

    def rollback(self):
        self.calls.append("rollback")
        if self.fail == "rollback":
            raise RuntimeError("storage unavailable")
        self.accepted = self.before
        self.candidate = None


def setup_controller(tmp_path):
    from agg.controller.config import ControllerConfig
    from agg.controller.core import Controller
    from agg.controller.events import EventLog
    from agg.ledger import Ledger
    from agg.telemetry.controller import Continual, Observation, Performance

    ledger = Ledger(tmp_path / "controller.jsonl")
    controller = Controller(ControllerConfig(), EventLog(ledger, "test"))
    for step in range(9):
        proposal = controller.observe(
            Observation(
                step,
                performance=Performance(task_accuracy=0.5 + step * 0.02, ood_score=0.8),
                continual=Continual(retained_performance=0.9 - step * 0.03),
            )
        )
    return controller, proposal, ledger


def test_protected_gate_rolls_back_and_audit_roundtrips(tmp_path):
    from agg.controller.events import Event, EventType

    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(
        proposal, {"continual.retained_performance": 0.85, "performance.ood_score": 0.6}
    )
    assert controller.start(proposal, backend).status == "pending"
    assert controller.finish().status == "rolled_back"
    assert backend.accepted == {"replay": 0.2} and "commit" not in backend.calls
    events = [Event.from_dict(row) for row in ledger.read()]
    assert any(e.kind == EventType.ROLLBACK for e in events)
    assert any(e.kind == EventType.REGRESSION for e in events)


def test_evaluation_window_single_pending_commit_and_cooldown(tmp_path):
    from agg.telemetry.controller import Continual, Observation, Performance

    controller, proposal, ledger = setup_controller(tmp_path)
    metrics = {"continual.retained_performance": 0.85, "performance.ood_score": 0.8}
    backend = BudgetBackend(proposal, metrics, elapsed=5)
    assert controller.start(proposal, backend).status == "pending"
    assert controller.start(proposal, backend).status == "rejected"
    assert controller.finish().status == "pending"
    backend.window = replace(backend.window, end_step=28)
    assert controller.finish().status == "committed"
    assert backend.accepted["replay"] > 0.2
    # Observations cannot move behind the evaluated candidate's finish step.
    next_proposal = controller.observe(
        Observation(
            29,
            performance=Performance(task_accuracy=0.8, ood_score=0.8),
            continual=Continual(retained_performance=0.6),
        )
    )
    assert (
        controller.start(next_proposal, BudgetBackend(next_proposal, metrics)).status == "rejected"
    )
    assert "cooldown" in ledger.read()[-1]["payload"]["reasons"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("parameters", {"fraction": 50}),
        ("target_component", "evaluator.reward"),
        ("protected_metrics", ()),
        ("baseline_metrics", {}),
        ("evaluation_window", 0),
    ],
)
def test_modified_proposals_cannot_weaken_controller_policy(tmp_path, field, value):
    controller, proposal, ledger = setup_controller(tmp_path)
    forged = replace(proposal, **{field: value})
    backend = BudgetBackend(proposal, {})
    assert controller.start(forged, backend).status == "rejected"
    assert backend.calls == []
    assert ledger.read()[-1]["payload"]["reasons"]


@pytest.mark.parametrize("fail", ["stage", "evaluate", "rollback"])
def test_execution_failures_restore_or_block(tmp_path, fail):
    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(proposal, {}, fail=fail)
    started = controller.start(proposal, backend)
    result = started if started.status != "pending" else controller.finish()
    assert result.status in {"rolled_back", "blocked"}
    assert "rollback" in backend.calls
    assert controller.blocked == (fail == "rollback")


def test_missing_protected_metric_cannot_commit(tmp_path):
    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(proposal, {"continual.retained_performance": 0.95})
    controller.start(proposal, backend)
    assert controller.finish().status == "rolled_back"
    assert backend.accepted == backend.before


def test_unsupported_action_is_explicit(tmp_path):
    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(proposal, {})
    backend.supported_actions = set()
    result = controller.start(proposal, backend)
    assert result.status == "unsupported" and backend.calls == []


def test_metric_direction_negative_controls_and_cumulative_anchor():
    from agg.controller.config import MetricGuard
    from agg.controller.evaluation import regression_reasons

    guards = (MetricGuard("false_applicability", 0.02, False), MetricGuard("transfer", 0.02))
    baseline = {"loss": 0.5, "false_applicability": 0.01, "transfer": 0.79}
    anchor = {"false_applicability": 0.01, "transfer": 0.8}
    improved = {"loss": 0.4, "false_applicability": 0.01, "transfer": 0.79}
    assert regression_reasons(baseline, improved, anchor, guards, "loss", False, 0.005) == []
    assert regression_reasons(
        baseline, {**improved, "transfer": 0.77}, anchor, guards, "loss", False, 0.005
    )
    assert regression_reasons(
        baseline, {**improved, "false_applicability": 0.2}, anchor, guards, "loss", False, 0.005
    )


def test_nonfinite_evaluation_and_stale_proposal_fail_closed(tmp_path):
    from agg.telemetry.controller import Observation

    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(
        proposal, {"continual.retained_performance": 0.9, "performance.ood_score": 0.8}
    )
    backend.window.metrics["performance.ood_score"] = float("nan")
    controller.start(proposal, backend)
    assert controller.finish().status == "rolled_back"
    controller.observe(Observation(40))
    assert controller.start(proposal, backend).status == "rejected"


def test_target_regression_aborts_early(tmp_path):
    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(
        proposal, {"continual.retained_performance": 0.1, "performance.ood_score": 0.8}, elapsed=1
    )
    controller.start(proposal, backend)
    assert controller.finish().status == "rolled_back"
    assert backend.calls == ["stage", "rollback"]


def test_commit_partial_failure_rolls_back(tmp_path):
    controller, proposal, ledger = setup_controller(tmp_path)

    class BrokenCommit(BudgetBackend):
        def commit(self):
            super().commit()
            raise RuntimeError("commit storage failed")

    backend = BrokenCommit(
        proposal, {"continual.retained_performance": 0.9, "performance.ood_score": 0.8}
    )
    controller.start(proposal, backend)
    assert controller.finish().status == "rolled_back"
    assert backend.accepted == backend.before


def test_audit_failure_does_not_execute_provider(tmp_path):
    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(proposal, {})

    def unavailable(record):
        raise OSError("disk unavailable")

    ledger.append = unavailable
    with pytest.raises(OSError):
        controller.start(proposal, backend)
    assert backend.calls == []


def test_anchor_captures_capability_when_interventions_begin(tmp_path):
    from agg.controller.config import ControllerConfig
    from agg.controller.core import Controller
    from agg.controller.events import EventLog
    from agg.ledger import Ledger
    from agg.telemetry.controller import Continual, Observation, Performance

    config = ControllerConfig(cooldown=0, evaluation_window=1)
    controller = Controller(config, EventLog(Ledger(tmp_path / "events.jsonl"), "anchor"))
    for step in range(9):
        proposal = controller.observe(
            Observation(
                step,
                performance=Performance(task_accuracy=0.5, ood_score=0.5 if step == 0 else 0.8),
                continual=Continual(retained_performance=0.9 - step * 0.03),
            )
        )
    backend = BudgetBackend(
        proposal, {"continual.retained_performance": 0.8, "performance.ood_score": 0.781}, elapsed=1
    )
    controller.start(proposal, backend)
    assert controller.finish().status == "committed"
    for step in range(10, 19):
        proposal = controller.observe(
            Observation(
                step,
                performance=Performance(task_accuracy=0.5, ood_score=0.781),
                continual=Continual(retained_performance=0.8 - (step - 10) * 0.02),
            )
        )
    backend = BudgetBackend(
        proposal, {"continual.retained_performance": 0.8, "performance.ood_score": 0.762}, elapsed=1
    )
    controller.start(proposal, backend)
    assert controller.finish().status == "rolled_back"


def test_staging_audit_failure_does_not_rollback_an_unstarted_transaction(tmp_path):
    controller, proposal, ledger = setup_controller(tmp_path)
    backend = BudgetBackend(proposal, {})
    append = ledger.append

    def unavailable(record):
        if record["payload"].get("status") == "staging":
            raise OSError("disk unavailable")
        append(record)

    ledger.append = unavailable
    with pytest.raises(OSError):
        controller.start(proposal, backend)
    assert backend.calls == []
