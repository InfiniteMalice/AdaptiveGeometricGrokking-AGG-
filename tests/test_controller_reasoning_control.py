from dataclasses import replace

import pytest

from agg.controller.config import ControllerConfig
from agg.controller.policy import Action, InterventionPolicy, SlopeComputePolicy
from agg.controller.temporal import TemporalTelemetry, TimeSeries
from agg.telemetry.controller import Observation, Performance, Reasoning, Search


def supported_progress():
    series = TimeSeries()
    for step in range(9):
        progress = series.update(step, 0.7)
    return progress


def state_observation(**state):
    return Observation(
        8,
        performance=Performance(task_accuracy=0.7, ood_score=0.8),
        reasoning=Reasoning(
            state_displacement=0.01, trajectory_cosine=1, active_evidence_fraction=0.8, **state
        ),
    )


def test_old_configs_remain_disabled_and_policy_state_changes():
    from agg.controller.config import ReasoningPolicyConfig
    from agg.controller.policy import ReasoningComputePolicy

    old = ControllerConfig.from_dict({"cooldown": 4})
    assert not old.reasoning.enabled
    assert isinstance(InterventionPolicy(old).compute, SlopeComputePolicy)
    config = replace(old, reasoning=ReasoningPolicyConfig(enabled=True))
    assert ControllerConfig.from_dict(config.to_dict()) == config
    policy = ReasoningComputePolicy(config)
    progress = supported_progress()
    low = state_observation(predictive_entropy=0.1, reasoning_progress=0)
    high = state_observation(predictive_entropy=2)
    assert policy.recommend(low, progress) == Action.DECREASE_REASONING_BUDGET
    assert policy.recommend(high, progress) == Action.INCREASE_REASONING_BUDGET
    assert policy.recommend(Observation(8), progress) == Action.OBSERVE_MORE
    assert policy.recommend(high, None) == Action.OBSERVE_MORE
    assert (
        policy.recommend(replace(high, search=Search(risk=0.99)), progress)
        == Action.REQUEST_HUMAN_REVIEW
    )
    proxy = replace(high, proxy_metrics=("reasoning.predictive_entropy",))
    assert policy.recommend(proxy, progress) == Action.OBSERVE_MORE
    assert (
        policy.recommend(replace(high, out_of_distribution=True), progress) == Action.OBSERVE_MORE
    )
    narrowing = replace(low, reasoning=replace(low.reasoning, active_evidence_fraction=0.1))
    assert policy.recommend(narrowing, progress) == Action.RETRIEVE_ALTERNATIVE_MEMORIES
    assert (
        policy.recommend(replace(high, search=Search(uncertainty=0.99)), progress)
        == Action.OBSERVE_MORE
    )
    stop_policy = ReasoningComputePolicy(
        replace(config, reasoning=replace(config.reasoning, allow_stop=True))
    )
    assert stop_policy.recommend(low, progress) == Action.STOP_REASONING


def test_default_policy_ignores_reasoning_and_reward_is_not_target():
    from agg.controller.config import ReasoningPolicyConfig
    from agg.controller.diagnosis import RuleDiagnosis

    temporal = TemporalTelemetry()
    config = ControllerConfig()
    for step in range(9):
        obs = replace(state_observation(predictive_entropy=2), step=step)
        summaries = temporal.update(obs)
    diagnoses = RuleDiagnosis(config).diagnose(obs, summaries)
    legacy = InterventionPolicy(config)
    absent = replace(obs, reasoning=Reasoning())
    assert (
        legacy.propose(obs, summaries, diagnoses, sequence=1).action
        == legacy.propose(absent, summaries, diagnoses, sequence=1).action
    )
    enabled = InterventionPolicy(replace(config, reasoning=ReasoningPolicyConfig(enabled=True)))
    proposal = enabled.propose(obs, summaries, diagnoses, sequence=1)
    assert proposal.action == Action.INCREASE_REASONING_BUDGET
    assert proposal.target_metric == "performance.task_accuracy"
    assert 0 < proposal.parameters["fraction"] <= config.max_fraction
    assert obs.performance.reward is None


def test_reasoning_action_uses_existing_protected_rollback(tmp_path):
    from agg.controller.config import ReasoningPolicyConfig
    from agg.controller.core import Controller
    from agg.controller.evaluation import EvaluationWindow
    from agg.controller.events import EventLog
    from agg.ledger import Ledger

    class BudgetTrial:
        supported_actions = {Action.INCREASE_REASONING_BUDGET}
        accepted = 100.0
        candidate = None

        def stage(self, proposal):
            self.proposal = proposal
            self.candidate = self.accepted * (1 + proposal.parameters["fraction"])

        def evaluate(self):
            return EvaluationWindow(
                8, 28, {"performance.task_accuracy": 0.95, "performance.ood_score": 0.4}, 2
            )

        def commit(self):
            self.accepted = self.candidate

        def rollback(self):
            self.candidate = None

        def finalize(self):
            self.candidate = None

    config = ControllerConfig(reasoning=ReasoningPolicyConfig(enabled=True))
    controller = Controller(config, EventLog(Ledger(tmp_path / "events.jsonl"), "test"))
    for step in range(9):
        proposal = controller.observe(replace(state_observation(predictive_entropy=2), step=step))
    provider = BudgetTrial()
    assert provider.accepted == 100
    assert controller.start(proposal, provider).status == "pending"
    assert provider.accepted == 100 and provider.candidate > 100
    assert controller.finish().status == "rolled_back"
    assert provider.accepted == 100 and provider.candidate is None


def test_stop_and_alternate_retrieval_have_no_fake_execution(tmp_path):
    from agg.controller.config import ReasoningPolicyConfig
    from agg.controller.core import Controller
    from agg.controller.events import EventLog
    from agg.ledger import Ledger

    class NoProvider:
        supported_actions = set()

        def stage(self, proposal):
            raise AssertionError("unsupported action cannot stage")

    controller = Controller(
        ControllerConfig(reasoning=ReasoningPolicyConfig(enabled=True, allow_stop=True)),
        EventLog(Ledger(tmp_path / "e.jsonl"), "test"),
    )
    for step in range(9):
        proposal = controller.observe(
            replace(state_observation(predictive_entropy=0.1, reasoning_progress=0), step=step)
        )
    assert proposal.action == Action.STOP_REASONING
    assert controller.start(proposal, NoProvider()).status == "unsupported"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entropy_low": -1},
        {"entropy_high": float("nan")},
        {"minimum_support": 1.1},
        {"enabled": "yes"},
    ],
)
def test_reasoning_policy_config_validation(kwargs):
    from agg.controller.config import ReasoningPolicyConfig

    with pytest.raises(ValueError):
        ReasoningPolicyConfig(**kwargs)
