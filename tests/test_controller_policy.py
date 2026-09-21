from dataclasses import replace


def trajectory(**kwargs):
    from agg.controller.config import ControllerConfig
    from agg.controller.diagnosis import RuleDiagnosis
    from agg.controller.temporal import TemporalTelemetry
    from agg.telemetry.controller import Continual, Observation, Performance

    config = ControllerConfig()
    temporal = TemporalTelemetry(config.temporal)
    for step in range(9):
        item = Observation(
            step,
            performance=Performance(task_accuracy=0.4 + step * 0.04),
            continual=Continual(retained_performance=0.9 - step * 0.025),
            **kwargs,
        )
        features = temporal.update(item)
    return config, item, features, RuleDiagnosis(config).diagnose(item, features)


def test_multiple_diagnoses_and_weak_evidence():
    from agg.controller.diagnosis import DiagnosisType, RuleDiagnosis
    from agg.telemetry.controller import Observation

    config, item, features, diagnoses = trajectory()
    kinds = {d.kind for d in diagnoses}
    assert DiagnosisType.HEALTHY_PROGRESS in kinds
    assert DiagnosisType.CATASTROPHIC_FORGETTING in kinds
    assert all(d.supporting_signals for d in diagnoses)
    assert RuleDiagnosis(config).diagnose(Observation(0), {})[0].kind == (
        DiagnosisType.INSUFFICIENT_EVIDENCE
    )
    unknown = RuleDiagnosis(config).diagnose(replace(item, out_of_distribution=True), features)
    assert unknown[0].kind == DiagnosisType.UNKNOWN_ANOMALY
    assert unknown[0].confidence < config.confidence_threshold


def test_proposal_is_bounded_and_unknown_only_observes():
    from agg.controller.diagnosis import RuleDiagnosis
    from agg.controller.policy import Action, InterventionPolicy
    from agg.telemetry.controller import Observation

    config, item, features, diagnoses = trajectory()
    policy = InterventionPolicy(config)
    proposal = policy.propose(item, features, diagnoses, sequence=1)
    assert proposal.action == Action.INCREASE_REPLAY
    assert 0 < proposal.parameters["fraction"] <= config.max_fraction
    assert proposal.target_metric == "continual.retained_performance"
    assert proposal.protected_metrics == config.protected_metrics
    empty = Observation(0)
    uncertain = policy.propose(empty, {}, RuleDiagnosis(config).diagnose(empty, {}), sequence=2)
    assert uncertain.action == Action.OBSERVE_MORE


def test_proposal_serialization():
    from agg.controller.policy import InterventionPolicy, InterventionProposal

    config, item, features, diagnoses = trajectory()
    proposal = InterventionPolicy(config).propose(item, features, diagnoses, sequence=7)
    assert InterventionProposal.from_dict(proposal.to_dict()) == proposal


def test_expensive_interventions_require_stronger_evidence():
    from agg.controller.config import ControllerConfig
    from agg.controller.diagnosis import Diagnosis, DiagnosisType
    from agg.controller.policy import Action, InterventionPolicy
    from agg.telemetry.controller import Observation, Performance

    policy = InterventionPolicy(ControllerConfig())
    observation = Observation(10, performance=Performance(task_accuracy=0.5))
    diagnosis = Diagnosis(
        DiagnosisType.CAPACITY_SATURATION,
        0.8,
        {"resources.adapter_utilization": 0.99},
        {},
        10,
        ("capacity",),
        0.8,
    )
    assert policy.propose(observation, {}, [diagnosis], sequence=1).action == Action.OBSERVE_MORE
    confident = replace(diagnosis, confidence=0.95)
    assert (
        policy.propose(observation, {}, [confident], sequence=2).action
        == Action.EXPAND_ADAPTER_CAPACITY
    )
