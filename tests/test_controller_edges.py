from dataclasses import replace

import pytest


@pytest.mark.parametrize("value", ["0.7", True, float("inf")])
def test_schema_rejects_non_numeric_measurements(value):
    from agg.telemetry.controller import Observation, Performance

    with pytest.raises(ValueError):
        Observation(0, performance=Performance(task_accuracy=value))


@pytest.mark.parametrize("kwargs", [{"window": 8.5}, {"min_samples": True}])
def test_temporal_config_requires_integer_counts(kwargs):
    from agg.controller.config import TemporalConfig

    with pytest.raises(ValueError):
        TemporalConfig(**kwargs)


def test_sparse_missing_component_and_linear_trajectory():
    from agg.controller.temporal import TemporalTelemetry, TimeSeries
    from agg.telemetry.controller import Observation, Performance

    temporal = TemporalTelemetry()
    for step in range(9):
        temporal.update(Observation(step, performance=Performance(task_accuracy=0.5)))
    temporal.update(Observation(9))
    result = temporal.update(Observation(10, performance=Performance(task_accuracy=0.5)))
    assert result["performance.task_accuracy"].samples == 1
    with pytest.raises(ValueError, match="component"):
        temporal.update(Observation(11, component="other"))
    line = TimeSeries()
    for step in range(9):
        result = line.update(step, 2 * step + 1)
    assert result.smoothed_derivative == pytest.approx(2)
    assert result.second_derivative == pytest.approx(0, abs=1e-10)


def test_policy_cannot_fabricate_baseline_evidence(tmp_path):
    from agg.controller.config import ControllerConfig
    from agg.controller.core import Controller
    from agg.controller.events import EventLog
    from agg.controller.policy import InterventionPolicy
    from agg.ledger import Ledger
    from agg.telemetry.controller import Observation, Performance

    class BadPolicy(InterventionPolicy):
        def propose(self, observation, temporal, diagnoses, *, sequence):
            proposal = super().propose(observation, temporal, diagnoses, sequence=sequence)
            return replace(proposal, baseline_metrics={"performance.task_accuracy": -100})

    config = ControllerConfig()
    controller = Controller(
        config, EventLog(Ledger(tmp_path / "events.jsonl"), "test"), policy=BadPolicy(config)
    )
    with pytest.raises(ValueError, match="baseline"):
        controller.observe(Observation(0, performance=Performance(task_accuracy=0.5)))


def test_proxy_history_cannot_be_upgraded_by_one_measured_sample():
    from agg.controller.config import ControllerConfig
    from agg.controller.diagnosis import RuleDiagnosis
    from agg.controller.temporal import TemporalTelemetry
    from agg.telemetry.controller import Continual, Observation

    config = ControllerConfig()
    temporal = TemporalTelemetry(config.temporal)
    for step in range(5):
        item = Observation(
            step,
            continual=Continual(retained_performance=0.9 - step * 0.03),
            proxy_metrics=("continual.retained_performance",) if step < 4 else (),
        )
        features = temporal.update(item)
    assert features["continual.retained_performance"].samples == 1
    diagnoses = RuleDiagnosis(config).diagnose(item, features)
    assert all(d.confidence < config.confidence_threshold for d in diagnoses)


@pytest.mark.parametrize("quality", ["contradictory", "out_of_distribution"])
def test_invalid_quality_history_cannot_establish_support(quality):
    from agg.controller.temporal import TemporalTelemetry
    from agg.telemetry.controller import Continual, Observation

    temporal = TemporalTelemetry()
    for step in range(5):
        item = Observation(
            step, continual=Continual(retained_performance=0.9 - step * 0.03), **{quality: step < 4}
        )
        features = temporal.update(item)
    assert features["continual.retained_performance"].samples == 1


def test_curriculum_holds_when_shallowest_stage_is_unmeasured():
    from agg.controller.curriculum import CurriculumGate

    gate = CurriculumGate({0: 0.8, 1: 0.75, 2: 0.7})
    assert gate.evaluate(2, {1: 0.6, 2: 0.9}).action == "hold"
