import json

import pytest


def test_schema_roundtrip_and_missing_values():
    from agg.telemetry.controller import Observation, Performance, Resources

    item = Observation(
        5,
        performance=Performance(task_accuracy=0.7),
        resources=Resources(precision_mode="INT8", storage_encoding="dense"),
    )
    assert Observation.from_dict(json.loads(json.dumps(item.to_dict()))) == item
    assert "continual.retained_performance" not in item.metrics()
    assert item.metrics()["performance.task_accuracy"] == 0.7
    assert item.resources.storage_encoding != item.resources.precision_mode
    with pytest.raises(ValueError):
        Observation(0, performance=Performance(task_loss=float("nan")))


def test_legacy_conversion_preserves_provenance_and_does_not_invent_retention():
    from agg.telemetry import TelemetrySnapshot
    from agg.telemetry.controller import Observation

    old = TelemetrySnapshot(
        8,
        "checkpoint",
        raw={
            "task": {"loss": 0.4, "accuracy": 0.8},
            "ood": {"accuracy": 0.6},
            "memorization": {"score": 0.9},
        },
    )
    new = Observation.from_legacy(old)
    assert new.performance.task_accuracy == 0.8
    assert new.performance.validation_score is None
    assert new.continual.retained_performance is None
    assert new.provenance["legacy"] == old.to_dict()
    assert "performance.memorization_score" in new.proxy_metrics


def test_irregular_quadratic_derivatives_and_support():
    from agg.controller.config import TemporalConfig
    from agg.controller.temporal import TimeSeries

    series = TimeSeries(TemporalConfig(window=9, min_samples=5))
    first = series.update(0, 3)
    assert first.second_derivative is None and first.confidence < 1
    for t in [1, 3, 4, 7, 8, 10]:
        result = series.update(t, 3 + 2 * t + t * t)
    assert result.smoothed_derivative == pytest.approx(22)
    assert result.second_derivative == pytest.approx(2)
    assert result.samples == 7 and result.rolling_variance > 0
    with pytest.raises(ValueError):
        series.update(10, 0)
    with pytest.raises(ValueError):
        series.update(11, float("inf"))


def test_noise_plateau_deceleration_and_gap_reset():
    from agg.controller.config import TemporalConfig
    from agg.controller.temporal import TimeSeries

    config = TemporalConfig(
        window=9, min_samples=5, plateau_slope=0.005, noise_threshold=0.03, max_gap=20
    )
    series = TimeSeries(config)
    for t in range(9):
        result = series.update(t, 0.5 + (0.002 if t % 2 else -0.002))
    assert result.trend == "plateauing"
    assert series.update(100, 0.8).samples == 1
    slow = TimeSeries(config)
    for t in range(9):
        result = slow.update(t, 0.2 + 0.08 * t - 0.003 * t * t)
    assert result.trend == "improving_but_decelerating"


def test_config_rejects_invalid_thresholds():
    from agg.controller.config import ControllerConfig, TemporalConfig

    for kwargs in ({"window": 2}, {"ema_alpha": 0}, {"noise_threshold": float("nan")}):
        with pytest.raises(ValueError):
            TemporalConfig(**kwargs)
    with pytest.raises(ValueError):
        ControllerConfig(confidence_threshold=1.1)
    config = ControllerConfig()
    assert ControllerConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
