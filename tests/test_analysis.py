import pytest

from agg.experiments.analysis import candidate_changes, phase_surface, telemetry_redundancy


def test_changes_are_candidates_at_the_original_training_step():
    points = candidate_changes([0, 10, 20, 30, 40, 50], [0, 0, 0, 1, 1, 1], window=2, threshold=0.9)
    assert points == [
        {
            "step": 30,
            "before_mean": 0.0,
            "after_mean": 1.0,
            "delta": 1.0,
            "label": "candidate_change_not_confirmed_transition",
        }
    ]


def test_redundancy_retains_constant_and_missing_columns_explicitly():
    result = telemetry_redundancy(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "c": [1.0, 1.0, 1.0]}
    )
    assert result["correlation"][0][1] == pytest.approx(1)
    assert result["correlation"][2][0] is None
    assert result["components"] == ["a", "b", "c"]


def test_phase_surface_keeps_censored_seeds():
    factors = {"gate": 0.5, "distance": 3, "density": 0.0, "context_length": 8}
    rows = [
        {**factors, "seed": 0, "crossing": {"stable_crossing": 20, "last_step": 30}},
        {**factors, "seed": 1, "crossing": {"stable_crossing": None, "last_step": 30}},
    ]
    surface = phase_surface(rows)
    assert len(surface) == 1
    assert surface[0]["censored_seeds"] == 1
    assert surface[0]["observed_median_step"] == 20
    assert surface[0]["seed_steps"] == {"0": 20, "1": None}
    with pytest.raises(ValueError):
        phase_surface(rows + rows[:1])
