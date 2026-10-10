import json
import random
from dataclasses import replace

import numpy as np
import pytest

from agg.experiments.empowerment import EmpowermentConfig, run_empowerment


def test_policy_cap_rejects_before_output_and_allows_retry(tmp_path):
    output = tmp_path / "nested" / "run"
    config = EmpowermentConfig(
        fixtures=("grid", "fork"),
        exhaustive_fixtures=("fork",),
        max_policies=7,
        all_starts=False,
    )
    with pytest.raises(ValueError, match="8 policies exceed enumeration cap 7"):
        run_empowerment(config, output)
    assert not output.exists()
    assert not (output / "config.json").exists()
    assert not output.parent.exists()

    result = run_empowerment(replace(config, max_policies=8), output)
    assert result["environments"][1]["starts"][0]["policy_count"] == 8
    assert (output / "summary.json").exists()


def test_all_exhaustive_caps_are_checked_before_measurements(tmp_path, monkeypatch):
    import agg.experiments.empowerment as empowerment

    def unexpected_measurement(*args, **kwargs):
        pytest.fail("measurement started before all policy caps were checked")

    monkeypatch.setattr(empowerment, "potential_empowerment", unexpected_measurement)
    config = EmpowermentConfig(
        fixtures=("fork", "grid"), exhaustive_fixtures=("fork", "grid"), max_policies=8
    )
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="15625 policies exceed enumeration cap 8"):
        run_empowerment(config, output)
    assert not output.exists()
    assert not (output / "config.json").exists()


def test_report_replay_scope_reward_controls_and_rng(tmp_path):
    config = EmpowermentConfig(fixtures=("fork", "distractors"), all_starts=True)
    assert EmpowermentConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    first = run_empowerment(config, tmp_path / "one")
    second = run_empowerment(config, tmp_path / "two")
    assert first["environments"] == second["environments"]
    assert first["schema_version"] == "agg.empowerment/1"
    assert first["budget"]["executed_environment_actions"] == 0
    assert first["budget"]["training_updates"] == 0
    assert first["checkpoint_hashes"] is None
    assert first["sampling_uncertainty"] is None
    assert random_before == random.getstate()
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]
    fork, distractors = first["environments"]
    assert fork["shortest_action_distances"][1][0] is None
    assert fork["starts"][0]["upper_bound_scope"] == "full_finite_mdp"
    negative = distractors["negative_control"]
    assert negative["restricted_effective_empowerment"] > 1
    assert negative["restricted_best_task_reward"] == 0
    assert negative["full_best_task_reward"] > 0.8
    assert negative["constant_reward_adaptation_advantage"] == pytest.approx(0)
    # No NaN/Infinity and no overwrite of evidence.
    json.dumps(first, allow_nan=False)
    assert json.loads((tmp_path / "one" / "summary.json").read_text()) == first
    with pytest.raises(FileExistsError):
        run_empowerment(config, tmp_path / "one")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixtures": ()},
        {"fixtures": ("unknown",)},
        {"gamma": 1},
        {"max_policies": 0},
        {"all_starts": "yes"},
        {"capacity_iterations": True},
        {"exhaustive_fixtures": ("grid",), "fixtures": ("fork",)},
    ],
)
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        EmpowermentConfig(**kwargs)


def test_reciprocal_overflow_is_distinct_from_absent_support(tmp_path):
    result = run_empowerment(
        EmpowermentConfig(
            fixtures=("gateway",), exhaustive_fixtures=(), gamma=1e-52, all_starts=False
        ),
        tmp_path / "tiny",
    )
    row = result["environments"][0]["starts"][0]
    reasons = row["rollout_hitting_unavailable_reasons"]
    assert "exceeds_float64_range" in reasons
    for count, log_count, reason in zip(
        row["independent_rollout_hitting_counts"],
        row["log_independent_rollout_hitting_counts"],
        reasons,
        strict=True,
    ):
        if reason == "exceeds_float64_range":
            assert count is None and np.isfinite(log_count) and log_count > 709
    assert (tmp_path / "tiny" / "summary.json").exists()
