import json

import numpy as np
import pytest

from agg.credit.multistep import optimize, sample_episodes
from agg.experiments.multistep import MultistepConfig, run_multistep


def test_optimizer_accepts_only_sample_improvement_inside_fixed_behavior_region():
    behavior = np.full(6, 0.5)
    episodes = sample_episodes(behavior, horizon=3, count=64, seed=3)
    result = optimize(behavior, episodes, k=3, gamma=0.9, delta=0.02, inner_steps=8)
    accepted = result["accepted"]
    assert accepted
    assert all(row["maximum_kl"] <= 0.02 for row in accepted)
    assert all(row["after"] > row["before"] for row in accepted)
    assert np.all((result["policy"] >= 1e-4) & (result["policy"] <= 1 - 1e-4))
    assert result["objective_evaluations"] >= len(accepted) + 1
    assert result["wall_seconds"] >= 0
    with pytest.raises(ValueError):
        optimize(behavior, episodes, k=2, gamma=0.9, delta=float("nan"))


def test_benchmark_counts_executed_learning_and_reference_costs(tmp_path):
    config = MultistepConfig(horizons=(3,), seeds=(3,), episodes=8, rounds=2, inner_steps=2)
    result = run_multistep(config, tmp_path / "run")
    assert len(result["runs"]) == 4
    assert result["budget"]["sampled_episodes"] == 64
    assert result["budget"]["sampled_actions"] == 192
    assert result["budget"]["reference_episodes"] == 48
    assert result["budget"]["reference_actions"] == 144
    assert all(row["status"] == "complete" for row in result["runs"])
    diagnostic = next(row for row in result["runs"] if row["arm"] == "discounted_diagnostic")
    assert diagnostic["actor_updates"] == 0
    assert diagnostic["final_policy"] == [0.5] * 6
    assert all(row["final_return"] >= 0 for row in result["runs"])
    again = run_multistep(config, tmp_path / "again")
    assert [r["final_policy"] for r in result["runs"]] == [r["final_policy"] for r in again["runs"]]
    log = [
        json.loads(line) for line in (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(log) == 64 + 48
    assert all(all(t["verified"] for t in row["episode"]["trace"]) for row in log)
    with pytest.raises(FileExistsError):
        run_multistep(config, tmp_path / "run")
