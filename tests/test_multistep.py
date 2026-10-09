from dataclasses import replace

import numpy as np
import pytest

from agg.credit.multistep import enumerate_episodes, exact_reference, pdis, sample_episodes


def test_full_horizon_identity_and_single_step_known_reference():
    behavior = np.full(3, 0.5)
    target = np.full(3, 0.75)
    episodes, weights = enumerate_episodes(behavior, horizon=2)
    single = pdis(target, behavior, episodes, k=1, gamma=0.9, weights=weights)
    full = pdis(target, behavior, episodes, k=2, gamma=0.9, weights=weights)
    assert single["value"] == pytest.approx(0.225)
    assert full["value"] == pytest.approx(0.28125)
    assert full["value"] == pytest.approx(
        exact_reference(target, 2, 0.9)[0] - exact_reference(behavior, 2, 0.9)[0]
    )
    assert full["gradient"] == pytest.approx([0.1265625, 0, 0.1265625])
    assert single["gradient"] == pytest.approx([0.084375, 0, 0.084375])
    assert all(all(record.verified for record in episode.trace) for episode in episodes)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_multistep_logit_gradient_matches_finite_differences(k):
    behavior = np.linspace(0.3, 0.6, 10)
    target = np.linspace(0.4, 0.7, 10)
    episodes, weights = enumerate_episodes(behavior, horizon=4)
    result = pdis(target, behavior, episodes, k=k, gamma=0.9, weights=weights)
    theta = np.log(target / (1 - target))
    for i in range(len(target)):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        value_plus = pdis(
            1 / (1 + np.exp(-plus)), behavior, episodes, k=k, gamma=0.9, weights=weights
        )["value"]
        value_minus = pdis(
            1 / (1 + np.exp(-minus)), behavior, episodes, k=k, gamma=0.9, weights=weights
        )["value"]
        assert result["gradient"][i] == pytest.approx((value_plus - value_minus) / 2e-6, abs=1e-9)


def test_executed_sampling_support_and_behavior_identity():
    behavior = np.full(6, 0.5)
    episodes = sample_episodes(behavior, horizon=3, count=20, seed=4)
    assert episodes == sample_episodes(behavior, horizon=3, count=20, seed=4)
    assert all(e.rewards[-1] == float(sum(e.actions) == 3) for e in episodes)
    with pytest.raises(ValueError, match="behavior"):
        pdis(behavior, behavior + 0.1, episodes, k=2, gamma=0.9)
    with pytest.raises(ValueError):
        pdis(np.ones(6), behavior, episodes, k=2, gamma=0.9)
    with pytest.raises(ValueError, match="verified"):
        pdis(behavior, behavior, [replace(episodes[0], rewards=(1.0, 1.0, 1.0))], k=2, gamma=0.9)
    result = pdis(behavior, behavior, episodes, k=2, gamma=0.9)
    assert result["importance"]["maximum"] == 1
    assert result["importance"]["terminal_ess"] == pytest.approx(20)
    assert result["variance"] >= 0
