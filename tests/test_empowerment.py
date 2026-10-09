import math

import numpy as np
import pytest

from agg.empowerment.environments import (
    FiniteEnvironment,
    betweenness,
    distances,
    fixture,
    goal_policies,
)
from agg.empowerment.estimators import (
    channel_capacity,
    discounted_occupancy,
    effective_information,
    potential_empowerment,
)


def test_fork_occupancy_information_and_exhaustive_capacity():
    env = fixture("fork")
    policies = goal_policies(env)
    channel = np.array([discounted_occupancy(env, p, 0, 0.9) for p in policies])
    assert np.allclose(channel, [[0.1, 0.9, 0], [0.1, 0.9, 0], [0.1, 0, 0.9]])
    assert effective_information(channel[1:]) == pytest.approx(0.9 * math.log(2))
    result = potential_empowerment(env, 0, 0.9, mode="exhaustive")
    assert result.policy_count == 8
    assert result.estimator_type == "exhaustive_policy_capacity"
    assert result.capacity.converged
    assert result.capacity.lower_nats == pytest.approx(0.9 * math.log(2))


def test_occupancy_includes_time_zero_and_resolves_infinite_cycles():
    env = FiniteEnvironment("cycle", ((1,), (0,)))
    policy = np.ones((2, 1))
    assert np.allclose(discounted_occupancy(env, policy, 0, 0.5), [2 / 3, 1 / 3])
    assert np.allclose(discounted_occupancy(env, policy, 0, 0), [1, 0])
    assert effective_information([[1, 0], [1, 0]]) == 0


def test_nonuniform_prior_is_optimized_not_confused_with_uniform_mi():
    channel = np.array([[1, 0], [0.5, 0.5]])
    result = channel_capacity(channel, tolerance=1e-10)
    assert result.converged
    assert result.lower_nats == pytest.approx(math.log(1.25), abs=1e-9)
    assert result.prior[1] == pytest.approx(0.4, abs=1e-5)
    assert result.lower_nats > effective_information(channel)
    duplicated = channel_capacity(np.vstack([channel, channel[0]]))
    assert duplicated.lower_nats == pytest.approx(result.lower_nats, abs=1e-8)
    unfinished = channel_capacity(channel, tolerance=1e-12, max_iterations=1)
    assert not unfinished.converged
    assert unfinished.lower_nats <= math.log(1.25) <= unfinished.upper_nats


def test_zero_probabilities_and_redundant_skills_are_finite():
    assert effective_information([[1, 0], [0, 1]], [1, 0]) == 0
    assert effective_information([[1, 0], [0, 1]]) == pytest.approx(math.log(2))
    assert channel_capacity([[1, 0], [1, 0]]).lower_nats == 0
    assert channel_capacity([[1]]).converged


@pytest.mark.parametrize(
    "channel", [[], [[0, 0]], [[-1, 2]], [[float("nan"), 1]], [[0.2, 0.2]], [[1, 0], [1]]]
)
def test_invalid_channels_rejected(channel):
    with pytest.raises(ValueError):
        effective_information(channel)


@pytest.mark.parametrize("prior", [[0, 0], [1], [-1, 2], [float("inf"), 0]])
def test_invalid_priors_rejected(prior):
    with pytest.raises(ValueError):
        effective_information([[1, 0], [0, 1]], prior)


def test_environment_is_immutable_and_rejects_invalid_actions():
    source = [[1, 0], [1, 1]]
    env = FiniteEnvironment("test", source)
    digest = env.digest
    source[0][0] = 0
    assert env.step(0, 0) == 1
    assert env.digest == digest
    for state, action in [(-1, 0), (2, 0), (0, -1), (0, 2), (0, 0.1), (True, 0)]:
        with pytest.raises(ValueError):
            env.step(state, action)
    for table in [[], [[2]], [[0.5]], [[True]], [[0], [1, 1]]]:
        with pytest.raises(ValueError):
            FiniteEnvironment("bad", table)


def test_unreachable_distances_and_bottleneck_graph_semantics():
    env = FiniteEnvironment("disconnected", ((0,), (1,)))
    assert np.isinf(distances(env)[0, 1])
    assert np.allclose(betweenness(env), [0, 0])
    gateway = fixture("gateway")
    scores = betweenness(gateway)
    assert scores[gateway.labels.index("1,2")] == pytest.approx(72 / 132)
    assert scores[gateway.labels.index("1,2")] < max(scores)
    assert np.isfinite(distances(gateway)).all()


def test_policy_validation_gamma_and_enumeration_cap():
    env = fixture("fork")
    for gamma in [-1, 1, float("nan")]:
        with pytest.raises(ValueError):
            discounted_occupancy(env, np.ones((3, 2)) / 2, 0, gamma)
    with pytest.raises(ValueError):
        discounted_occupancy(env, np.ones((3, 2)), 0, 0.9)
    with pytest.raises(ValueError, match="cap"):
        potential_empowerment(fixture("grid"), 0, 0.9, mode="exhaustive", max_policies=4)
    result = potential_empowerment(fixture("grid"), 0, 0.9, mode="goals")
    assert result.estimator_type == "restricted_goal_policy_capacity"
    assert result.upper_bound_scope == "restricted_channel_only"


def test_reward_anisotropy_does_not_change_information():
    env = fixture("distractors")
    policies = goal_policies(env)
    channel = np.array([discounted_occupancy(env, p, 0, 0.9) for p in policies])
    assert effective_information(channel) > 1
    # All controllable leaves have the same task reward: no adaptation advantage.
    irrelevant = np.array([0.0] + [1.0] * (env.n_states - 1))
    assert np.ptp(channel @ irrelevant) == pytest.approx(0)
    useful = np.zeros(env.n_states)
    useful[-1] = 1
    assert np.ptp(channel @ useful) > 0.8
