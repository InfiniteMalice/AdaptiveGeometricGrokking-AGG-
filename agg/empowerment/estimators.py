"""Discounted-occupancy channel reference (Ji et al., 2026, §§3–4, app. B–E).

All information is in nats. 'Exact' occupancy means a known finite linear solve,
not sampling and not exact arithmetic. Capacity bounds have floating-point error.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .environments import FiniteEnvironment, goal_policies, index


def probabilities(value: Any, ndim: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not array.size or 0 in array.shape:
        raise ValueError("probabilities have invalid shape")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(array.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return array


def discounted_occupancy(
    env: FiniteEnvironment, policy: Any, start: int, gamma: float
) -> NDArray[np.float64]:
    start = index(start, env.n_states)
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite in [0, 1)")
    policy_array = probabilities(policy, 2)
    if policy_array.shape != (env.n_states, env.n_actions):
        raise ValueError("policy must have state-action shape")
    transition = np.zeros((env.n_states, env.n_states))
    for state, row in enumerate(env.transitions):
        for action, target in enumerate(row):
            transition[state, target] += policy_array[state, action]
    rhs = np.zeros(env.n_states)
    rhs[start] = 1 - gamma
    matrix = np.eye(env.n_states) - gamma * transition.T
    occupancy = np.linalg.solve(matrix, rhs)
    if (occupancy < -1e-12).any() or not np.allclose(matrix @ occupancy, rhs, atol=1e-12):
        raise ArithmeticError("occupancy solve failed residual/nonnegativity checks")
    occupancy = np.maximum(occupancy, 0)  # Only floating-point negative roundoff.
    probabilities(occupancy, 1)  # Do not silently normalize an ill-conditioned solve.
    return occupancy


def _divergences(channel: NDArray[np.float64], prior: NDArray[np.float64]) -> NDArray[np.float64]:
    marginal = prior @ channel
    logs = np.zeros_like(channel)
    present = channel > 0
    np.log(channel, out=logs, where=present)
    log_marginal = np.full_like(marginal, -np.inf)
    np.log(marginal, out=log_marginal, where=marginal > 0)
    terms = np.zeros_like(channel)
    np.subtract(logs, log_marginal, out=terms, where=present)
    np.multiply(channel, terms, out=terms, where=present)
    return terms.sum(axis=1)


def effective_information(channel: Any, prior: Any = None) -> float:
    array = probabilities(channel, 2)
    source = np.full(len(array), 1 / len(array)) if prior is None else probabilities(prior, 1)
    if source.shape != (len(array),):
        raise ValueError("prior must have one probability per skill")
    active = source > 0
    value = float(source[active] @ _divergences(array[active], source[active]))
    return max(0.0, value)  # MI can be negative only by roundoff for a valid channel.


@dataclass(frozen=True)
class Capacity:
    lower_nats: float
    upper_nats: float
    gap_nats: float
    prior: tuple[float, ...]
    iterations: int
    converged: bool


def channel_capacity(
    channel: Any, *, tolerance: float = 1e-9, max_iterations: int = 20000
) -> Capacity:
    """Blahut-Arimoto; mutual information <= capacity <= maximum row divergence."""
    array = probabilities(channel, 2)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("capacity tolerance must be positive and finite")
    index(max_iterations - 1, max_iterations)  # Positive integer, excluding bool below.
    if isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be a positive integer")
    log_prior = np.full(len(array), -np.log(len(array)))
    for iteration in range(1, max_iterations + 1):
        prior = np.exp(log_prior)
        prior /= prior.sum()
        divergence = _divergences(array, prior)
        if not np.isfinite(divergence).all():
            raise ArithmeticError("capacity solver lost channel support")
        lower = max(0.0, float(prior @ divergence))
        upper = max(lower, float(divergence.max()))
        gap = upper - lower
        if gap <= tolerance or iteration == max_iterations:
            return Capacity(lower, upper, gap, tuple(prior.tolist()), iteration, gap <= tolerance)
        log_prior += divergence
        log_prior -= np.logaddexp.reduce(log_prior)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Potential:
    capacity: Capacity
    estimator_type: str
    upper_bound_scope: str
    policy_count: int
    total_deterministic_policies: int
    occupancies: tuple[tuple[float, ...], ...]


def potential_empowerment(
    env: FiniteEnvironment,
    start: int,
    gamma: float,
    *,
    mode: str = "goals",
    max_policies: int = 4096,
    tolerance: float = 1e-9,
    max_iterations: int = 20000,
) -> Potential:
    total = env.n_actions**env.n_states
    index(max_policies - 1, max_policies)
    if isinstance(max_policies, bool):
        raise ValueError("max_policies must be a positive integer")
    if mode == "exhaustive":
        if total > max_policies:
            raise ValueError(f"{total} policies exceed enumeration cap {max_policies}")
        actions = product(range(env.n_actions), repeat=env.n_states)
        identity = np.eye(env.n_actions)
        policies: Iterator[NDArray[np.float64]] = (identity[list(policy)] for policy in actions)
        estimator = "exhaustive_policy_capacity"
        scope = "full_finite_mdp"
    elif mode == "goals":
        policies = iter(goal_policies(env))
        estimator = "restricted_goal_policy_capacity"
        scope = "restricted_channel_only"
    else:
        raise ValueError("mode must be exhaustive or goals")
    occupancy = np.array([discounted_occupancy(env, p, start, gamma) for p in policies])
    capacity = channel_capacity(occupancy, tolerance=tolerance, max_iterations=max_iterations)
    return Potential(
        capacity,
        estimator,
        scope,
        len(occupancy),
        total,
        tuple(tuple(row) for row in occupancy.tolist()),
    )
