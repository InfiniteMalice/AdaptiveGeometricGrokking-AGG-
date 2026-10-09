"""Finite, oracle-critic k-step PDIS experiment, independent of supervised training."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from agg.supervision import ExecutionEvidence, ToyEnvironment


def _policy(policy: np.ndarray, horizon: int) -> np.ndarray:
    p = np.asarray(policy, dtype=float)
    if (
        type(horizon) is not int
        or not 1 <= horizon <= 8
        or p.shape != (horizon * (horizon + 1) // 2,)
        or not np.all(np.isfinite(p))
        or np.any((p <= 0) | (p >= 1))
    ):
        raise ValueError("Finite policy with full support and integer horizon 1..8 required")
    return p


def _hash(policy: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(policy, dtype="<f8").tobytes()).hexdigest()


def _index(t: int, count: int) -> int:
    return t * (t + 1) // 2 + count


@dataclass(frozen=True)
class Episode:
    actions: tuple[int, ...]
    states: tuple[int, ...]
    rewards: tuple[float, ...]
    behavior_hash: str
    trace: tuple[ExecutionEvidence, ...]


def _execute(actions: tuple[int, ...], behavior: np.ndarray) -> Episode:
    env = ToyEnvironment()
    states = []
    for t, action in enumerate(actions):
        states.append(_index(t, env.state))
        env.execute(
            "add", action, intent="delayed terminal outcome", expected_state=env.state + action
        )
    rewards = (0.0,) * (len(actions) - 1) + (float(env.state == len(actions)),)
    return Episode(actions, tuple(states), rewards, _hash(behavior), env.trace)


def sample_episodes(behavior: np.ndarray, *, horizon: int, count: int, seed: int) -> list[Episode]:
    behavior = _policy(behavior, horizon)
    if type(count) is not int or count < 1:
        raise ValueError("Positive episode count required")
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(count):
        actions: list[int] = []
        for t in range(horizon):
            actions.append(int(rng.random() < behavior[_index(t, sum(actions))]))
        episodes.append(_execute(tuple(actions), behavior))
    return episodes


def enumerate_episodes(behavior: np.ndarray, *, horizon: int) -> tuple[list[Episode], np.ndarray]:
    behavior = _policy(behavior, horizon)
    episodes = [_execute(a, behavior) for a in itertools.product((0, 1), repeat=horizon)]
    weights = np.array(
        [
            np.prod(
                [
                    behavior[s] if a else 1 - behavior[s]
                    for s, a in zip(e.states, e.actions, strict=True)
                ]
            )
            for e in episodes
        ]
    )
    return episodes, weights


def exact_reference(policy: np.ndarray, horizon: int, gamma: float) -> tuple[float, np.ndarray]:
    policy = _policy(policy, horizon)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite in [0,1]")
    values = np.zeros((horizon + 1, horizon + 1))
    advantages = np.zeros((len(policy), 2))
    for t in reversed(range(horizon)):
        for count in range(t + 1):
            s = _index(t, count)
            q = np.array(
                [
                    float(count + a == horizon)
                    if t == horizon - 1
                    else gamma * values[t + 1, count + a]
                    for a in (0, 1)
                ]
            )
            values[t, count] = (1 - policy[s]) * q[0] + policy[s] * q[1]
            advantages[s] = q - values[t, count]
    return float(values[0, 0]), advantages


def _validate(e: Episode, behavior: np.ndarray, horizon: int) -> None:
    if e.behavior_hash != _hash(behavior):
        raise ValueError("Stale or mixed behavior policy")
    if not all(len(x) == horizon for x in (e.actions, e.states, e.rewards, e.trace)):
        raise ValueError("Incomplete verified episode")
    count = 0
    for t, (a, s, r, record) in enumerate(
        zip(e.actions, e.states, e.rewards, e.trace, strict=True)
    ):
        after = count + a
        if (
            type(a) is not int
            or a not in (0, 1)
            or s != _index(t, count)
            or r != (float(after == horizon) if t == horizon - 1 else 0.0)
            or record.order != t
            or record.action != "add"
            or record.argument != a
            or record.pre_state != count
            or record.post_state != after
            or record.result != after
            or record.expected_state != after
            or record.error is not None
            or not record.verified
            or record.verifier_confidence != 1.0
            or record.verifier_provenance != "toy-integer-register/exact-state/v1"
            or record.pre_hash != hashlib.sha256(str(count).encode("ascii")).hexdigest()
            or record.post_hash != hashlib.sha256(str(after).encode("ascii")).hexdigest()
        ):
            raise ValueError("Episode disagrees with verified executed outcome")
        count = after


def pdis(
    target: np.ndarray,
    behavior: np.ndarray,
    episodes: list[Episode],
    *,
    k: int,
    gamma: float,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Episodes required")
    horizon = len(episodes[0].actions)
    target, behavior = _policy(target, horizon), _policy(behavior, horizon)
    if type(k) is not int or not 1 <= k <= horizon:
        raise ValueError("k must lie in 1..horizon")
    _, advantages = exact_reference(behavior, horizon, gamma)
    n = len(episodes)
    mass = np.full(n, 1 / n) if weights is None else np.asarray(weights, dtype=float)
    if (
        mass.shape != (n,)
        or not np.all(np.isfinite(mass))
        or np.any(mass < 0)
        or not np.isclose(mass.sum(), 1.0, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Normalized nonnegative finite reference weights required")
    outcomes = np.zeros(n)
    gradients = np.zeros((n, len(target)))
    ratios = np.zeros((n, horizon))
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for i, e in enumerate(episodes):
            _validate(e, behavior, horizon)
            for t in range(horizon):
                start = max(0, t - k + 1)
                window = list(zip(e.states[start : t + 1], e.actions[start : t + 1], strict=True))
                ratio = float(
                    np.prod(
                        [
                            target[s] / behavior[s] if a else (1 - target[s]) / (1 - behavior[s])
                            for s, a in window
                        ]
                    )
                )
                ratios[i, t] = ratio
                term = gamma**t * ratio * advantages[e.states[t], e.actions[t]]
                outcomes[i] += term
                for s, a in window:
                    gradients[i, s] += term * (a - target[s])
        value = float(mass @ outcomes)
        gradient = mass @ gradients
        variance = (
            float(mass @ ((outcomes - value) ** 2))
            if weights is not None
            else float(np.var(outcomes, ddof=1))
            if n > 1
            else None
        )
    if not (np.isfinite(value) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(ratios))):
        raise ValueError("Nonfinite importance objective")
    scaled = ratios[:, -1] / max(float(ratios[:, -1].max()), np.finfo(float).tiny)
    weighted = mass * scaled
    ess = float(weighted.sum() ** 2 / np.sum(weighted**2)) if np.any(weighted) else None
    return {
        "value": value,
        "gradient": gradient,
        "variance": variance,
        "variance_kind": "exact_behavior_population" if weights is not None else "sample_ddof1",
        "trajectory_values": outcomes.tolist(),
        "episodes": n,
        "importance": {
            "maximum": float(ratios.max()),
            "minimum": float(ratios.min()),
            "terminal_ess": ess,
            "ess_unit": "weighted_episode_records",
        },
    }


def optimize(
    behavior: np.ndarray,
    episodes: list[Episode],
    *,
    k: int,
    gamma: float,
    delta: float = 0.02,
    inner_steps: int = 8,
) -> dict[str, Any]:
    """Sample-surrogate ascent with bounded backtracking, not an SCP solver.

    The trust region stays relative to the batch's original behavior throughout.
    No target-policy exact return is consulted to accept or reject an update.
    """
    import time

    start = time.perf_counter()
    if (
        not np.isfinite(delta)
        or delta <= 0
        or type(inner_steps) is not int
        or not 1 <= inner_steps <= 100
    ):
        raise ValueError("Positive finite KL radius and 1..100 inner steps required")
    policy = np.asarray(behavior, dtype=float).copy()
    if np.any((policy < 1e-4) | (policy > 1 - 1e-4)):
        raise ValueError("Optimizer requires behavior probability floor 1e-4")
    result = pdis(policy, behavior, episodes, k=k, gamma=gamma)
    evaluations = 1
    accepted = []
    rejected = []
    for step in range(inner_steps):
        gradient = result["gradient"]
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-12:
            break
        direction = gradient / norm
        logits = np.log(policy / (1 - policy))
        for backtrack in range(20):
            candidate = 1 / (1 + np.exp(-(logits + 2.0 ** (-backtrack) * direction)))
            if np.any((candidate < 1e-4) | (candidate > 1 - 1e-4)):
                rejected.append({"step": step, "backtrack": backtrack, "reason": "support_floor"})
                continue
            kl = float(
                np.max(
                    candidate * np.log(candidate / behavior)
                    + (1 - candidate) * np.log((1 - candidate) / (1 - behavior))
                )
            )
            if kl > delta:
                rejected.append(
                    {"step": step, "backtrack": backtrack, "reason": "maximum_kl", "maximum_kl": kl}
                )
                continue
            proposal = pdis(candidate, behavior, episodes, k=k, gamma=gamma)
            evaluations += 1
            if proposal["value"] <= result["value"] + 1e-12:
                rejected.append({"step": step, "backtrack": backtrack, "reason": "no_improvement"})
                continue
            accepted.append(
                {
                    "before": result["value"],
                    "after": proposal["value"],
                    "maximum_kl": kl,
                    "backtrack": backtrack,
                }
            )
            policy, result = candidate, proposal
            break
        else:
            break
    return {
        "policy": policy,
        "accepted": accepted,
        "rejected": rejected,
        "objective_evaluations": evaluations,
        "wall_seconds": time.perf_counter() - start,
        "diagnostics": {key: value for key, value in result.items() if key != "gradient"},
    }
