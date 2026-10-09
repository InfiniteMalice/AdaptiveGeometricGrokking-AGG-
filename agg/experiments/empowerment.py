"""Opt-in known-MDP measurement protocol; no exploration or model training."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agg.empowerment.environments import betweenness, distances, fixture, goal_policies
from agg.empowerment.estimators import (
    discounted_occupancy,
    effective_information,
    potential_empowerment,
)

from .runner import write_json


@dataclass(frozen=True)
class EmpowermentConfig:
    fixtures: tuple[str, ...] = ("fork", "grid", "gateway", "central", "distractors")
    exhaustive_fixtures: tuple[str, ...] = ("fork",)
    gamma: float = 0.9
    all_starts: bool = True
    max_policies: int = 4096
    capacity_tolerance: float = 1e-9
    capacity_iterations: int = 20000

    def __post_init__(self) -> None:
        if (
            not self.fixtures
            or len(set(self.fixtures)) != len(self.fixtures)
            or not set(self.exhaustive_fixtures) <= set(self.fixtures)
            or len(set(self.exhaustive_fixtures)) != len(self.exhaustive_fixtures)
            or type(self.all_starts) is not bool
            or not np.isfinite(self.gamma)
            or not 0 <= self.gamma < 1
            or not np.isfinite(self.capacity_tolerance)
            or self.capacity_tolerance <= 0
            or any(
                type(v) is not int or v < 1 for v in (self.max_policies, self.capacity_iterations)
            )
        ):
            raise ValueError("invalid empowerment configuration")
        for name in self.fixtures:
            fixture(name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EmpowermentConfig":
        converted: dict[str, Any] = {
            k: tuple(v) if k in {"fixtures", "exhaustive_fixtures"} else v
            for k, v in values.items()
        }
        return cls(**converted)


def run_empowerment(config: EmpowermentConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.to_dict())
    started = time.perf_counter()
    environments: list[dict[str, Any]] = []
    solves = iterations = 0
    for name in config.fixtures:
        env = fixture(name)
        distance = distances(env)
        centrality = betweenness(env)
        rows = []
        for start in range(env.n_states) if config.all_starts else (0,):
            estimate = potential_empowerment(
                env,
                start,
                config.gamma,
                mode="exhaustive" if name in config.exhaustive_fixtures else "goals",
                max_policies=config.max_policies,
                tolerance=config.capacity_tolerance,
                max_iterations=config.capacity_iterations,
            )
            channel = np.asarray(estimate.occupancies)
            marginal = channel.mean(axis=0)
            reachable = np.isfinite(distance[start])
            outcome_support = (channel > 0).any(axis=0)
            task_reward = np.zeros(env.n_states)
            task_reward[-1] = 1
            returns = channel @ task_reward
            row = asdict(estimate)
            row.update(
                {
                    "start": start,
                    "effective_empowerment": effective_information(channel),
                    "effective_prior": [1 / len(channel)] * len(channel),
                    "reachable_states": np.flatnonzero(reachable).tolist(),
                    "reachable_outcome_coverage": float(outcome_support.sum() / reachable.sum()),
                    "distinct_occupancy_rows": len(np.unique(np.round(channel, 12), axis=0)),
                    "distinct_row_protocol": "float64 occupancies rounded to 12 decimals",
                    "independent_rollout_hitting_counts": [
                        float(1 / mass) if mass > 0 else None for mass in marginal
                    ],
                    "bottleneck_centrality": float(centrality[start]),
                    "task_reward_vector": task_reward.tolist(),
                    "per_skill_normalized_reward": returns.tolist(),
                    "best_skill_reward": float(returns.max()),
                    "uniform_skill_reward": float(returns.mean()),
                    "reward_adaptation_advantage": float(returns.max() - returns.mean()),
                }
            )
            rows.append(row)
            solves += estimate.policy_count
            iterations += estimate.capacity.iterations
        negative = None
        if name == "distractors":
            policies = goal_policies(env)
            channel = np.array([discounted_occupancy(env, p, 0, config.gamma) for p in policies])
            reward = np.zeros(env.n_states)
            reward[-1] = 1
            constant = np.ones(env.n_states)
            negative = {
                "protocol": "omit the only task-rewarding goal policy; same start and gamma",
                "restricted_policy_indices": list(range(len(channel) - 1)),
                "task_reward_vector": reward.tolist(),
                "restricted_effective_empowerment": effective_information(channel[:-1]),
                "restricted_best_task_reward": float((channel[:-1] @ reward).max()),
                "full_best_task_reward": float((channel @ reward).max()),
                "constant_reward_vector": constant.tolist(),
                "constant_reward_adaptation_advantage": float(np.ptp(channel @ constant)),
            }
            solves += len(channel)
        environments.append(
            {
                "name": name,
                "version": env.version,
                "sha256": env.digest,
                "transitions": [list(row) for row in env.transitions],
                "labels": list(env.labels),
                "shortest_action_distances": [
                    [float(v) if np.isfinite(v) else None for v in row] for row in distance
                ],
                "unavailable_distance_reason": "unreachable under any permitted action path",
                "starts": rows,
                "negative_control": negative,
            }
        )
    config_digest = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    result = {
        "schema_version": "agg.empowerment/1",
        "measurement_version": "discounted-occupancy-capacity/1",
        "units": {
            "information": "nats",
            "reward": "normalized discounted state reward",
            "shortest_distance": "environment actions",
            "rollout_hitting": "restarts",
        },
        "protocol": "known MDP; t=0 included; stationary skills; fitted capacity prior",
        "config_sha256": config_digest,
        "environment_seed": None,
        "model_seed": None,
        "seed_unavailable_reason": "deterministic enumeration without sampling or models",
        "checkpoint_hashes": None,
        "partition_hashes": None,
        "artifact_unavailable_reason": "no trained model or train/test partition in this reference",
        "sampling_uncertainty": None,
        "uncertainty_reason": "known finite channels; numerical capacity interval per start",
        "measurement_kind": "finite_model_calculation",
        "budget": {
            "occupancy_linear_solves": solves,
            "capacity_iterations": iterations,
            "executed_environment_actions": 0,
            "training_updates": 0,
        },
        "environments": environments,
        "wall_seconds": time.perf_counter() - started,
        "claims": "fixture checks only; no learned adaptation or generalization result",
    }
    # Reject non-JSON numbers before the shared writer; unavailable values stay null.
    result = json.loads(json.dumps(result, allow_nan=False))
    write_json(output / "summary.json", result)
    return result
