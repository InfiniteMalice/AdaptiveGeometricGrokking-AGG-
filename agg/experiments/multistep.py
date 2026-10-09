"""Executed delayed-reward fixture; exact oracle-critic results are not AGG evidence."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agg.credit import discounted_returns
from agg.credit.multistep import (
    enumerate_episodes,
    exact_reference,
    optimize,
    pdis,
    sample_episodes,
)


@dataclass(frozen=True)
class MultistepConfig:
    horizons: tuple[int, ...] = (3, 5)
    seeds: tuple[int, ...] = (3, 5, 7)
    episodes: int = 64
    rounds: int = 8
    inner_steps: int = 8
    gamma: float = 0.9
    delta: float = 0.02

    def __post_init__(self) -> None:
        if (
            not self.horizons
            or not self.seeds
            or len(set(self.horizons)) != len(self.horizons)
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(h) is not int or not 3 <= h <= 8 for h in self.horizons)
            or any(type(s) is not int or s < 0 for s in self.seeds)
            or any(
                type(v) is not int or v < 1 for v in (self.episodes, self.rounds, self.inner_steps)
            )
            or self.inner_steps > 100
            or not np.isfinite(self.gamma)
            or not 0 <= self.gamma <= 1
            or not np.isfinite(self.delta)
            or self.delta <= 0
        ):
            raise ValueError("Invalid finite multistep grid/budget")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MultistepConfig":
        converted: dict[str, Any] = {
            k: tuple(v) if k in ("horizons", "seeds") else v for k, v in values.items()
        }
        return cls(**converted)


def run_multistep(config: MultistepConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    start = time.perf_counter()
    runs = []
    budget = {
        "sampled_episodes": 0,
        "sampled_actions": 0,
        "reference_episodes": 0,
        "reference_actions": 0,
    }
    with (output / "episodes.jsonl").open("w") as log:
        for horizon in config.horizons:
            for seed in config.seeds:
                for k in (None, 1, 2, horizon):
                    policy = np.full(horizon * (horizon + 1) // 2, 0.5)
                    arm = "discounted_diagnostic" if k is None else f"k{k}"
                    row: dict[str, Any] = {
                        "horizon": horizon,
                        "seed": seed,
                        "arm": arm,
                        "rounds": [],
                        "actor_updates": 0,
                        "status": "running",
                    }
                    runs.append(row)
                    try:
                        for iteration in range(config.rounds):
                            before, _ = exact_reference(policy, horizon, config.gamma)
                            episodes = sample_episodes(
                                policy,
                                horizon=horizon,
                                count=config.episodes,
                                seed=seed * 100000 + iteration,
                            )
                            budget["sampled_episodes"] += len(episodes)
                            budget["sampled_actions"] += len(episodes) * horizon
                            for e in episodes:
                                log.write(
                                    json.dumps(
                                        {
                                            "arm": arm,
                                            "seed": seed,
                                            "horizon": horizon,
                                            "round": iteration,
                                            "kind": "sampled",
                                            "episode": asdict(e),
                                        }
                                    )
                                    + "\n"
                                )
                            step: dict[str, Any] = {
                                "round": iteration,
                                "behavior": policy.tolist(),
                                "before_return": before,
                                "sampled_episodes": len(episodes),
                                "observed_successes": sum(e.rewards[-1] for e in episodes),
                                "discounted_return_mean": float(
                                    np.mean(
                                        [
                                            discounted_returns(list(e.rewards), config.gamma)[0]
                                            for e in episodes
                                        ]
                                    )
                                ),
                            }
                            if k is not None:
                                update = optimize(
                                    policy,
                                    episodes,
                                    k=k,
                                    gamma=config.gamma,
                                    delta=config.delta,
                                    inner_steps=config.inner_steps,
                                )
                                candidate = update.pop("policy")
                                analysis_start = time.perf_counter()
                                exact_episodes, weights = enumerate_episodes(
                                    policy, horizon=horizon
                                )
                                budget["reference_episodes"] += len(exact_episodes)
                                budget["reference_actions"] += len(exact_episodes) * horizon
                                for e in exact_episodes:
                                    log.write(
                                        json.dumps(
                                            {
                                                "arm": arm,
                                                "seed": seed,
                                                "horizon": horizon,
                                                "round": iteration,
                                                "kind": "exhaustive_reference",
                                                "episode": asdict(e),
                                            }
                                        )
                                        + "\n"
                                    )
                                reference = pdis(
                                    candidate,
                                    policy,
                                    exact_episodes,
                                    k=k,
                                    gamma=config.gamma,
                                    weights=weights,
                                )
                                reference.pop("gradient")
                                after, _ = exact_reference(candidate, horizon, config.gamma)
                                step.update(
                                    {
                                        "optimizer": update,
                                        "reference": reference,
                                        "surrogate_bias": reference["value"] - (after - before),
                                        "sample_error": update["diagnostics"]["value"]
                                        - reference["value"],
                                        "analysis_wall_seconds": time.perf_counter()
                                        - analysis_start,
                                    }
                                )
                                row["actor_updates"] += len(update["accepted"])
                                policy = candidate
                            step["after_return"] = exact_reference(policy, horizon, config.gamma)[0]
                            row["rounds"].append(step)
                        row["status"] = "complete"
                    except Exception as exc:
                        row["status"] = "failed"
                        row["failure"] = f"{type(exc).__name__}: {exc}"
                        row["partial_execution_cost_unknown"] = True
                    row["final_policy"] = policy.tolist()
                    row["final_return"] = exact_reference(policy, horizon, config.gamma)[0]
    result = {
        "schema_version": 1,
        "config": asdict(config),
        "runs": runs,
        "budget": budget,
        "wall_seconds": time.perf_counter() - start,
        "interpretation": "Known finite MDP with exact behavior critic; no ID/OOD or AGG claim",
        "uncertainty": "Report seed-level values; three seeds do not establish calibrated CIs",
        "solver": "bounded gradient/backtracking, not SCP; exact return never accepts updates",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
