"""Known deterministic MDPs. Graph distances count actions, not rollout restarts."""

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def index(value: int, size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
        raise ValueError(f"index must be an integer in [0, {size})")
    return int(value)


@dataclass(frozen=True)
class FiniteEnvironment:
    name: str
    transitions: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...] = ()
    version: str = "agg.finite-environment/1"

    def __post_init__(self) -> None:
        table = tuple(tuple(row) for row in self.transitions)
        if not table or not table[0] or any(len(row) != len(table[0]) for row in table):
            raise ValueError("transition table must be a nonempty rectangular state-action array")
        table = tuple(tuple(index(v, len(table)) for v in row) for row in table)
        labels = tuple(self.labels) or tuple(str(i) for i in range(len(table)))
        if len(labels) != len(table) or len(set(labels)) != len(labels):
            raise ValueError("state labels must be unique and match the state count")
        object.__setattr__(self, "transitions", table)
        object.__setattr__(self, "labels", labels)

    @property
    def n_states(self) -> int:
        return len(self.transitions)

    @property
    def n_actions(self) -> int:
        return len(self.transitions[0])

    @property
    def digest(self) -> str:
        payload = [self.version, self.name, self.transitions, self.labels]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def step(self, state: int, action: int) -> int:
        return self.transitions[index(state, self.n_states)][index(action, self.n_actions)]


def grid(name: str, cells: tuple[tuple[int, int], ...]) -> FiniteEnvironment:
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("grid cells must be nonempty and unique")
    lookup = {cell: i for i, cell in enumerate(cells)}
    # Stay, up, right, down, left; blocked actions are self loops.
    moves = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
    table = tuple(
        tuple(lookup.get((r + dr, c + dc), i) for dr, dc in moves) for i, (r, c) in enumerate(cells)
    )
    return FiniteEnvironment(name, table, tuple(f"{r},{c}" for r, c in cells))


def fixture(name: str) -> FiniteEnvironment:
    if name == "fork":
        return FiniteEnvironment(name, ((1, 2), (1, 1), (2, 2)))
    if name == "distractors":
        return FiniteEnvironment(name, (tuple(range(1, 7)),) + tuple((s,) * 6 for s in range(1, 7)))
    if name == "grid":
        return grid(name, tuple((r, c) for r in range(2) for c in range(3)))
    if name == "gateway":
        return grid(name, tuple((r, c) for r in range(3) for c in range(5) if c != 2 or r == 1))
    if name == "central":
        return grid(name, tuple((r, c) for r in range(5) for c in range(5) if r == 2 or c == 2))
    raise ValueError(f"unknown finite environment {name!r}")


def distances(env: FiniteEnvironment) -> NDArray[np.float64]:
    result = np.full((env.n_states, env.n_states), np.inf)
    for start in range(env.n_states):
        result[start, start] = 0
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for target in set(env.transitions[state]):
                if not np.isfinite(result[start, target]):
                    result[start, target] = result[start, state] + 1
                    queue.append(target)
    return result


def goal_policies(env: FiniteEnvironment) -> NDArray[np.float64]:
    """One greedy shortest-path stationary policy per goal; ties use action order.

    At an unreachable goal or a goal without a stay action, the best available
    action is used. These are restricted candidates, not an optimal skill basis.
    """
    distance = distances(env)
    policies = np.zeros((env.n_states, env.n_states, env.n_actions))
    for goal in range(env.n_states):
        for state in range(env.n_states):
            action = int(np.argmin(distance[list(env.transitions[state]), goal]))
            policies[goal, state, action] = 1
    return policies


def betweenness(env: FiniteEnvironment) -> NDArray[np.float64]:
    """Normalized directed shortest-path betweenness; duplicate actions count once."""
    n = env.n_states
    centrality = np.zeros(n)
    for start in range(n):
        parents: list[list[int]] = [[] for _ in range(n)]
        counts = np.zeros(n)
        counts[start] = 1
        depth = [-1] * n
        depth[start] = 0
        queue = deque([start])
        order = []
        while queue:
            state = queue.popleft()
            order.append(state)
            for target in sorted(set(env.transitions[state])):
                if depth[target] < 0:
                    depth[target] = depth[state] + 1
                    queue.append(target)
                if depth[target] == depth[state] + 1:
                    counts[target] += counts[state]
                    parents[target].append(state)
        dependency = np.zeros(n)
        for target in reversed(order):
            for parent in parents[target]:
                dependency[parent] += counts[parent] / counts[target] * (1 + dependency[target])
            if target != start:
                centrality[target] += dependency[target]
    return centrality / ((n - 1) * (n - 2)) if n > 2 else centrality
