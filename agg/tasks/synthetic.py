from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class Split:
    x: Tensor
    y: Tensor


@dataclass
class TaskData:
    train: Split
    id: Split
    ood: Split
    vocab_size: int
    classes: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _partition(x: Tensor, y: Tensor, ood: Tensor, seed: int, fraction: float) -> list[Split]:
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    pool = torch.where(~ood)[0]
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))]
    count = max(1, min(len(pool) - 1, round(len(pool) * fraction)))
    indices = [pool[:count], pool[count:], torch.where(ood)[0]]
    if any(len(i) == 0 for i in indices):
        raise ValueError("task must have nonempty train, ID and OOD partitions")
    return [Split(x[i], y[i]) for i in indices]


def modular_addition(p: int = 31, *, seed: int = 0, train_fraction: float = 0.4) -> TaskData:
    if p < 5:
        raise ValueError("modulus must be at least 5")
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    x = torch.cat([pairs, torch.full((p * p, 1), p)], dim=1)
    y = pairs.sum(1) % p
    # A held-out corner separates structural extrapolation from random ID holdout.
    ood = (pairs[:, 0] >= int(0.8 * p)) & (pairs[:, 1] >= int(0.8 * p))
    parts = _partition(x, y, ood, seed, train_fraction)
    return TaskData(
        parts[0],
        parts[1],
        parts[2],
        p + 1,
        p,
        {"task": "modular", "modulus": p, "ood_rule": "both operands >= floor(0.8*p)"},
    )


def hierarchy(*, depth: int = 3, branching: int = 2, seed: int = 0) -> TaskData:
    if depth < 2 or branching < 2:
        raise ValueError("depth and branching must be at least 2")
    nodes = (branching ** (depth + 1) - 1) // (branching - 1)
    if nodes > 1000:
        raise ValueError("exhaustive hierarchy limited to 1000 nodes")
    pairs = torch.cartesian_prod(torch.arange(nodes), torch.arange(nodes))
    labels = []
    for ancestor, descendant in pairs.tolist():
        parent = (descendant - 1) // branching
        found = False
        while parent >= 0:
            found |= parent == ancestor
            parent = (parent - 1) // branching
        labels.append(int(found))
    x = torch.cat([pairs, torch.full((len(pairs), 1), nodes)], dim=1)
    y = torch.tensor(labels)
    # Withhold selected deepest leaves, preserving other leaves during training.
    first_leaf = (branching**depth - 1) // (branching - 1)
    ood = (pairs[:, 1] >= first_leaf) & (pairs[:, 1] % 2 == 0)
    parts = _partition(x, y, ood, seed, 0.7)
    return TaskData(
        parts[0],
        parts[1],
        parts[2],
        nodes + 1,
        2,
        {
            "task": "hierarchy",
            "depth": depth,
            "branching": branching,
            "ood_rule": "even deepest descendant nodes",
            "class_imbalance": "natural exhaustive ancestry prevalence; report balanced accuracy",
        },
    )


def retrieval(
    *,
    samples: int = 128,
    length: int = 12,
    distance: int = 3,
    density: float = 0.5,
    hard: bool = False,
    seed: int = 0,
    keys: int = 16,
    values: int = 4,
) -> TaskData:
    """Memory token=1+key*values+value; query token=1+keys*values+key.

    Position distance is query index minus relevant-memory index. Distractor
    density counts occupied non-relevant memory slots; pads control total length.
    ID reserves one value per key; train and ID bindings cannot overlap even with
    zero distractors. OOD uses those held-out bindings at a strictly longer distance.
    """
    if samples < 10 or length < 3 or not 1 <= distance < length:
        raise ValueError("need samples>=10, length>=3 and 1<=distance<length")
    if distance >= length - 1:
        raise ValueError("distance must leave room for a strictly longer OOD condition")
    if not 0 <= density <= 1 or keys < 4 or values < 2:
        raise ValueError("density must be in [0,1], keys>=4 and values>=2")
    generator = torch.Generator().manual_seed(seed)

    def generate(count: int, gap: int, held_out: bool) -> Split:
        x = torch.zeros((count, length), dtype=torch.long)
        query = torch.randint(keys, (count,), generator=generator)
        reserved = (query + seed) % values
        if held_out:
            y = reserved
        else:
            draw = torch.randint(values - 1, (count,), generator=generator)
            y = draw + (draw >= reserved).long()
        pos = length - 1 - gap
        x[:, pos] = 1 + query * values + y
        x[:, -1] = 1 + keys * values + query
        slots = [i for i in range(length - 1) if i != pos]
        distractors = round(density * len(slots))
        for row in range(count):
            selected = torch.randperm(len(slots), generator=generator)[:distractors]
            candidates = [k for k in range(keys) if k != int(query[row])]
            if hard:
                # Nearby key IDs share coarse groups, a declared synthetic similarity.
                candidates = sorted(candidates, key=lambda k: abs(k - int(query[row])))[:3]
            for index in selected.tolist():
                k = candidates[int(torch.randint(len(candidates), (), generator=generator))]
                v = int(torch.randint(values, (), generator=generator))
                x[row, slots[index]] = 1 + k * values + v
        return Split(x, y)

    ood_distance = min(length - 1, max(distance + 1, distance * 2))
    return TaskData(
        generate(samples, distance, False),
        generate(max(32, samples // 2), distance, True),
        generate(max(32, samples // 2), ood_distance, True),
        1 + keys * values + keys,
        values,
        {
            "task": "retrieval",
            "length": length,
            "distance": distance,
            "density": density,
            "hard": hard,
            "keys": keys,
            "values": values,
            "relevant_position": length - 1 - distance,
            "ood_distance": ood_distance,
            "id_rule": "reserved value=(key+seed)%values; disjoint query-value bindings",
            "ood_rule": "held-out bindings at strictly longer distance",
        },
    )
