"""Outcome-labeled interventions; task semantics are decoded independently."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .synthetic import Split


def oracle(x: Tensor, metadata: dict[str, Any]) -> Tensor:
    if x.ndim != 2 or x.dtype != torch.long or len(x) == 0:
        raise ValueError("oracle requires nonempty integer token rows")
    task = metadata["task"]
    if task == "modular":
        p = metadata["modulus"]
        if x.shape[1] != 3 or not bool(((x[:, :2] >= 0) & (x[:, :2] < p)).all()):
            raise ValueError("invalid modular operands")
        if not bool((x[:, -1] == p).all()):
            raise ValueError("invalid modular query token")
        return x[:, :2].sum(1) % p
    if task == "hierarchy":
        branching, depth = metadata["branching"], metadata["depth"]
        nodes = (branching ** (depth + 1) - 1) // (branching - 1)
        if (
            x.shape[1] != 3
            or not bool(((x[:, :2] >= 0) & (x[:, :2] < nodes)).all())
            or not bool((x[:, -1] == nodes).all())
        ):
            raise ValueError("invalid fixed-tree node query")
        result = []
        for ancestor, descendant, _ in x.tolist():
            lineage = []
            while descendant > 0:
                descendant = (descendant - 1) // branching
                lineage.append(descendant)
            result.append(int(ancestor in lineage))
        return torch.tensor(result, dtype=torch.long)
    if task != "retrieval":
        raise ValueError("unsupported task oracle")
    keys, values = metadata["keys"], metadata["values"]
    query = x[:, -1] - (1 + keys * values)
    memories = x[:, :-1]
    if (
        x.shape[1] < 2
        or not bool(((query >= 0) & (query < keys)).all())
        or not bool(((memories >= 0) & (memories <= keys * values)).all())
    ):
        raise ValueError("invalid retrieval tokens")
    relevant = (memories > 0) & ((memories - 1) // values == query[:, None])
    if not bool((relevant.sum(1) == 1).all()):
        raise ValueError("retrieval oracle requires a unique relevant memory")
    return ((memories[relevant] - 1) % values).long()


@dataclass
class InterventionPairs:
    original: Split
    transformed: Split
    clusters: tuple[str, ...]
    source_indices: list[int]
    missing: list[dict[str, Any]]
    attempted: int
    kind: str
    transformation: str


def intervention_pairs(
    split: Split, clusters: tuple[str, ...], metadata: dict[str, Any], *, kind: str, seed: int
) -> InterventionPairs:
    if kind not in {"invariant", "decisive"} or len(clusters) != len(split.y):
        raise ValueError("aligned clusters and invariant/decisive kind required")
    original_labels = oracle(split.x, metadata)
    if not torch.equal(original_labels, split.y):
        raise ValueError("source labels disagree with independent oracle")
    generator = torch.Generator().manual_seed(seed)
    task = metadata["task"]
    changed, indices, missing = [], [], []
    support = {tuple(row): i for i, row in enumerate(split.x.tolist())}
    for index, row in enumerate(split.x):
        candidate = row.clone()
        if task == "retrieval":
            values = metadata["values"]
            query = int(row[-1]) - (1 + metadata["keys"] * values)
            relevant = (row[:-1] > 0) & ((row[:-1] - 1) // values == query)
            if kind == "decisive":
                pos = int(torch.where(relevant)[0][0])
                candidate[pos] = 1 + query * values + (int(split.y[index]) + 1) % values
            else:
                positions = torch.where(~relevant)[0]
                # A random cyclic shift is a permutation; no-op rows stay missing.
                if len(positions) > 1:
                    shift = int(torch.randint(1, len(positions), (), generator=generator))
                    candidate[positions] = row[positions].roll(shift)
        elif task == "modular" and kind == "invariant":
            candidate[:2] = row[:2].flip(0)
            if tuple(candidate.tolist()) not in support:
                candidate = row.clone()
        else:
            # Change exactly one operative input, within this private stratum.
            different = (split.x[:, :2] != row[:2]).sum(1) == 1
            label_match = split.y == split.y[index]
            eligible = torch.where(
                different & (label_match if kind == "invariant" else ~label_match)
            )[0]
            if len(eligible):
                choice = int(torch.randint(len(eligible), (), generator=generator))
                candidate = split.x[int(eligible[choice])].clone()
        if torch.equal(candidate, row):
            missing.append({"source_index": index, "reason": "no nontrivial supported transform"})
            continue
        changed.append(candidate)
        indices.append(index)
    original = Split(split.x[indices].clone(), split.y[indices].clone())
    transformed_x = torch.stack(changed) if changed else split.x[:0].clone()
    transformed_y = oracle(transformed_x, metadata) if changed else split.y[:0].clone()
    valid = original.y == transformed_y
    if len(valid) and not bool((valid if kind == "invariant" else ~valid).all()):
        raise ValueError("transformation failed independent semantic validation")
    names = {
        ("retrieval", "invariant"): "irrelevant_position_permutation",
        ("retrieval", "decisive"): "relevant_value_replacement",
        ("modular", "invariant"): "operand_swap",
    }
    # Shared endpoints create dependence between source clusters. Bootstrap
    # connected components, never pretend reused transformed rows are independent.
    parents = {cluster: cluster for cluster in clusters}

    def root(cluster: str) -> str:
        while parents[cluster] != cluster:
            cluster = parents[cluster]
        return cluster

    if task != "retrieval":
        for index, row in zip(indices, changed, strict=True):
            target = support[tuple(row.tolist())]
            first, second = root(clusters[index]), root(clusters[target])
            parents[max(first, second)] = min(first, second)
    return InterventionPairs(
        original,
        Split(transformed_x, transformed_y),
        tuple(root(clusters[i]) for i in indices),
        indices,
        missing,
        len(split.y),
        kind,
        names.get((task, kind), "one_input_replacement_within_private_stratum"),
    )
