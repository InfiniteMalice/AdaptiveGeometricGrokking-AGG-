"""Opt-in structural partitions; protected roles never become trainer fields.

Role access is a trusted-host API, not a sandbox. Final authorization belongs to
the frozen-run reporting API; candidate code receives only development TaskData.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .synthetic import Split, TaskData, hierarchy, modular_addition, retrieval

ROLES = ("training", "selection", "audit", "final")


def _join(parts: list[Split]) -> Split:
    return Split(torch.cat([p.x for p in parts]), torch.cat([p.y for p in parts]))


def _fingerprint(split: Split) -> str:
    digest = hashlib.sha256()
    for tensor in (split.x, split.y):
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class RoleData:
    id: Split
    ood: Split
    id_clusters: tuple[str, ...]
    ood_clusters: tuple[str, ...]

    def combined(self) -> Split:
        return _join([self.id, self.ood])


class EvaluationProtocol:
    def __init__(
        self, roles: dict[str, RoleData], vocab_size: int, classes: int, metadata: dict[str, Any]
    ):
        self._roles = roles
        self.vocab_size, self.classes, self.metadata = vocab_size, classes, metadata

    def evaluation(self, role: str) -> RoleData:
        if role not in ROLES[1:]:
            raise ValueError("evaluation role must be selection, audit or final")
        # Give callers private tensors: evaluation cannot rewrite another role.
        import copy

        return copy.deepcopy(self._roles[role])

    def development(self) -> TaskData:
        import copy

        selection = self.evaluation("selection")
        return TaskData(
            copy.deepcopy(self._roles["training"].combined()), selection.id, selection.ood,
            self.vocab_size, self.classes, copy.deepcopy(self.metadata),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agg.partitions/1",
            "metadata": dict(self.metadata),
            "roles": {
                role: {
                    stratum: {
                        "count": len(getattr(data, stratum).y),
                        "sha256": _fingerprint(getattr(data, stratum)),
                        "clusters": list(getattr(data, stratum + "_clusters")),
                    }
                    for stratum in ("id", "ood")
                }
                for role, data in self._roles.items()
            },
        }


def _assign(groups: list[str], seed: int, roles: tuple[str, ...]) -> dict[str, str]:
    groups = sorted(set(groups))
    if len(groups) < len(roles):
        raise ValueError("too few independent structures for nonempty protected roles")
    order = torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed)).tolist()
    return {groups[index]: roles[i % len(roles)] for i, index in enumerate(order)}


def _grouped(
    data: TaskData, groups: Sequence[str | None], ood: torch.Tensor, seed: int, *, modular: bool
) -> EvaluationProtocol:
    source = _join([data.train, data.id, data.ood])
    usable = [i for i, g in enumerate(groups) if g is not None]
    if modular:
        assignments = {}
        for is_ood, roles in ((False, ROLES), (True, ROLES[1:])):
            keys = [str(groups[i]) for i in usable if bool(ood[i]) == is_ood]
            assignments.update(_assign(keys, seed + int(is_ood), roles))
    else:
        assignments = _assign([str(groups[i]) for i in usable], seed, ROLES)
    result = {}
    for role in ROLES:
        parts = []
        clusters = []
        for is_ood in (False, True):
            indices = [i for i in usable if assignments[str(groups[i])] == role
                       and bool(ood[i]) == is_ood]
            if not indices and not (modular and role == "training" and is_ood):
                raise ValueError("too few structures in an ID/OOD stratum")
            parts.append(Split(source.x[indices], source.y[indices]))
            clusters.append(tuple(str(groups[i]) for i in indices))
        result[role] = RoleData(parts[0], parts[1], clusters[0], clusters[1])
    return EvaluationProtocol(result, data.vocab_size, data.classes, {
        **data.metadata, "evaluation_protocol": "structural-four-role/1", "data_seed": seed,
        "group_rule": "unordered operand pair" if modular else "disjoint depth-two subtrees",
    })


def make_protocol(
    task: str, *, seed: int, modulus: int = 17, depth: int = 3, samples: int = 128,
    length: int = 12, distance: int = 3, density: float = 0.5, hard: bool = False,
) -> EvaluationProtocol:
    if type(seed) is not int or not 0 <= seed < 2**32 - 4:
        raise ValueError("data seed must be an integer in [0, 2**32-4)")
    if task == "modular":
        data = modular_addition(modulus, seed=seed)
        source = _join([data.train, data.id, data.ood])
        groups = [f"pair:{min(a, b)}:{max(a, b)}" for a, b, _ in source.x.tolist()]
        ood = (source.x[:, :2] >= int(0.8 * modulus)).all(1)
        return _grouped(data, groups, ood, seed, modular=True)
    if task == "hierarchy":
        if depth < 3:
            raise ValueError("protected hierarchy requires depth >= 3")
        data = hierarchy(depth=depth, seed=seed)
        source = _join([data.train, data.id, data.ood])

        def root(node: int) -> int | None:
            # Removing shared ancestors prevents train/audit lineage leakage.
            if node < 3:
                return None
            while node > 6:
                node = (node - 1) // 2
            return node

        hierarchy_groups = [
            f"subtree:{root(a)}" if root(a) is not None and root(a) == root(b) else None
            for a, b, _ in source.x.tolist()
        ]
        # Withhold deepest descendants inside each evaluation role. The training
        # role still uses all rows of its own disjoint subtree.
        ood = source.x[:, 1] >= 2**depth - 1
        return _grouped(data, hierarchy_groups, ood, seed, modular=False)
    if task != "retrieval":
        raise ValueError("unknown protocol task")
    # Reuse the generator on disjoint key vocabularies, then remap tokens. This
    # prevents protected bindings leaking through irrelevant distractors too.
    key_order = torch.randperm(16, generator=torch.Generator().manual_seed(seed)).tolist()
    role_data = {}
    metadata = {}
    for index, role in enumerate(ROLES):
        keys = key_order[4 * index:4 * index + 4]
        data = retrieval(samples=samples, length=length, distance=distance, density=density,
                         hard=hard, seed=seed + index, keys=4)
        id_split = _join([data.train, data.id])
        ood_split = data.ood
        clusters = []
        for split in (id_split, ood_split):
            local_query = split.x[:, -1] - 17
            clusters.append(tuple(f"key:{keys[k]}" for k in local_query.tolist()))
            memories = split.x[:, :-1]
            original = memories.clone()
            for local, global_key in enumerate(keys):
                mask = (original > 0) & ((original - 1) // 4 == local)
                memories[mask] = 1 + global_key * 4 + (original[mask] - 1) % 4
                split.x[local_query == local, -1] = 65 + global_key
        role_data[role] = RoleData(id_split, ood_split, *clusters)
        metadata = data.metadata
    return EvaluationProtocol(role_data, 81, 4, {
        **metadata, "keys": 16, "data_seed": seed,
        "evaluation_protocol": "structural-four-role/1",
        "group_rule": "disjoint query and distractor key vocabularies",
        "id_rule": "unseen role keys at configured distance; mixed values",
        "ood_rule": "same role keys at longer distance; generator reserved values",
    })
