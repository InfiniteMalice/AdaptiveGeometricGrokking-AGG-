from dataclasses import replace
from itertools import combinations

import pytest
import torch

from agg.experiments.config import ExperimentConfig
from agg.experiments.runner import make_data
from agg.tasks.protocol import make_protocol


def test_modular_commuted_structures_never_cross_roles():
    protocol = make_protocol("modular", seed=31, modulus=17)
    roles = [protocol.development().train]
    roles.extend(protocol.evaluation(role).combined() for role in ("selection", "audit", "final"))
    groups = [{tuple(sorted(row[:2])) for row in split.x.tolist()} for split in roles]
    assert all(not a & b for a, b in combinations(groups, 2))
    for split in roles:
        assert torch.equal(split.y, split.x[:, :2].sum(1) % 17)
    assert protocol.manifest() == make_protocol("modular", seed=31, modulus=17).manifest()
    assert protocol.manifest() != make_protocol("modular", seed=32, modulus=17).manifest()
    assert set(vars(protocol.development())) == {
        "train",
        "id",
        "ood",
        "vocab_size",
        "classes",
        "metadata",
    }


def test_hierarchy_subtrees_and_labels_are_isolated():
    protocol = make_protocol("hierarchy", seed=7, depth=4)
    splits = [protocol.development().train]
    splits.extend(protocol.evaluation(role).combined() for role in ("selection", "audit", "final"))
    nodes = [set(split.x[:, :2].flatten().tolist()) for split in splits]
    assert all(not a & b for a, b in combinations(nodes, 2))
    for split in splits:
        for (a, d, _), label in zip(split.x.tolist(), split.y.tolist(), strict=True):
            ancestors = set()
            while d:
                d = (d - 1) // 2
                ancestors.add(d)
            assert label == int(a in ancestors)
        assert set(split.y.tolist()) == {0, 1}


@pytest.mark.parametrize("density", [0, 1])
def test_retrieval_all_keys_including_distractors_respect_roles(density):
    protocol = make_protocol("retrieval", seed=13, samples=40, density=density)
    splits = [protocol.development().train]
    splits.extend(protocol.evaluation(role).combined() for role in ("selection", "audit", "final"))
    key_sets = []
    for split in splits:
        memories = split.x[:, :-1]
        memory_keys = (memories[memories > 0] - 1) // 4
        query_keys = split.x[:, -1] - 65
        key_sets.append(set(memory_keys.tolist()) | set(query_keys.tolist()))
        for row, target, key in zip(split.x, split.y, query_keys, strict=True):
            found = [
                int((token - 1) % 4)
                for token in row[:-1]
                if token > 0 and int((token - 1) // 4) == int(key)
            ]
            assert found == [int(target)]
    assert all(not a & b for a, b in combinations(key_sets, 2))
    for role in ("selection", "audit", "final"):
        data = protocol.evaluation(role)
        assert len(data.id_clusters) == len(data.id.y)
        assert len(set(data.id_clusters)) >= 2


def test_too_few_structures_fail_instead_of_random_row_fallback():
    with pytest.raises(ValueError, match="structur"):
        make_protocol("modular", seed=1, modulus=5)
    with pytest.raises(ValueError, match="depth"):
        make_protocol("hierarchy", seed=1, depth=2)
    with pytest.raises(ValueError, match="depth"):
        make_protocol("hierarchy", seed=1, depth=3)


def test_data_seed_is_independent_and_legacy_defaults_preserved():
    config = ExperimentConfig(independent_evaluation=True, data_seed=12)
    a = make_data(config)
    b = make_data(replace(config, training=replace(config.training, seed=101)))
    assert torch.equal(a.train.x, b.train.x)
    legacy = make_data(ExperimentConfig())
    assert legacy.metadata.get("evaluation_protocol") is None
    with pytest.raises(ValueError, match="data_seed"):
        ExperimentConfig(independent_evaluation=True)
