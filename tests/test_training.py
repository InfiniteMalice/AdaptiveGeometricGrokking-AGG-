import pytest
import torch

from agg.models import TinyTransformer
from agg.tasks import hierarchy, modular_addition, retrieval
from agg.training import TrainConfig, train


def test_modular_labels_disjoint_and_seeded():
    data = modular_addition(11, seed=7)
    again = modular_addition(11, seed=7)
    assert torch.equal(data.train.x, again.train.x)
    sets = []
    for part in (data.train, data.id, data.ood):
        assert torch.equal(part.y, part.x[:, :2].sum(1) % 11)
        sets.append(set(map(tuple, part.x.tolist())))
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert len(set.union(*sets)) == 121


def test_hierarchy_ancestry_not_numeric_order():
    data = hierarchy(depth=3, branching=2, seed=4)
    pairs = {}
    for part in (data.train, data.id, data.ood):
        pairs.update(
            {tuple(x[:2]): y for x, y in zip(part.x.tolist(), part.y.tolist(), strict=True)}
        )
    assert pairs[(1, 7)] == 1
    assert pairs[(2, 7)] == 0
    assert pairs[(3, 3)] == 0  # strict ancestry


def test_retrieval_distance_independent_from_length_and_distractors():
    for length in (8, 16):
        data = retrieval(samples=30, length=length, distance=3, density=0.5, seed=6)
        x, y = data.train.x, data.train.y
        target = length - 1 - 3
        assert torch.equal((x[:, target] - 1) % 4, y)
        assert torch.equal((x[:, target] - 1) // 4, x[:, -1] - 65)
        assert ((x[:, :-1] != 0).sum(1) == 1 + round(0.5 * (length - 2))).all()
        assert data.metadata["relevant_position"] == target
    with pytest.raises(ValueError):
        retrieval(distance=8, length=8)


def test_zero_distractor_retrieval_has_no_train_id_or_ood_leakage():
    data = retrieval(samples=128, length=12, distance=3, density=0, seed=0)
    train_rows = set(map(tuple, data.train.x.tolist()))
    assert not train_rows.intersection(map(tuple, data.id.x.tolist()))
    assert not train_rows.intersection(map(tuple, data.ood.x.tolist()))
    assert data.metadata["ood_distance"] > data.metadata["distance"]
    with pytest.raises(ValueError, match="longer"):
        retrieval(length=12, distance=11)


def test_causal_hidden_states_cannot_see_future():
    torch.manual_seed(3)
    model = TinyTransformer(20, 5, width=16, layers=2, heads=2, max_length=6)
    model.eval()
    x = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 17, 18]])
    logits, hidden = model(x, return_hidden=True)
    _, other = model(changed, return_hidden=True)
    assert logits.shape == (1, 5)
    assert len(hidden) == 3
    for h, h2 in zip(hidden, other, strict=True):
        torch.testing.assert_close(h[:, :2], h2[:, :2])


def test_training_is_reproducible_and_checkpoint_has_optimizer_rng(tmp_path):
    config = TrainConfig(steps=3, eval_every=1, width=16, heads=2, layers=2, seed=4)
    first = train(modular_addition(7, seed=4), config, tmp_path / "a")
    second = train(modular_addition(7, seed=4), config, tmp_path / "b")
    assert first.history == second.history
    assert len(first.history) == 4
    for key, tensor in first.model.state_dict().items():
        torch.testing.assert_close(tensor, second.model.state_dict()[key], rtol=0, atol=0)
    checkpoint = torch.load(tmp_path / "a" / "checkpoint-3.pt", weights_only=False)
    assert {"optimizer", "rng", "model", "config", "step"} <= checkpoint.keys()
    assert (tmp_path / "a" / "metrics.jsonl").exists()
