import pytest
import torch

from agg.experiments.candidates import ActivationPrecisionModel
from agg.experiments.checkpoints import load_model, save_model
from agg.geometry import AdaptedModel, GeometryAdapter
from agg.models import TinyTransformer


@pytest.mark.parametrize("kind", ["baseline", "euclidean", "hyperbolic", "product", "activation"])
def test_weights_only_round_trip_exact_outputs_and_architecture(tmp_path, kind):
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = TinyTransformer(17, 5, width=12, layers=3, heads=3, max_length=9)
    if kind != "baseline":
        geometry = "product" if kind == "activation" else kind
        model = AdaptedModel(
            model,
            GeometryAdapter(
                12,
                6,
                geometry=geometry,
                gate=0.3,
                learned_gate=True,
                curvature=0.7,
                learned_curvature=True,
            ),
        )
    if kind == "activation":
        model = ActivationPrecisionModel(model, "INT4")
    model.eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    before = model(tokens, return_hidden=True)
    path = tmp_path / "model.pt"
    save_model(model, path)
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 1
    loaded = load_model(path)
    after = loaded(tokens, return_hidden=True)
    assert type(loaded) is type(model)
    assert not loaded.training
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    for left, right in zip(before[1], after[1], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["version", "type", "state", "nonfinite"])
def test_checkpoint_rejects_incompatible_or_corrupt_payload(tmp_path, corruption):
    path = tmp_path / "model.pt"
    save_model(TinyTransformer(8, 3, width=8, heads=2), path)
    payload = torch.load(path, weights_only=True)
    if corruption == "version":
        payload["format_version"] = 99
    elif corruption == "type":
        payload["architecture"]["type"] = "arbitrary.module.Class"
    elif corruption == "state":
        payload["state_dict"].pop("embedding.weight")
    else:
        payload["state_dict"]["embedding.weight"].fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_dtype_modes_and_rng_preserved(tmp_path):
    model = TinyTransformer(8, 3, width=8, heads=2).double()
    model.blocks[0].eval()
    path = tmp_path / "model.pt"
    save_model(model, path)
    rng = torch.get_rng_state().clone()
    loaded = load_model(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.embedding.weight.dtype == torch.float64
    assert loaded.training and not loaded.blocks[0].training


def test_unknown_model_rejected_before_writing(tmp_path):
    path = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="Unsupported"):
        save_model(torch.nn.Linear(2, 2), path)
    assert not path.exists()
