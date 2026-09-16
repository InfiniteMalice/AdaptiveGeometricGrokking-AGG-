import copy

import pytest
import torch

from agg.experiments.diagnostics import observe
from agg.geometry import AdaptedModel, GeometryAdapter
from agg.models import TinyTransformer
from agg.tasks import retrieval


@pytest.mark.parametrize(
    "attribution,memorization,expected_calls",
    [
        (True, True, 6),
        (False, True, 3),
        (True, False, 5),
        (False, False, 2),
    ],
)
def test_observation_flags_remove_real_model_calls(attribution, memorization, expected_calls):
    data = retrieval(samples=10, length=4, distance=1, keys=4, values=2)
    model = TinyTransformer(data.vocab_size, data.classes, width=4, heads=1, layers=1)
    calls = []
    hook = model.register_forward_hook(lambda module, args, output: calls.append(args[0].clone()))
    try:
        raw, hidden = observe(
            model, data, attribution=attribution, memorization=memorization, gates=False
        )
    finally:
        hook.remove()
    assert len(calls) == expected_calls
    assert ("memorization" in raw) == memorization
    assert ("attribution" in raw) == attribution
    assert ("retrieval" in raw) == attribution
    assert "gates" not in raw
    assert len(hidden) == 2
    assert model.training


def test_gate_telemetry_can_be_removed_without_changing_model():
    data = retrieval(samples=10, length=4, distance=1, keys=4, values=2)
    base = TinyTransformer(data.vocab_size, data.classes, width=4, heads=1, layers=1)
    model = AdaptedModel(base, GeometryAdapter(4, 2, gate=0.25))
    before = copy.deepcopy(model.state_dict())
    on, hidden_on = observe(model, data, attribution=False, memorization=False)
    off, hidden_off = observe(model, data, attribution=False, memorization=False, gates=False)
    assert on["gates"]["adapter.gate"]["value"] == 0.25
    assert "gates" not in off
    assert all(torch.equal(a, b) for a, b in zip(hidden_on, hidden_off, strict=True))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_representation_diagnostics_exact_reference_and_known_distortion():
    from agg.experiments.diagnostics import representation_diagnostics

    h = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    hidden = [h, h + 1, h + 2]
    raw = representation_diagnostics(hidden, reference_hidden=hidden)
    assert raw["reconstruction"]["final_hidden_mse"] == 0
    assert raw["reconstruction"]["pairwise_distance_relative_error"] == 0
    assert raw["spectra"][-1]["participation_ratio"] == pytest.approx(1)
    assert raw["topology"][-1]["sample_count"] == 3
    changed = representation_diagnostics([x * 2 for x in hidden], reference_hidden=hidden)
    assert changed["reconstruction"]["pairwise_distance_relative_error"] == pytest.approx(1)
    assert changed["reconstruction"]["final_hidden_mse"] > 0
    assert changed["trajectory"]["path_length"] == pytest.approx(
        2 * raw["trajectory"]["path_length"]
    )


def test_representation_reference_alignment_is_checked():
    from agg.experiments.diagnostics import representation_diagnostics

    with pytest.raises(ValueError, match="aligned"):
        representation_diagnostics([torch.zeros(2, 3)], reference_hidden=[torch.zeros(3, 3)])
