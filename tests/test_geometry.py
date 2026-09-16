import copy

import pytest
import torch
import torch.nn.functional as F


def test_gate_endpoints_and_exception_restore():
    from agg.gating import ScalarGate

    gate = ScalarGate(0.25, learned=True)
    base, adapted = torch.zeros(2), torch.ones(2)
    with pytest.raises(RuntimeError), gate.force(0):
        assert torch.equal(gate(base, adapted), base)
        with gate.force(1):
            assert torch.equal(gate(base, adapted), adapted)
        raise RuntimeError("restore")
    assert gate.value().item() == pytest.approx(0.25)
    assert gate.statistics()["entropy"] > 0


@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic", "product"])
def test_adapter_real_forward_backward_and_bypass(geometry):
    from agg.geometry import GeometryAdapter

    torch.manual_seed(7)
    adapter = GeometryAdapter(8, 4, geometry=geometry, learned_gate=True, learned_curvature=True)
    h = torch.randn(3, 2, 8, requires_grad=True)
    with adapter.gate.force(0):
        assert torch.equal(adapter(h), h)
    result = adapter(h)
    assert result.shape == h.shape
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert any(p.grad.abs().sum() > 0 for p in adapter.parameters())


@pytest.mark.parametrize("geometry", ["hyperbolic", "product"])
def test_curvature_changes_computation_and_matches_euclidean_budget(geometry):
    from agg.geometry import GeometryAdapter

    torch.manual_seed(13)
    a = GeometryAdapter(8, 4, geometry=geometry, gate=1, learned_curvature=True)
    b = copy.deepcopy(a)
    with torch.no_grad():
        b.manifold.isp_c.add_(2)
    h = torch.randn(5, 8)
    assert not torch.allclose(a(h), b(h), atol=1e-6)
    control = GeometryAdapter(8, 4, learned_curvature=True)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in control.parameters())


def test_distillation_ground_truth_and_teacher_detachment():
    from agg.distillation import distillation_loss

    student = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    teacher = -student.detach().clone().requires_grad_()
    teacher.retain_grad()
    labels = torch.tensor([0, 1])
    assert torch.allclose(
        distillation_loss(student, labels, teacher, teacher_weight=0),
        F.cross_entropy(student, labels),
    )
    sh = torch.randn(2, 3, requires_grad=True)
    th = torch.randn(2, 3, requires_grad=True)
    loss = distillation_loss(
        student,
        labels,
        teacher,
        student_hidden=sh,
        teacher_hidden=th,
        representation_weight=0.2,
        relation_weight=0.3,
    )
    loss.backward()
    assert student.grad is not None and sh.grad is not None
    assert teacher.grad is None and th.grad is None


def test_per_layer_and_model_wrapper():
    from agg.geometry import AdaptedModel, GeometryAdapter, LayerGeometryAdapters
    from agg.models import TinyTransformer

    adapters = LayerGeometryAdapters([GeometryAdapter(8, 4, gate=0), GeometryAdapter(8, 4, gate=1)])
    h = torch.randn(2, 3, 8)
    assert torch.equal(adapters(h, 0), h)
    assert not torch.equal(adapters(h, 1), h)
    base = TinyTransformer(10, 3, width=8, heads=2)
    wrapped = AdaptedModel(base, adapters.adapters[0])
    tokens = torch.randint(0, 10, (2, 3))
    logits, hidden = wrapped(tokens, return_hidden=True)
    assert torch.equal(logits, base(tokens))
    assert len(hidden) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimension": 0},
        {"dimension": 9},
        {"dimension": 4, "curvature": float("nan")},
        {"dimension": 4, "gate": -1},
        {"dimension": 4, "geometry": "invalid"},
    ],
)
def test_adapter_invalid_config(kwargs):
    from agg.geometry import GeometryAdapter

    with pytest.raises(ValueError):
        GeometryAdapter(8, **kwargs)


def test_forced_off_adapter_bypasses_nonfinite_adapted_path():
    from agg.geometry import GeometryAdapter

    adapter = GeometryAdapter(4, 2)
    with torch.no_grad():
        adapter.prototypes.fill_(float("nan"))
    h = torch.ones(2, 4)
    with adapter.gate.force(0):
        assert torch.equal(adapter(h), h)


def test_missing_geoopt_is_actionable_and_euclidean_still_works(monkeypatch):
    import sys

    from agg.geometry import GeometryAdapter

    monkeypatch.setitem(sys.modules, "geoopt", None)
    GeometryAdapter(4, 2)(torch.ones(1, 4))
    with pytest.raises(ImportError, match="geometry"):
        GeometryAdapter(4, 2, geometry="hyperbolic")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"teacher_weight": float("nan")},
        {"representation_weight": 1},
        {"relation_weight": -1},
    ],
)
def test_distillation_rejects_invalid_objectives(kwargs):
    from agg.distillation import distillation_loss

    with pytest.raises(ValueError):
        distillation_loss(
            torch.ones(2, 3), torch.zeros(2, dtype=torch.long), torch.ones(2, 3), **kwargs
        )
