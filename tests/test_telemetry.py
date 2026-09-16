from pathlib import Path

import pytest
import torch

from agg.attribution import intervention_attribution, linear_cka
from agg.complexity import PrequentialCodeLengthProxy
from agg.depth import depth_derivatives, trajectory_statistics
from agg.dimension import spectral_statistics
from agg.horizon import effective_horizons
from agg.probes import ProbeEvidence, retrieval_crossing, summarize_crossings
from agg.telemetry import CATEGORIES, TelemetryCollector, TelemetryHistory
from agg.topology import connectivity, geometry_statistics


def test_spectra_degenerate_and_equal_axes():
    zero = spectral_statistics(torch.zeros(4, 3))
    assert zero["effective_rank"] == 0
    assert zero["participation_ratio"] == 0
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = spectral_statistics(x)
    assert result["effective_rank"] == pytest.approx(2)
    assert result["participation_ratio"] == pytest.approx(2)
    assert result["explained_variance"] == pytest.approx([0.5, 1.0])


def test_quadratic_derivatives_on_nonuniform_depth():
    t = torch.tensor([0.0, 0.1, 0.4, 0.7, 1.0], dtype=torch.float64)
    first, second = depth_derivatives(t * t, t)
    assert torch.allclose(first, 2 * t, atol=1e-10)
    assert torch.allclose(second, torch.full_like(t, 2), atol=1e-10)
    with pytest.raises(ValueError):
        depth_derivatives(torch.ones(2))
    path = trajectory_statistics(torch.stack([t, t], dim=1))
    assert path["path_length"] == pytest.approx(2**0.5)


def test_graph_and_geometry():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    assert connectivity(x, radius=1.1)["components"] == 2
    assert geometry_statistics(x)["pairwise_distance_mean"] > 1


def test_probe_evidence_independent_and_crossing_censoring():
    evidence = ProbeEvidence.from_predictions(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )
    assert evidence.memorization == 1
    assert evidence.retrieval == 1
    assert evidence.generalization == 0.5
    crossing = retrieval_crossing(
        [0, 10, 20, 30, 40], [0.5] * 5, [0.7, 0.4, 0.8, 0.9, 0.9], sustain=3
    )
    assert crossing.first_crossing == 0
    assert crossing.stable_crossing == 20
    assert crossing.interval == (10, 20)
    absent = retrieval_crossing([0, 10], [0.5] * 2, [0.1] * 2)
    summary = summarize_crossings([crossing, absent])
    assert summary["censored_seeds"] == 1
    assert summary["observed_median_step"] == 20


def test_horizons_and_interventions():
    result = effective_horizons([1, 2, 4], [0.9, 0.8, 0.4], nominal=8, threshold=0.75)
    assert result["retrieval_horizon"] == 2
    assert result["causal_horizon"] is None
    assert intervention_attribution(0.9, 0.4) == pytest.approx(0.5)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert linear_cka(x, 3 * x) == pytest.approx(1)
    assert linear_cka(torch.ones(3, 2), torch.ones(3, 2)) == 0


def test_proxy_code_length():
    result = PrequentialCodeLengthProxy().estimate([0.5, 0.25])
    assert result["bits"] == pytest.approx(3)
    assert result["proxy"] is True
    assert "proxy" in result["method"]
    with pytest.raises(ValueError):
        PrequentialCodeLengthProxy().estimate([0.0])


def test_schema_history_and_collector(tmp_path: Path):
    snapshots = TelemetryCollector().collect(
        3, "checkpoint.pt", {"task": {"loss": 0.4}}, hidden=[torch.randn(5, 4)] * 3
    )
    snapshot = snapshots[0]
    assert set(snapshot.availability) == set(CATEGORIES)
    assert snapshot.availability["task"]
    assert not snapshot.availability["topology"]
    assert snapshot.raw["topology"] is None
    assert snapshot.proxy["dimension"]
    history = TelemetryHistory(tmp_path / "history.jsonl")
    for item in snapshots:
        history.append(item)
    loaded = history.read()
    assert loaded[0].to_dict() == snapshot.to_dict()
    with pytest.raises(ValueError):
        TelemetryCollector().collect(0, "x", {"task": {"loss": float("nan")}})


def test_invalid_probe_values_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        ProbeEvidence.from_predictions(
            torch.tensor([float("nan")]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
        )


def test_nonfinite_crossing_breaks_sustained_run():
    crossing = retrieval_crossing([0, 1, 2, 3], [0.0] * 4, [1.0, float("nan"), 1.0, 1.0], sustain=3)
    assert crossing.first_crossing == 0
    assert crossing.censored
    assert crossing.stable_crossing is None


def test_missing_topology_dependency_is_actionable():
    import importlib.util

    from agg.topology import persistence_summary

    if importlib.util.find_spec("ripser") is None:
        with pytest.raises(ImportError, match="pip install"):
            persistence_summary(torch.eye(3))
    else:
        result = persistence_summary(torch.eye(3))
        assert len(result["dimensions"]) == 2


def test_horizon_distinguishes_nominal_causal_and_retrieval():
    result = effective_horizons(
        [1, 2, 4], [0.9, 0.8, 0.4], nominal=8, attribution=[0.9, 0.3, 0.2], causal=[0.9, 0.9, 0.9]
    )
    assert result["nominal_horizon"] == 8
    assert result["retrieval_horizon"] == 2
    assert result["attribution_horizon"] == 1
    assert result["causal_horizon"] == 4


def test_collector_derivatives_and_task_only_ablation():
    collector = TelemetryCollector()
    snapshots = collector.collect(0, "x", {}, [torch.eye(3)] * 3)
    assert snapshots[1].derived["depth_first"]["effective_rank"] == 0
    assert snapshots[1].derived["depth_second"]["effective_rank"] == 0
    task_only = collector.collect(0, "x", {"task": {"loss": 1.0}})
    assert len(task_only) == 1
    assert not task_only[0].availability["dimension"]
    assert not task_only[0].availability["depth_first"]
