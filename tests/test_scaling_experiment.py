import pytest

from agg.experiments.scaling import ScalingConfig, audit_scaling, run_scaling


def test_scaling_grid_matches_budgets_and_excludes_large_models_from_fit(tmp_path, monkeypatch):
    from agg.tasks.protocol import EvaluationProtocol

    original = EvaluationProtocol.evaluation
    protected = False

    def guarded(protocol, role):
        assert protected or role == "selection"
        return original(protocol, role)

    monkeypatch.setattr(EvaluationProtocol, "evaluation", guarded)
    config = ScalingConfig(
        fit_widths=(4, 6, 8),
        extrapolation_widths=(10,),
        seeds=(3,),
        volumes=(16,),
        steps=(1,),
        dimension=2,
        samples=16,
        bootstrap_draws=20,
        measurement_repeats=3,
    )
    output = tmp_path / "scaling"
    result = run_scaling(config, output)
    assert len(result["cells"]) == 12
    assert all(
        c["budget"]["updates"] == 1 and c["budget"]["tokens"] == 128 for c in result["cells"]
    )
    assert all(c["training_flops"] is None and c["flops_missing_reason"] for c in result["cells"])
    assert all(c["training_rows_available"] == 16 for c in result["cells"])
    for width in (4, 6, 8, 10):
        counts = [c["trainable_parameters"] for c in result["cells"] if c["width"] == width]
        assert len(set(counts)) == 1
    protected = True
    report = audit_scaling(output, seed=13)
    assert len(report["cells"]) == 12
    assert report["final_test"] is None
    held_ids = {c["id"] for c in result["cells"] if c["fit_role"] == "extrapolation"}
    assert all(not set(g["fit"]["fit_ids"]) & held_ids for g in report["scaling_groups"])
    assert len(report["geometry_contrasts"]) == 8
    assert all(c["parameter_matched"] and c["budget_matched"] for c in report["geometry_contrasts"])
    for cell in report["cells"]:
        assert cell["audit_budget"]["forward_calls"] == 21
        assert cell["audit_budget"]["token_inputs"] > 0
        assert cell["measurement"]["horizon"]["oracle_valid"]
        assert set(cell["measurement"]["compression"]) == {"FP16", "INT8"}
        assert "required_update" in cell["measurement"]["causal"]["decisive"]["metrics"]
    with pytest.raises(ValueError, match="already"):
        audit_scaling(output)


def test_scaling_config_requires_real_extrapolation_and_controlled_volumes():
    with pytest.raises(ValueError):
        ScalingConfig(fit_widths=(8, 12, 16), extrapolation_widths=(12,))
    with pytest.raises(ValueError):
        ScalingConfig(volumes=(0,))


def small_config(**kwargs):
    return ScalingConfig(
        fit_widths=(4, 6, 8),
        extrapolation_widths=(10,),
        seeds=(3,),
        volumes=(16,),
        steps=(1,),
        dimension=2,
        samples=16,
        bootstrap_draws=20,
        measurement_repeats=3,
        **kwargs,
    )


def test_constructor_failure_is_recorded_and_does_not_abort_grid(tmp_path, monkeypatch):
    import agg.experiments.scaling as scaling

    original = scaling.GeometryAdapter

    def failing_adapter(width, dimension, geometry, **kwargs):
        if geometry == "hyperbolic":
            raise ImportError("optional geometry unavailable")
        return original(width, dimension, geometry, **kwargs)

    monkeypatch.setattr(scaling, "GeometryAdapter", failing_adapter)
    run = tmp_path / "construction"
    result = run_scaling(small_config(geometries=("hyperbolic", "euclidean")), run)
    assert len(result["cells"]) == 8
    assert result["cells"][0]["status"] == "failed"
    assert result["cells"][1]["status"] == "completed"
    failed = [c for c in result["cells"] if c["status"] == "failed"]
    assert len(failed) == 4
    assert all(c["parameters"] is None and c["budget"]["updates"] == 0 for c in failed)
    report = audit_scaling(run)
    failed_group = next(g for g in report["scaling_groups"] if g["geometry"] == "hyperbolic")
    assert len(failed_group["fit"]["excluded"]) == 4
    assert all(not f["available"] for f in failed_group["fit"]["forms"].values())


@pytest.mark.parametrize("error_type", [OSError, KeyError, TypeError])
def test_unexpected_audit_cell_failure_is_recorded_and_later_cells_run(
    tmp_path, monkeypatch, error_type
):
    import json

    import agg.experiments.scaling as scaling

    run = tmp_path / "cell-failure"
    grid = run_scaling(small_config(geometries=("euclidean", "product")), run)
    original = scaling.load_model
    attempted = []

    def failing_load(path):
        attempted.append(path.name)
        if path == run / grid["cells"][0]["checkpoint"]:
            raise error_type("checkpoint unavailable")
        return original(path)

    monkeypatch.setattr(scaling, "load_model", failing_load)
    report = audit_scaling(run)
    failed, *remaining = report["cells"]
    assert len(attempted) == len(grid["cells"]) == len(report["cells"])
    assert failed["measurement"] is None and failed["error"] is None
    assert failed["missing_reason"].startswith(error_type.__name__ + ":")
    assert "checkpoint unavailable" in failed["missing_reason"]
    assert failed["audit_budget"]["forward_calls"] == 0
    assert all(row["measurement"] is not None and row["error"] is not None for row in remaining)
    assert report["geometry_contrasts"][0]["paired_accuracy"] is None
    assert json.loads((run / "scaling-audit.json").read_text()) == report
    assert not (run / "scaling-audit-failure.json").exists()


def test_diagnostic_failure_preserves_primary_fit_population(tmp_path, monkeypatch):
    import agg.experiments.scaling as scaling

    run = tmp_path / "diagnostic"
    run_scaling(small_config(geometries=("euclidean",)), run)
    original = scaling.ActivationPrecisionModel

    def broken_precision(model, precision):
        if precision == "FP16":
            raise FloatingPointError("Precision overflow")
        return original(model, precision)

    monkeypatch.setattr(scaling, "ActivationPrecisionModel", broken_precision)
    report = audit_scaling(run)
    for row in report["cells"]:
        assert row["error"] is not None
        assert row["error"] == pytest.approx(1 - row["measurement"]["scores"]["ood"]["accuracy"])
        assert "Precision overflow" in row["measurement"]["compression"]["FP16"]["missing_reason"]
        assert "paired_accuracy" in row["measurement"]["compression"]["INT8"]
        assert row["measurement"]["horizon"]["oracle_valid"]
    fitted = report["scaling_groups"][0]["fit"]
    assert len(fitted["fit_ids"]) == 3 and not fitted["excluded"]
    assert fitted["forms"]["log_linear"]["available"]


@pytest.mark.parametrize("diagnostic", ["measure_resources", "_correct"])
def test_auxiliary_failure_cannot_remove_primary_models(tmp_path, monkeypatch, diagnostic):
    import agg.experiments.scaling as scaling

    def broken(*args, **kwargs):
        raise RuntimeError("auxiliary measurement unavailable")

    monkeypatch.setattr(scaling, diagnostic, broken)
    run = tmp_path / diagnostic
    result = run_scaling(small_config(geometries=("euclidean", "product")), run)
    assert all(c["status"] == "completed" for c in result["cells"])
    report = audit_scaling(run)
    assert all(c["error"] is not None for c in report["cells"])
    assert all(len(g["fit"]["fit_ids"]) == 3 for g in report["scaling_groups"])
