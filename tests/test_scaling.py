import math

import pytest

from agg.evaluation.scaling import fit_scaling


def rows():
    return [
        {
            "id": f"{n}-{seed}",
            "parameters": n,
            "seed": seed,
            "fit_role": "fit" if n < 160 else "extrapolation",
            "error": 2 / math.sqrt(n),
        }
        for n in (10, 20, 40, 80, 160)
        for seed in (3, 5, 7)
    ]


def test_known_power_law_and_larger_results_never_enter_fit():
    samples = rows()
    result = fit_scaling(samples, seed=9, draws=40)
    power = result["forms"]["power"]
    assert power["coefficients"] == pytest.approx([math.log(2), -0.5])
    assert power["extrapolation_rmse"] < 1e-12
    assert len(power["coefficient_interval"]) == 2
    for row in samples:
        if row["fit_role"] == "extrapolation":
            row["error"] = 0.99
    changed = fit_scaling(samples, seed=9, draws=40)["forms"]["power"]
    assert changed["coefficients"] == power["coefficients"]
    assert changed["coefficient_interval"] == power["coefficient_interval"]
    assert changed["extrapolation_rmse"] > 0.8
    assert changed["extrapolation"][0]["prediction"] == pytest.approx(2 / math.sqrt(160))


def test_zero_error_and_missing_runs_are_explicit():
    samples = rows()
    samples[0]["error"] = 0
    samples[1]["error"] = None
    samples[1]["missing_reason"] = "training failed"
    result = fit_scaling(samples, draws=20)
    assert result["forms"]["power"]["available"] is False
    assert "positive" in result["forms"]["power"]["missing_reason"]
    assert result["excluded"][0]["missing_reason"] == "training failed"
    assert result["forms"]["log_linear"]["available"] is True


def test_sparse_seed_and_nonextrapolating_configuration():
    result = fit_scaling([r for r in rows() if r["seed"] == 3], draws=20)
    assert result["forms"]["power"]["coefficient_interval"] is None
    assert result["uncertainty_missing_reason"]
    samples = rows()
    samples[-1]["parameters"] = 5
    with pytest.raises(ValueError, match="larger"):
        fit_scaling(samples)


def test_invalid_fit_values_and_insufficient_sizes():
    samples = rows()
    samples[0]["error"] = -0.1
    with pytest.raises(ValueError):
        fit_scaling(samples)
    result = fit_scaling([r for r in rows() if r["parameters"] in (10, 160)], draws=20)
    assert all(not form["available"] for form in result["forms"].values())
