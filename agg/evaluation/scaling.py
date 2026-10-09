"""Predeclared small-model fits and strictly excluded larger-model extrapolation."""

import math
from typing import Any

import numpy as np


def fit_scaling(rows: list[dict[str, Any]], *, seed: int = 0, draws: int = 200) -> dict[str, Any]:
    if type(draws) is not int or draws < 20 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("at least 20 bootstrap draws and valid seed required")
    for row in rows:
        if row["fit_role"] not in {"fit", "extrapolation"}:
            raise ValueError("explicit fit/extrapolation role required")
        if row["parameters"] is None and row["error"] is None:
            continue
        if (
            type(row["parameters"]) not in (int, float)
            or not math.isfinite(row["parameters"])
            or row["parameters"] <= 0
        ):
            raise ValueError("positive finite parameter counts required")
        if row["error"] is not None and (
            type(row["error"]) not in (int, float)
            or not math.isfinite(row["error"])
            or not 0 <= row["error"] <= 1
        ):
            raise ValueError("error rates must be finite in [0,1] or explicit missing")
    declared_fit = [r for r in rows if r["fit_role"] == "fit"]
    declared_test = [r for r in rows if r["fit_role"] == "extrapolation"]
    known_fit = [r["parameters"] for r in declared_fit if r["parameters"] is not None]
    known_test = [r["parameters"] for r in declared_test if r["parameters"] is not None]
    if (
        not declared_fit
        or not declared_test
        or (known_fit and known_test and min(known_test) <= max(known_fit))
    ):
        raise ValueError("held-out configurations must be larger than every fit configuration")
    fitted = [r for r in declared_fit if r["error"] is not None]
    test = [r for r in declared_test if r["error"] is not None]
    seeds = sorted({r["seed"] for r in fitted})
    result: dict[str, Any] = {
        "schema_version": "agg.scaling-fit/1",
        "fit_ids": [r["id"] for r in fitted],
        "extrapolation_ids": [r["id"] for r in test],
        "excluded": [
            {"id": r["id"], "missing_reason": r.get("missing_reason", "measurement unavailable")}
            for r in rows
            if r["error"] is None
        ],
        "forms": {},
        "model_seeds": seeds,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "uncertainty_method": (
            "whole-model-seed percentile bootstrap across fitted sizes; "
            "descriptive mean-fit interval"
        ),
        "uncertainty_missing_reason": "at least two independent model seeds required"
        if len(seeds) < 2
        else None,
        "uncertainty_warning": "few model seeds; interval coverage is not calibrated"
        if len(seeds) < 5
        else "descriptive, conditional on fixed data and form",
        "selection": "forms preregistered; no form selected using extrapolation outcomes",
    }
    if len({r["parameters"] for r in fitted}) < 3:
        result["forms"] = {
            name: {
                "available": False,
                "missing_reason": "at least three distinct fitted sizes required",
            }
            for name in ("power", "log_linear")
        }
        return result
    x = np.array([math.log(r["parameters"]) for r in fitted])
    design = np.column_stack((np.ones(len(x)), x))
    observed = np.array([r["error"] for r in fitted])
    test_design = np.array([[1, math.log(r["parameters"])] for r in test]).reshape(-1, 2)
    result["fit_response_range"] = [float(observed.min()), float(observed.max())]
    result["flat_response"] = bool(np.ptp(observed) < 1e-12)
    result["near_perfect_fit_observations"] = int((observed <= 0.01).sum())
    for name in ("power", "log_linear"):
        if name == "power" and (observed <= 0).any():
            result["forms"][name] = {
                "available": False,
                "missing_reason": (
                    "power-error fit requires strictly positive observed errors; "
                    "no floor substituted"
                ),
            }
            continue
        response = np.log(observed) if name == "power" else observed
        coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        if rank < 2:
            result["forms"][name] = {"available": False, "missing_reason": "rank-deficient fit"}
            continue

        def predict(matrix: Any, coef: Any, *, form=name) -> Any:
            with np.errstate(over="ignore", invalid="ignore"):
                raw = matrix @ coef
                return np.exp(raw) if form == "power" else raw

        fitted_prediction = predict(design, coefficients)
        prediction = predict(test_design, coefficients)
        boot = []
        if len(seeds) >= 2:
            rng = np.random.default_rng(seed)
            for _ in range(draws):
                sampled = rng.choice(seeds, size=len(seeds), replace=True)
                indices = [i for s in sampled for i, row in enumerate(fitted) if row["seed"] == s]
                estimate, _, boot_rank, _ = np.linalg.lstsq(
                    design[indices], response[indices], rcond=None
                )
                if boot_rank == 2 and np.isfinite(estimate).all():
                    boot.append(estimate)
        interval = np.quantile(np.array(boot), [0.025, 0.975], axis=0).T.tolist() if boot else None
        extrapolation = []
        for i, row in enumerate(test):
            value = float(prediction[i])
            valid = math.isfinite(value)
            samples = np.array([predict(test_design[i], coef) for coef in boot])
            finite_samples = samples[np.isfinite(samples)]
            extrapolation.append(
                {
                    "id": row["id"],
                    "seed": row["seed"],
                    "parameters": row["parameters"],
                    "observed_error": row["error"],
                    "prediction": value if valid else None,
                    "residual": row["error"] - value if valid else None,
                    "prediction_interval": np.quantile(finite_samples, [0.025, 0.975]).tolist()
                    if len(finite_samples)
                    else None,
                    "interval_valid_draws": len(finite_samples),
                    "outside_error_domain": not valid or not 0 <= value <= 1,
                    "missing_reason": None
                    if valid
                    else "nonfinite extrapolation; no clipping applied",
                }
            )
        residuals = [
            r["error"] - float(value) for r, value in zip(fitted, fitted_prediction, strict=True)
        ]
        held_residuals = [r["residual"] for r in extrapolation if r["residual"] is not None]
        result["forms"][name] = {
            "available": True,
            "formula": "error=exp(a)*parameters**b"
            if name == "power"
            else "error=a+b*log(parameters)",
            "coefficients": coefficients.tolist(),
            "coefficient_interval": interval,
            "bootstrap_valid_draws": len(boot),
            "fit_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            "fit_residuals": [
                {"id": r["id"], "residual": e} for r, e in zip(fitted, residuals, strict=True)
            ],
            "extrapolation": extrapolation,
            "extrapolation_rmse": float(np.sqrt(np.mean(np.square(held_residuals))))
            if held_residuals
            else None,
            "extrapolation_missing_reason": None
            if held_residuals
            else "no finite held-out residuals",
        }
    return result
