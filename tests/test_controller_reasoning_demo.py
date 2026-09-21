import json


def test_synthetic_state_vs_dynamics_reproducible_and_audited(tmp_path):
    from agg.controller.reasoning_demo import run_demo

    first = run_demo(tmp_path / "one")
    assert first == run_demo(tmp_path / "two")
    assert first["synthetic"] is True
    assert set(first["variants"]) == {"absolute", "first_order", "second_order", "independent"}
    for name, result in first["variants"].items():
        assert 0 <= result["routing_precision"] <= 1
        assert 0 <= result["routing_stability"] <= 1
        assert result["compute_usage_proxy"] > 0
        assert result["protected_regressions"] is None
        assert len(result["recovery_steps"]) == 2
        events = [
            json.loads(line)
            for line in (tmp_path / "one" / f"{name}.jsonl").read_text().splitlines()
        ]
        assert len(events) == 60
        assert events[0]["payload"]["observation"]["provenance"]["synthetic"]
    # Check actual control perturbations reach routing; no requirement that C wins.
    forecasts = [
        first["variants"][v]["trace"][10]["forecast_entropy"]
        for v in ("absolute", "first_order", "second_order")
    ]
    assert len(set(forecasts)) == 3
