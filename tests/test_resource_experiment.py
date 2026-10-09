import json

import pytest

from agg.controller.resources import ResourceProfile
from agg.experiments.resources import ResourceConfig, audit_resources, run_resources


def test_resource_matrix_has_four_measured_arms_and_frozen_audit(tmp_path, monkeypatch):
    from agg.tasks.protocol import EvaluationProtocol

    original = EvaluationProtocol.evaluation
    audit_allowed = False

    def guarded(protocol, role):
        assert audit_allowed or role == "selection"
        return original(protocol, role)

    monkeypatch.setattr(EvaluationProtocol, "evaluation", guarded)
    config = ResourceConfig(
        tasks=("retrieval", "modular"),
        seeds=(5,),
        initial_steps=1,
        intervention_steps=1,
        samples=16,
        repeats=3,
        profile=ResourceProfile(max_model_bytes=1, max_training_updates=1),
    )
    run = tmp_path / "resources"
    result = run_resources(config, run)
    assert len(result["cells"]) == 2
    for cell in result["cells"]:
        assert set(cell["arms"]) == {"fixed", "existing", "joint", "experience"}
        assert all(
            c["selection_resources"]["metrics"]["resources.latency_ms"] > 0
            for c in cell["candidates"]
            if c["checkpoint"]
        )
        for name, arm in cell["arms"].items():
            assert arm["retained_resources"]["metrics"]["resources.latency_ms"] > 0
            assert arm["budget"]["executed_updates"] == 1
            if name in ("joint", "experience"):
                assert arm["status"] != "committed"
                assert not arm["retained_feasibility"]["joint_feasible"]
    assert len(result["experience"]) == 2
    assert all(r["role"] == "selection" for r in result["experience"])
    audit_allowed = True
    report = audit_resources(run, seed=3)
    assert len(report["cells"]) == 2 and report["final_test"] is None
    assert all(len(c["candidates"]) == 4 for c in report["cells"])
    with pytest.raises(ValueError, match="already"):
        audit_resources(run)
    frozen = json.loads((run / "selection-frozen.json").read_text())
    assert "resource-results.json" in frozen["artifacts"]


def test_experience_only_proposes_from_verified_selection_conditions():
    from agg.experiments.resources import choose_experience

    base = {
        "task": "retrieval",
        "role": "selection",
        "status": "committed",
        "joint_feasible": True,
        "objective": 1.2,
        "fraction": 0.1,
        "source_cell": 0,
        "hardware": {"device": "cpu"},
    }
    assert choose_experience([base], "retrieval") == base
    for change in (
        {"role": "audit"},
        {"task": "modular"},
        {"status": "rolled_back"},
        {"joint_feasible": False},
        {"objective": None},
    ):
        assert choose_experience([{**base, **change}], "retrieval") is None


def test_experiment_does_not_replace_interrupted_work_with_zero(tmp_path, monkeypatch):
    import agg.controller.training as training_module

    original = training_module.train

    def interrupt_after_update(data, config, output, **kwargs):
        def interrupt(step, model, metrics):
            if step == 1:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write('{"step":')
                raise OSError("interrupted metrics write")

        return original(data, config, output, callback=interrupt, **kwargs)

    monkeypatch.setattr(training_module, "train", interrupt_after_update)
    result = run_resources(
        ResourceConfig(
            tasks=("retrieval",),
            seeds=(5,),
            initial_steps=1,
            intervention_steps=2,
            repeats=3,
            samples=16,
            profile=ResourceProfile(max_training_updates=2),
        ),
        tmp_path / "partial",
    )
    for name in ("existing", "joint", "experience"):
        arm = result["cells"][0]["arms"][name]
        assert arm["status"] == "rolled_back"
        assert arm["budget"]["executed_updates"] is None
        assert arm["budget"]["completed_updates_lower_bound"] == 1
        assert not arm["retained_feasibility"]["joint_feasible"]
