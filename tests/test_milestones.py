import json
from dataclasses import replace

import pytest

from agg.experiments.causal import causal_report
from agg.experiments.config import ExperimentConfig, Features
from agg.experiments.independent import file_hash, report_run
from agg.experiments.milestones import milestone_report
from agg.experiments.runner import run_experiment
from agg.telemetry.milestones import Measurement, milestone_diagnostics
from agg.training import TrainConfig


def observation(step, values, *, verified=True):
    return {
        "step": step,
        "evidence": {
            key: [Measurement(value, "fixture-oracle", verified)] for key, value in values.items()
        },
    }


def retrieval_values():
    return {
        "correct_invariance": 0.9,
        "required_update": 0.9,
        "relevant_flip": 0.9,
        "irrelevant_flip": 0.05,
        "ood_accuracy": 0.9,
        "decisive_joint": 0.9,
        "train_accuracy": 0.4,
        "train_loss": 1.2,
    }


def test_nonmonotonic_milestones_confirm_and_keep_later_regressions():
    good = retrieval_values()
    points = [observation(i, good) for i in range(3)]
    points.append(observation(3, {**good, "required_update": 0.0}))
    result = milestone_diagnostics(points)
    retrieval = result["milestones"]["retrieval_reuse"]
    assert retrieval["crossing"]["stable_crossing"] == 0
    assert retrieval["crossing"]["confirmation_step"] == 2
    assert retrieval["history"][-1]["satisfied"] is False
    assert retrieval["contradictions_after_confirmation"] == [3]
    assert result["milestones"]["memorization_gap"]["crossing"]["censored"] is True
    assert result["milestones"]["structural_generalization"]["crossing"]["censored"] is False
    assert retrieval["history"][1]["previous"] == retrieval["history"][0]["measurements"]
    assert result["reward_authorized"] is False


def test_missing_conflicts_and_unverified_evidence_break_sustain():
    good = retrieval_values()
    points = [
        observation(0, good),
        observation(1, good, verified=False),
        observation(2, good),
        observation(3, {k: v for k, v in good.items() if k != "required_update"}),
        observation(4, good),
    ]
    points[-1]["evidence"]["required_update"].append(Measurement(0.1, "other-oracle", True))
    result = milestone_diagnostics(points)["milestones"]["retrieval_reuse"]
    assert result["crossing"]["stable_crossing"] is None
    assert result["crossing"]["last_step"] == 4
    assert [h["verification_state"] for h in result["history"]] == [
        "observed",
        "unverified",
        "observed",
        "missing",
        "conflicting",
    ]
    assert result["history"][-1]["conflicts"] == ["required_update"]


def test_loss_alone_and_single_compression_event_do_not_establish_milestone():
    points = [
        observation(0, {"train_loss": 0.0}),
        observation(
            1,
            {
                "parameters_reduced": 1.0,
                "baseline_ood": 0.9,
                "compression_id_gain": 0.0,
                "compression_ood_gain": 0.0,
            },
        ),
    ]
    result = milestone_diagnostics(points)
    assert all(r["crossing"]["censored"] for r in result["milestones"].values())
    assert result["milestones"]["compression_robustness"]["history"][-1]["satisfied"] is True
    with pytest.raises(ValueError):
        Measurement(float("nan"), "oracle", True)
    with pytest.raises(ValueError):
        milestone_diagnostics([observation(1, {}), observation(1, {})])


def test_missing_evidence_cannot_bound_property_onset():
    good = retrieval_values()
    points = [observation(0, good), observation(1, {}), *[observation(i, good) for i in (2, 3, 4)]]
    crossing = milestone_diagnostics(points)["milestones"]["retrieval_reuse"]["crossing"]
    assert crossing["stable_crossing"] == 2
    assert crossing["confirmation_step"] == 4
    assert crossing["interval"] == [None, 2]


def test_real_milestone_report_preserves_frozen_model_and_missing_compression(tmp_path):
    config = ExperimentConfig(
        task="retrieval",
        data_seed=79,
        independent_evaluation=True,
        samples=16,
        dimensions=(4,),
        training=TrainConfig(steps=3, eval_every=1, width=8, heads=2, layers=1),
        features=replace(Features.baseline(), telemetry=False),
    )
    run = tmp_path / "run"
    run_experiment(config, run)
    before = file_hash(run / "selected-state.pt")
    causal_report(run, seed=7)
    report_run(run, bootstrap_seed=7)
    result = milestone_report(run)
    assert result["reward_authorized"] is False
    assert len(result["future_validation"]["accuracy"]["raw_loss"]["pairs"]) == 3
    assert result["milestones"]["compression_robustness"]["crossing"]["censored"] is True
    assert file_hash(run / "selected-state.pt") == before
    assert not (run / "final-attempt.json").exists()
    assert json.loads((run / "milestone-report.json").read_text()) == result
    with pytest.raises(ValueError, match="already"):
        milestone_report(run)
    # Explicitly controlled source failures distinguish undefined conditional
    # metrics from failed execution; both reasons must survive composition.
    import shutil

    copy = tmp_path / "missing-sources"
    shutil.copytree(run, copy, ignore=shutil.ignore_patterns("milestone-*"))
    source = json.loads((copy / "causal-report.json").read_text())
    source["trajectory"][0]["measurement"] = None
    source["trajectory"][0]["missing_reason"] = "fixture checkpoint failed"
    metric = source["trajectory"][1]["measurement"]["ood"]["decisive"]["metrics"]["required_update"]
    metric.update(value=None, missing_reason="no eligible observations")
    (copy / "causal-report.json").write_text(json.dumps(source))
    missing = milestone_report(copy)["milestones"]["retrieval_reuse"]["history"]
    assert (
        missing[0]["evidence"]["required_update"][0]["missing_reason"]
        == "fixture checkpoint failed"
    )
    assert (
        missing[1]["evidence"]["required_update"][0]["missing_reason"] == "no eligible observations"
    )


def test_hierarchy_uses_full_stratum_balanced_accuracy(tmp_path):
    config = ExperimentConfig(
        task="hierarchy",
        tree_depth=4,
        data_seed=79,
        independent_evaluation=True,
        dimensions=(4,),
        training=TrainConfig(steps=1, eval_every=1, width=8, heads=2, layers=1),
        features=Features.baseline(),
    )
    run = tmp_path / "hierarchy"
    run_experiment(config, run)
    causal = causal_report(run, seed=7)
    value = causal["trajectory"][0]["measurement"]["ood"]["structural_accuracy"]["balanced_value"]
    result = milestone_report(run)
    measured = result["milestones"]["structural_generalization"]["history"][0]
    assert measured["measurements"]["ood_accuracy"] == value
    assert "full_stratum_balanced" in measured["evidence"]["ood_accuracy"][0]["source"]
