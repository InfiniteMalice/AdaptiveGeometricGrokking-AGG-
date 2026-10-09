import json
from dataclasses import replace

import pytest

from agg.evaluation.independent import paired_cluster_comparison
from agg.experiments.config import ExperimentConfig, Features
from agg.experiments.independent import file_hash, report_run
from agg.experiments.runner import run_experiment
from agg.training import TrainConfig


def test_cluster_bootstrap_preserves_pairing_and_cluster_dependence():
    result = paired_cluster_comparison(
        [0, 0, 0, 0], [1, 1, 1, 0], ["a", "a", "a", "b"], seed=5, draws=500
    )
    assert result["effect"] == 0.75
    assert result["interval"] == [0, 1]
    assert result["clusters"] == 2
    same = paired_cluster_comparison([1, 0], [1, 0], ["a", "b"], seed=3)
    assert same["effect"] == 0
    assert same["interval"] == [0, 0]
    one = paired_cluster_comparison([0, 0], [1, 1], ["a", "a"], seed=5)
    assert one["effect"] == 1
    assert one["interval"] is None


@pytest.mark.parametrize(
    "before,after,clusters", [([], [], []), ([0], [1, 0], ["a"]), ([float("nan")], [1], ["a"])]
)
def test_invalid_paired_evidence_fails_closed(before, after, clusters):
    with pytest.raises(ValueError):
        paired_cluster_comparison(before, after, clusters, seed=0)


def config():
    return ExperimentConfig(
        independent_evaluation=True,
        data_seed=91,
        training=TrainConfig(steps=2, eval_every=1, width=8, heads=2, layers=1),
        dimensions=(4,),
        geometries=("euclidean",),
        candidate_steps=1,
        features=replace(Features.baseline(), consolidation=True, distillation=True),
    )


def test_real_training_freezes_before_audit_and_final_is_explicit(tmp_path):
    output = tmp_path / "run"
    summary = run_experiment(config(), output)
    assert summary["interventions"] == 1
    selected_hash = file_hash(output / "selected-state.pt")
    manifest_hash = file_hash(output / "selection-frozen.json")
    records = [
        json.loads(line) for line in (output / "candidate-attempts.jsonl").read_text().splitlines()
    ]
    assert records[-1]["optimizer_state"] == "candidate-0001-optimizer.pt"
    assert (output / records[-1]["optimizer_state"]).is_file()
    with pytest.raises(ValueError, match="authorization"):
        report_run(output, role="final")
    assert not (output / "final-report.json").exists()
    audit = report_run(output, role="audit")
    assert audit["role"] == "audit"
    assert len(audit["candidates"]) == 1
    assert audit["selected"]["id"]["selection_gap"] is not None
    assert audit["final_test"] is None
    assert file_hash(output / "selected-state.pt") == selected_hash
    final = report_run(
        output, role="final", authorization="unit-fixture-only", manifest_sha256=manifest_hash
    )
    assert final["authorization"] == "unit-fixture-only"
    assert final["manifest_sha256"] == manifest_hash
    with pytest.raises(ValueError, match="already"):
        report_run(output, role="audit")


def test_changed_artifact_is_rejected_before_independent_evaluation(tmp_path):
    output = tmp_path / "run"
    run_experiment(config(), output)
    (output / "experiment.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        report_run(output, role="audit")
    assert not (output / "audit-report.json").exists()


def test_balanced_comparison_missing_required_class_is_unavailable():
    result = paired_cluster_comparison(
        [0, 0], [1, 1], ["a", "b"], seed=1, labels=[0, 0], expected_classes=[0, 1]
    )
    assert result["effect"] is None
    assert result["interval"] is None
    assert result["unavailable_reason"] is not None


def test_finite_overflow_candidate_is_missing_without_aborting_audit(tmp_path):
    import torch

    from agg.consolidation import Proposal, trial
    from agg.evaluation.model import make_evaluator
    from agg.experiments.checkpoints import save_model
    from agg.experiments.independent import CandidateRecorder, freeze_run
    from agg.experiments.runner import evaluation_protocol
    from agg.ledger import Ledger
    from agg.models import TinyTransformer

    cfg = config()
    output = tmp_path / "run"
    output.mkdir()
    from dataclasses import asdict

    (output / "experiment.json").write_text(json.dumps(asdict(cfg)))
    protocol = evaluation_protocol(cfg)
    data = protocol.development()
    model = TinyTransformer(data.vocab_size, data.classes, width=8, heads=2, layers=1)
    save_model(model, output / "baseline-state.pt")
    save_model(model, output / "selected-state.pt")

    def overflow(candidate):
        with torch.no_grad():
            candidate.readout.weight.zero_()
            candidate.readout.bias.fill_(-3e38)
            candidate.readout.bias[0] = 3e38
        return candidate

    result = trial(
        model,
        Proposal("overflow", "readout", {}, overflow),
        make_evaluator(model, data),
        cfg.constraints,
        Ledger(output / "ledger.jsonl"),
        run_id="test",
        step=0,
        recorder=CandidateRecorder(output, parent_checkpoint=None),
    )
    assert not result.accepted
    freeze_run(output, protocol.manifest())
    report = report_run(output)
    assert report["candidates"][0]["measurement"] is None
    assert "FloatingPointError" in report["candidates"][0]["missing_reason"]
    assert report["selected"]["id"]["score"] is not None


def test_baseline_load_failure_has_durable_reason(tmp_path, monkeypatch):
    import agg.experiments.independent as independent

    output = tmp_path / "run"
    run_experiment(config(), output)

    def missing_dependency(path):
        raise RuntimeError("required model dependency unavailable")

    monkeypatch.setattr(independent, "load_model", missing_dependency)
    with pytest.raises(RuntimeError, match="dependency unavailable"):
        report_run(output)
    failures = output / "independent-failures.jsonl"
    assert failures.is_file()
    assert "dependency unavailable" in json.loads(failures.read_text())["reason"]
