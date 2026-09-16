import json

import pytest

from agg.experiments import ExperimentConfig, Features, run_experiment
from agg.experiments.matrix import ablation_matrix, retrieval_factorial
from agg.training import TrainConfig


def tiny_config(**kwargs):
    return ExperimentConfig(
        training=TrainConfig(steps=2, eval_every=1, width=16, heads=2, layers=2),
        modulus=7,
        candidate_steps=1,
        dimensions=(8,),
        geometries=("euclidean",),
        **kwargs,
    )


def test_baseline_flags_remove_all_agg_artifacts(tmp_path):
    result = run_experiment(tiny_config(features=Features.baseline()), tmp_path / "baseline")
    assert (tmp_path / "baseline" / "training" / "metrics.jsonl").exists()
    assert not (tmp_path / "baseline" / "telemetry.jsonl").exists()
    assert not (tmp_path / "baseline" / "ledger.jsonl").exists()
    assert not (tmp_path / "baseline" / "storage.json").exists()
    assert result["interventions"] == 0


def test_full_pipeline_records_candidates_evidence_and_rollback(tmp_path):
    result = run_experiment(tiny_config(), tmp_path / "full")
    records = [
        json.loads(line) for line in (tmp_path / "full" / "ledger.jsonl").read_text().splitlines()
    ]
    assert {"geometry", "pruning", "precision", "self_distillation"} <= {
        r["intervention_type"] for r in records
    }
    assert result["interventions"] == len(records)
    assert (tmp_path / "full" / "telemetry.jsonl").exists()
    assert (tmp_path / "full" / "execution.json").exists()
    storage = json.loads((tmp_path / "full" / "storage.json").read_text())
    assert storage["illustrative"]["tiny"]["selected_bytes"] == "dense"
    assert storage["illustrative"]["sparse"]["selected_bytes"] == "bitcos_like"
    assert all(r["state_before"] == r["state_retained"] for r in records if not r["accepted"])


def test_factorial_has_independent_axes_and_all_ablation_modes():
    grid = retrieval_factorial(
        gates=(0.0, 1.0), distances=(1, 3), densities=(0.0, 1.0), lengths=(8, 16), seeds=(1, 2)
    )
    assert len(grid) == 32
    assert len({(c.gate, c.distance, c.density, c.context_length) for c in grid}) == 16
    assert all(c.task == "retrieval" for c in grid)
    matrix = ablation_matrix()
    assert {
        "baseline",
        "telemetry_only",
        "full_a",
        "a_b",
        "a_c",
        "a_b_c",
        "immediate_credit",
        "discounted_credit",
        "storage_only",
    } <= matrix.keys()


def test_invalid_config_fails_before_creating_run(tmp_path):
    with pytest.raises(ValueError):
        ExperimentConfig(task="unknown")
    with pytest.raises(ValueError):
        ExperimentConfig(dimensions=(128,), training=TrainConfig(width=16))
    with pytest.raises(ValueError):
        retrieval_factorial(distances=(7,), lengths=(8,))


def test_state_hash_distinguishes_inference_precision():
    import copy

    from agg.consolidation import state_hash
    from agg.experiments.candidates import ActivationPrecisionModel
    from agg.models import TinyTransformer

    base = TinyTransformer(10, 3, width=16, heads=2)
    a = ActivationPrecisionModel(copy.deepcopy(base), "INT8")
    b = ActivationPrecisionModel(copy.deepcopy(base), "INT4")
    assert state_hash(a) != state_hash(b)


def test_fit_candidate_restores_teacher_mode():
    from agg.experiments.candidates import fit_candidate
    from agg.models import TinyTransformer
    from agg.tasks import modular_addition

    teacher = TinyTransformer(8, 7, width=16, heads=2)
    student = TinyTransformer(8, 7, width=16, heads=2)
    teacher.train()
    teacher.norm.eval()
    before = [m.training for m in teacher.modules()]
    fit_candidate(student, teacher, modular_addition(7), tiny_config(), True)
    assert before == [m.training for m in teacher.modules()]


def test_fit_setup_failure_does_not_mutate_teacher():
    import torch

    from agg.experiments.candidates import fit_candidate
    from agg.models import TinyTransformer
    from agg.tasks import modular_addition

    teacher = TinyTransformer(8, 7, width=16, heads=2)
    teacher.train()
    with pytest.raises(ValueError):
        fit_candidate(torch.nn.Identity(), teacher, modular_addition(7), tiny_config(), True)
    assert teacher.training


def test_baseline_cli_removes_training_time_adapter(tmp_path):
    import subprocess
    import sys

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "agg.experiments.cli",
            "run",
            "--config",
            "configs/retrieval-smoke.json",
            "--ablation",
            "baseline",
            "--steps",
            "1",
            "--output",
            str(tmp_path / "baseline"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert not (tmp_path / "baseline" / "telemetry.jsonl").exists()


def test_geometry_only_observation_does_not_compute_derivatives_or_attribution(
    tmp_path, monkeypatch
):
    import agg.experiments.diagnostics
    import agg.telemetry

    def forbidden(*args, **kwargs):
        pytest.fail("disabled diagnostic executed")

    monkeypatch.setattr(agg.telemetry, "depth_derivatives", forbidden)
    monkeypatch.setattr(agg.experiments.diagnostics, "representation_diagnostics", forbidden)
    flags = Features.baseline()
    flags.telemetry = flags.pruning = flags.consolidation = True
    run_experiment(tiny_config(features=flags, telemetry_mode="geometry"), tmp_path / "geometry")


def test_dimension_disabled_training_adapter_uses_model_width(tmp_path):
    from agg.experiments.checkpoints import load_model

    flags = Features.baseline()
    flags.gating = True
    run_experiment(tiny_config(features=flags, adapt_during_training=True), tmp_path / "gates")
    model = load_model(tmp_path / "gates" / "selected-state.pt")
    assert model.adapter.dimension == 16


def test_disabled_quantization_never_runs_precision_candidates(tmp_path):
    features = Features.baseline()
    features.consolidation = True
    features.pruning = True
    run_experiment(tiny_config(features=features), tmp_path / "prune")
    records = [
        json.loads(line) for line in (tmp_path / "prune" / "ledger.jsonl").read_text().splitlines()
    ]
    assert {r["intervention_type"] for r in records} == {"pruning"}


@pytest.mark.parametrize(
    "name,kinds,artifact",
    [
        ("baseline", set(), None),
        ("telemetry_only", set(), "telemetry.jsonl"),
        ("self_distillation_only", {"self_distillation"}, None),
        ("geometry_only", {"geometry"}, None),
        ("dimension_only", {"geometry"}, None),
        ("gating_only", {"geometry"}, "gate-ablation.json"),
        ("epiplexity_only", set(), "complexity.json"),
        ("pruning_only", {"pruning"}, None),
        ("quantization_only", {"precision", "activation_precision"}, "precision-map.json"),
        ("storage_only", set(), "storage.json"),
        ("executable_only", set(), "execution.json"),
        ("teacher_only", set(), "execution.json"),
        ("process_only", set(), "execution.json"),
        ("verified_only", set(), "execution.json"),
        ("immediate_credit", set(), "execution.json"),
        ("discounted_credit", set(), "execution.json"),
        ("full_a", {"self_distillation"}, "execution.json"),
        ("a_b", {"self_distillation", "geometry"}, "gate-ablation.json"),
        (
            "a_c",
            {"self_distillation", "pruning", "precision", "activation_precision"},
            "storage.json",
        ),
        (
            "a_b_c",
            {"self_distillation", "geometry", "pruning", "precision", "activation_precision"},
            "storage.json",
        ),
    ],
)
def test_every_named_ablation_removes_other_interventions(tmp_path, name, kinds, artifact):
    config = tiny_config(
        features=ablation_matrix()[name],
        precisions=("INT8",),
        activation_precisions=("INT8",),
        gamma=0 if name == "immediate_credit" else 0.9,
    )
    output = tmp_path / name
    run_experiment(config, output)
    path = output / "ledger.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    assert {record["intervention_type"] for record in records} == kinds
    if artifact:
        assert (output / artifact).exists()
    if name == "teacher_only":
        evidence = json.loads((output / "execution.json").read_text())
        assert evidence["trace"] == []
        assert evidence["credit"][1]["local_evidence"]["primary_source"] == "teacher"
    if name == "verified_only":
        evidence = json.loads((output / "execution.json").read_text())
        assert evidence["credit"][1]["local_evidence"]["value"] == -1
