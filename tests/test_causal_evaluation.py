import json
from dataclasses import replace

import pytest
import torch

from agg.evaluation.causal import intervention_metrics
from agg.experiments.causal import causal_report, phase_checkpoints
from agg.experiments.config import ExperimentConfig, Features
from agg.experiments.independent import file_hash
from agg.experiments.runner import run_experiment
from agg.tasks.causal import intervention_pairs, oracle
from agg.tasks.protocol import make_protocol
from agg.training import TrainConfig


@pytest.mark.parametrize("task", ["modular", "hierarchy", "retrieval"])
def test_oracle_validates_pairs_and_protected_support(task):
    protocol = make_protocol(task, seed=31, depth=4)
    data = protocol.evaluation("audit")
    for name in ("id", "ood"):
        split = getattr(data, name)
        assert torch.equal(oracle(split.x, protocol.metadata), split.y)
        for kind in ("invariant", "decisive"):
            pairs = intervention_pairs(
                split, getattr(data, name + "_clusters"), protocol.metadata, kind=kind, seed=9
            )
            assert pairs.attempted == len(split.y)
            assert pairs.attempted == len(pairs.original.y) + len(pairs.missing)
            if not len(pairs.original.y):
                continue
            expected = oracle(pairs.transformed.x, protocol.metadata)
            assert torch.equal(expected, pairs.transformed.y)
            assert bool((pairs.original.y == expected).all()) == (kind == "invariant")
            assert bool((pairs.original.x != pairs.transformed.x).any(1).all())
            if task != "retrieval":
                support = {tuple(row) for row in split.x.tolist()}
                assert all(tuple(row) in support for row in pairs.transformed.x.tolist())
            else:
                assert torch.equal(pairs.original.x[:, -1], pairs.transformed.x[:, -1])


def test_wrong_consistency_is_not_correct_invariance_and_updates_are_rewarded():
    result = intervention_metrics(
        [1, 1], [1, 0], [0, 1], [0, 1], ["a", "b"], kind="invariant", seed=3
    )
    assert result["correct_invariance"]["value"] == 0
    assert result["spurious_flip"]["value"] == 0.5
    decisive = intervention_metrics(
        [0, 0], [1, 1], [0, 1], [1, 0], ["a", "b"], kind="decisive", seed=3
    )
    assert decisive["required_update"]["value"] == 1
    assert decisive["required_update"]["denominator"] == 1
    assert decisive["joint_correctness"]["value"] == 0.5
    no_original = intervention_metrics([1], [1], [0], [1], ["a"], kind="decisive", seed=3)
    assert no_original["required_update"]["value"] is None
    assert no_original["failure_recovery"]["value"] == 1
    with pytest.raises(ValueError, match="labels"):
        intervention_metrics([0], [0], [0], [1], ["a"], kind="invariant", seed=3)


def test_oracle_rejects_ambiguous_retrieval_and_noop_is_missing():
    from agg.tasks.synthetic import Split

    metadata = {"task": "retrieval", "keys": 4, "values": 4}
    with pytest.raises(ValueError, match="unique"):
        oracle(torch.tensor([[1, 2, 17]]), metadata)
    split = Split(torch.tensor([[0, 1, 17]]), torch.tensor([0]))
    pairs = intervention_pairs(split, ("a",), metadata, kind="invariant", seed=1)
    assert len(pairs.original.y) == 0
    assert len(pairs.missing) == 1


def test_phase_mapping_preserves_censoring_and_requires_post_confirmation():
    assert phase_checkpoints([0, 5, 10], None)["during"]["step"] is None
    crossing = {
        "stable_crossing": 5,
        "confirmation_step": 10,
        "interval": [0, 5],
        "censored": False,
        "last_step": 10,
    }
    result = phase_checkpoints([0, 5, 10], crossing)
    assert result["before"]["step"] == 0
    assert result["during"]["step"] == 5
    assert result["after"]["step"] is None
    assert phase_checkpoints([0, 5, 10, 15], crossing)["after"]["step"] == 15


def test_frozen_causal_trajectory_is_report_only_and_one_use(tmp_path, monkeypatch):
    config = ExperimentConfig(
        task="retrieval",
        independent_evaluation=True,
        data_seed=71,
        samples=16,
        dimensions=(4,),
        training=TrainConfig(steps=2, eval_every=1, width=8, heads=2, layers=1),
        features=replace(Features.baseline(), telemetry=False),
    )
    run = tmp_path / "run"
    run_experiment(config, run)
    before = file_hash(run / "selected-state.pt")
    report = causal_report(run, seed=18)
    assert [row["step"] for row in report["trajectory"]] == [0, 1, 2]
    assert report["phase"]["during"]["step"] is None
    assert report["selected"]["id"]["decisive"]["metrics"]["samples"] == 48
    accuracy = report["selected"]["id"]["structural_accuracy"]
    assert accuracy["denominator"] == 48
    assert accuracy["uncertainty"]["clusters"] == 4
    sensitivity = report["selected"]["id"]["causal_sensitivity"]
    assert sensitivity["decisive_denominator"] == 48
    assert sensitivity["uncertainty_unavailable_reason"] is not None
    assert file_hash(run / "selected-state.pt") == before
    assert not (run / "final-attempt.json").exists()
    assert json.loads((run / "causal-report.json").read_text()) == report
    with pytest.raises(ValueError, match="already"):
        causal_report(run)
    other = tmp_path / "failed-checkpoint"
    run_experiment(config, other)
    from agg.experiments import causal

    load = causal.load_model

    def fail_one(path):
        if path.name == "inference-1.pt":
            raise FloatingPointError("fixture failure")
        return load(path)

    monkeypatch.setattr(causal, "load_model", fail_one)
    mapper = causal.phase_checkpoints
    seen_steps = []

    def with_crossing(steps, crossing):
        seen_steps.extend(steps)
        return mapper(
            steps,
            {"stable_crossing": 1, "confirmation_step": 1, "interval": [0, 1], "censored": False},
        )

    monkeypatch.setattr(causal, "phase_checkpoints", with_crossing)
    failed = causal_report(other)
    assert failed["trajectory"][1]["measurement"] is None
    assert "fixture failure" in failed["trajectory"][1]["missing_reason"]
    assert failed["trajectory"][2]["measurement"] is not None
    assert seen_steps == [0, 2]
    assert failed["phase"]["during"]["step"] is None
    assert failed["phase"]["after"]["step"] == 2


def test_shared_transformation_endpoints_merge_statistical_clusters():
    from agg.tasks.synthetic import Split

    split = Split(torch.tensor([[0, 1, 5], [1, 0, 5]]), torch.tensor([1, 1]))
    pairs = intervention_pairs(
        split, ("a", "b"), {"task": "modular", "modulus": 5}, kind="invariant", seed=2
    )
    assert len(pairs.original.y) == 2
    assert len(set(pairs.clusters)) == 1


@pytest.mark.parametrize(
    "metadata,kind,rows",
    [
        ({"task": "modular", "modulus": 5}, "decisive",
         [[0, 0, 5], [0, 1, 5], [0, 2, 5], [0, 3, 5]]),
        ({"task": "hierarchy", "branching": 2, "depth": 2}, "invariant",
         [[0, 1, 7], [0, 2, 7], [0, 3, 7], [0, 4, 7]]),
        ({"task": "hierarchy", "branching": 2, "depth": 2}, "decisive",
         [[1, 3, 7], [1, 2, 7], [1, 4, 7], [1, 5, 7]]),
    ],
)
def test_generic_replacements_use_only_same_cluster_candidates(metadata, kind, rows):
    from agg.tasks.synthetic import Split

    x = torch.tensor(rows)
    split = Split(x, oracle(x, metadata))
    clusters = ("a", "a", "b", "b")
    for seed in range(8):
        pairs = intervention_pairs(split, clusters, metadata, kind=kind, seed=seed)
        assert pairs.source_indices == [0, 1, 2, 3]
        assert torch.equal(pairs.transformed.x, x[[1, 0, 3, 2]])
        assert pairs.clusters == clusters
        assert pairs.missing == []

    # Keep semantically eligible alternatives, but put each in a different cluster.
    missing = intervention_pairs(split, ("a", "b", "c", "d"), metadata, kind=kind, seed=0)
    assert missing.attempted == 4
    assert missing.source_indices == []
    assert missing.clusters == ()
    assert len(missing.transformed.y) == 0
    assert missing.missing == [
        {"source_index": i, "reason": "no nontrivial supported transform"} for i in range(4)
    ]


def test_modular_decisive_pairs_are_missing_under_unordered_pair_clusters():
    protocol = make_protocol("modular", seed=31)
    data = protocol.evaluation("audit")
    for stratum in ("id", "ood"):
        split = getattr(data, stratum)
        pairs = intervention_pairs(
            split, getattr(data, stratum + "_clusters"), protocol.metadata,
            kind="decisive", seed=9,
        )
        assert len(pairs.original.y) == 0
        assert len(pairs.missing) == pairs.attempted == len(split.y)
