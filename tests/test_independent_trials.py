import pytest
import torch

from agg.consolidation import Proposal, state_hash, trial
from agg.evaluation import Constraints, Evaluation
from agg.experiments.checkpoints import load_model
from agg.experiments.independent import CandidateRecorder
from agg.ledger import Ledger
from agg.models import TinyTransformer


def test_all_candidate_attempts_are_recorded_and_parent_preserved(tmp_path):
    model = TinyTransformer(8, 3, width=8, heads=2, layers=1)
    original = state_hash(model)
    recorder = CandidateRecorder(tmp_path, parent_checkpoint=None)

    def broken(candidate):
        raise RuntimeError("deliberate apply failure")

    for apply, constraints in (
        (lambda m: m, Constraints()),
        (lambda m: m, Constraints(require_cost_reduction=True)),
        (broken, Constraints()),
    ):
        trial(
            model,
            Proposal("fixture", "model", {}, apply),
            lambda m: Evaluation(1, 1, 1, cost=1),
            constraints,
            Ledger(tmp_path / "ledger.jsonl"),
            run_id="fixture",
            step=0,
            recorder=recorder,
        )
    assert state_hash(model) == original
    records = recorder.records()
    assert [r["event"] for r in records] == ["proposed", "completed"] * 3
    outcomes = [r for r in records if r["event"] == "completed"]
    assert [r["status"] for r in outcomes] == ["accepted", "rejected", "failed"]
    assert [r["candidate_index"] for r in outcomes] == [1, 2, 3]
    assert outcomes[1]["checkpoint"] is not None  # rejected models remain auditable
    assert outcomes[2]["checkpoint"] is None
    assert all(r["parent_state"] == original for r in outcomes)
    assert all(r["compute_flops"] is None for r in outcomes)


def test_reference_failure_is_accounted_before_propagation(tmp_path):
    model = TinyTransformer(8, 3, width=8, heads=2, layers=1)
    recorder = CandidateRecorder(tmp_path, parent_checkpoint=None)
    rng = torch.get_rng_state().clone()

    def failing_evaluator(candidate):
        torch.rand(4)
        raise RuntimeError("reference unavailable")

    with pytest.raises(RuntimeError, match="reference unavailable"):
        trial(
            model,
            Proposal("fixture", "model", {}, lambda m: m),
            failing_evaluator,
            Constraints(),
            Ledger(tmp_path / "ledger.jsonl"),
            run_id="test",
            step=0,
            recorder=recorder,
        )
    assert torch.equal(rng, torch.get_rng_state())
    assert recorder.records()[-1]["status"] == "reference_failed"
    assert recorder.records()[-1]["selection_after"] is None


def test_unavailable_intent_sink_prevents_candidate_execution(tmp_path):
    model = TinyTransformer(8, 3, width=8, heads=2, layers=1)
    recorder = CandidateRecorder(tmp_path, parent_checkpoint=None)
    (tmp_path / "candidate-attempts.jsonl").mkdir()
    original = state_hash(model)
    with pytest.raises(OSError):
        trial(
            model,
            Proposal("fixture", "model", {}, lambda m: m),
            lambda m: Evaluation(1, 1, 1),
            Constraints(),
            Ledger(tmp_path / "ledger.jsonl"),
            run_id="test",
            step=0,
            recorder=recorder,
        )
    assert state_hash(model) == original


def test_sequential_candidate_parent_checkpoint_matches_actual_parent(tmp_path):
    model = TinyTransformer(8, 3, width=8, heads=2, layers=1)
    recorder = CandidateRecorder(tmp_path, parent_checkpoint=None)

    def change(candidate):
        with torch.no_grad():
            candidate.readout.bias.add_(0.5)
        return candidate

    first = trial(
        model,
        Proposal("fixture", "model", {}, change),
        lambda m: Evaluation(1, 1, 1),
        Constraints(),
        Ledger(tmp_path / "log.jsonl"),
        run_id="test",
        step=0,
        recorder=recorder,
    )
    trial(
        first.model,
        Proposal("fixture", "model", {}, lambda m: m),
        lambda m: Evaluation(1, 1, 1),
        Constraints(),
        Ledger(tmp_path / "log.jsonl"),
        run_id="test",
        step=0,
        recorder=recorder,
    )
    record = recorder.records()[-1]
    assert record["parent_checkpoint"] is not None
    parent = load_model(tmp_path / record["parent_checkpoint"])
    assert state_hash(parent) == record["parent_state"] == state_hash(first.model)
