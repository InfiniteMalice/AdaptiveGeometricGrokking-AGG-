import copy

import pytest
import torch
from torch import nn

from agg.consolidation import Proposal, trial
from agg.evaluation import Constraints, Evaluation
from agg.ledger import Ledger


def assessment(model):
    accuracy = float(model.weight.detach().mean())
    return Evaluation(accuracy, accuracy, 1.0, cost=10.0)


def test_rejected_candidate_restores_parameters_buffers_and_rng(tmp_path):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()

    def destroy(candidate):
        with torch.no_grad():
            candidate.weight.zero_()
        torch.rand(10)
        return candidate

    result = trial(
        model,
        Proposal("prune", "weight", {}, destroy),
        assessment,
        Constraints(),
        Ledger(tmp_path / "ledger.jsonl"),
        run_id="test",
        step=0,
    )
    assert not result.accepted
    assert result.model is model
    assert "ID" in result.reason
    torch.testing.assert_close(before["weight"], model.weight, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    record = Ledger(tmp_path / "ledger.jsonl").read()[0]
    assert record["accepted"] is False
    assert record["state_before"] == record["state_retained"]
    assert record["id_delta"] == -1


def test_mechanism_failure_rejected_even_when_behavior_identical(tmp_path):
    model = nn.Linear(2, 2)
    result = trial(
        model,
        Proposal("geometry", "all", {}, lambda m: m),
        lambda m: Evaluation(1, 1, 0.2, cost=5),
        Constraints(mechanism_min=0.9),
        Ledger(tmp_path / "ledger.jsonl"),
        run_id="test",
        step=1,
        reference=Evaluation(1, 1, 1, cost=10),
    )
    assert not result.accepted
    assert "mechanism" in result.reason


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_candidate_fails_closed(bad):
    reasons = Constraints().violations(Evaluation(1, 1, 1), Evaluation(bad, 1, 1))
    assert reasons


def test_exception_is_ledgered_and_reference_untouched(tmp_path):
    model = nn.Linear(2, 2)

    def explode(candidate):
        raise RuntimeError("candidate failed")

    result = trial(
        model,
        Proposal("precision", "weight", {}, explode),
        lambda m: Evaluation(1, 1, 1),
        Constraints(),
        Ledger(tmp_path / "ledger.jsonl"),
        run_id="test",
        step=1,
    )
    assert not result.accepted
    assert result.model is model
    assert "candidate failed" in result.reason


def test_feasible_candidate_accepted_but_cost_increase_rejected(tmp_path):
    model = nn.Linear(2, 2)
    accepted = trial(
        model,
        Proposal("noop", "model", {}, lambda m: m),
        lambda m: Evaluation(1, 1, 1, cost=1),
        Constraints(),
        Ledger(tmp_path / "a.jsonl"),
        run_id="test",
        step=0,
    )
    assert accepted.accepted and accepted.model is not model
    constraints = Constraints(require_cost_reduction=True)
    assert constraints.violations(Evaluation(1, 1, 1, cost=1), Evaluation(1, 1, 1, cost=2))


def test_execution_and_reproducibility_are_independent_constraints():
    baseline = Evaluation(1, 1, 1, execution_score=1)
    candidate = Evaluation(1, 1, 1, execution_score=0, reproducible=False)
    reasons = Constraints(require_execution=True).violations(baseline, candidate)
    assert any("execution" in r for r in reasons)
    assert any("reproduc" in r for r in reasons)


@pytest.mark.parametrize("tolerance, accepted", [(0.4, True), (0.15, False)])
def test_observer_records_measured_deltas_against_local_before_not_anchor(
    tmp_path, tolerance, accepted
):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.8)

    def change(candidate):
        with torch.no_grad():
            candidate.weight.fill_(0.7)
        return candidate

    def observer(candidate):
        value = float(candidate.weight.detach().mean())
        return {
            "memorization": {"score": value},
            "retrieval": {"score": 1 - value},
            "generalization": {"score": value},
            "sparsity": {"parameters": 4, "active_parameters": 4},
            "geometry": {"anisotropy": value},
            "dimension": {"effective_rank": value},
            "complexity": {"bits": value, "proxy": True},
        }

    ledger = Ledger(tmp_path / "observer.jsonl")
    result = trial(
        model,
        Proposal("precision", "weight", {"weight": "INT8"}, change),
        assessment,
        Constraints(id_tolerance=tolerance, ood_tolerance=tolerance),
        ledger,
        run_id="test",
        step=1,
        reference=Evaluation(1, 1, 1, cost=10),
        observer=observer,
    )
    assert result.accepted is accepted
    record = ledger.read()[0]
    assert record["evaluation_before"]["id_accuracy"] == pytest.approx(0.8)
    assert record["evaluation_reference"]["id_accuracy"] == 1
    assert record["id_delta"] == pytest.approx(-0.1)
    assert record["memorization_delta"] == pytest.approx(-0.1)
    assert record["retrieval_delta"] == pytest.approx(0.1)
    assert record["generalization_delta"] == pytest.approx(-0.1)
    assert record["parameter_delta"] == 0 and record["sparsity_delta"] == 0
    assert record["telemetry_after"]["geometry"]["anisotropy"] == pytest.approx(0.7)
    assert record["precision_delta"]["after"] == {"weight": "INT8"}
    assert record["precision_delta"]["before"] is None


def test_baseline_evaluator_exception_restores_all_rng_and_propagates(tmp_path):
    import random

    import numpy as np

    model = nn.Linear(2, 2)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )

    def broken_reference(candidate):
        random.random()
        np.random.rand()
        torch.rand(4)
        raise RuntimeError("invalid baseline")

    ledger = Ledger(tmp_path / "baseline.jsonl")
    with pytest.raises(RuntimeError, match="invalid baseline"):
        trial(
            model,
            Proposal("noop", "all", {}, lambda m: m),
            broken_reference,
            Constraints(),
            ledger,
            run_id="test",
            step=0,
        )
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert ledger.read() == []


@pytest.mark.parametrize("fail_call", [1, 2])
def test_observer_exception_rejects_and_cannot_mutate_original(tmp_path, fail_call):
    model = nn.Linear(2, 2)
    original = copy.deepcopy(model.state_dict())
    calls = 0

    def bad_observer(candidate):
        nonlocal calls
        calls += 1
        if calls == fail_call:
            with torch.no_grad():
                candidate.weight.zero_()
            raise RuntimeError("observer unavailable")
        return {"memorization": {"score": 0.5}}

    ledger = Ledger(tmp_path / "failure.jsonl")
    result = trial(
        model,
        Proposal("noop", "all", {}, lambda m: m),
        lambda m: Evaluation(1, 1, 1),
        Constraints(),
        ledger,
        run_id="test",
        step=0,
        observer=bad_observer,
    )
    assert not result.accepted and result.model is model
    assert "observer unavailable" in result.reason
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    assert ledger.read()[0]["telemetry_after"] is None
