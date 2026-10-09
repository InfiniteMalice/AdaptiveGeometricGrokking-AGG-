from dataclasses import replace

import pytest
import torch

from agg.consolidation import Proposal, trial
from agg.evaluation import Constraints, Evaluation
from agg.experiments.config import ExperimentConfig
from agg.experiments.independent import selection_comparison
from agg.experiments.runner import make_data
from agg.ledger import Ledger


def test_conservative_acceptance_uses_selection_only_and_keeps_protection(tmp_path):
    model = torch.nn.Linear(1, 1)
    before = Evaluation(0.8, 0.8, 1, cost=1)
    for floor, candidate, accepted in (
        (None, Evaluation(0.8, 0.8, 1, cost=1), True),
        (0.01, Evaluation(0.8, 0.8, 1, cost=1), False),
        (0.01, Evaluation(0.9, 0.9, 1, cost=1), True),
        (0.01, Evaluation(0.9, 0.7, 1, cost=0.5), False),
    ):
        outcomes = iter([before, candidate])
        result = trial(
            model,
            Proposal("test", "model", {}, lambda m: m),
            lambda m, outcomes=outcomes: next(outcomes),
            Constraints(),
            Ledger(tmp_path / "log.jsonl"),
            run_id="test",
            step=0,
            selection_gain_floor=floor,
        )
        assert result.accepted is accepted


def test_comparison_changes_selection_size_without_changing_training():
    base = ExperimentConfig(task="retrieval", independent_evaluation=True, data_seed=19, samples=32)
    arms = selection_comparison(base)
    assert len(arms) == 4
    reference = make_data(base)
    for config in arms.values():
        assert config.training == base.training
        assert config.candidate_steps == base.candidate_steps
        assert torch.equal(make_data(config).train.x, reference.train.x)
    assert len(make_data(arms["larger_selection"]).id.y) > len(reference.id.y)
    assert arms["conservative"].selection_gain_floor == 0.01
    with pytest.raises(ValueError):
        replace(base, selection_gain_floor=float("nan"))
