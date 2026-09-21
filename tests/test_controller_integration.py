import copy

import pytest


def test_four_phase_synthetic_demo(tmp_path):
    from agg.controller.demo import run_demo

    result = run_demo(tmp_path / "demo")
    assert result["outcomes"] == ["rolled_back", "committed"]
    assert result["actions"] == ["increase_replay", "increase_replay_diversity"]
    assert result["accepted_budget"] > result["initial_budget"]
    assert "protected regression" in (tmp_path / "demo" / "timeline.txt").read_text()


@pytest.mark.parametrize(
    "samples,task,score",
    [
        (2, "modular", "accuracy"),
        (3, "hierarchy", "balanced_accuracy"),
        (3, "hierarchy", "accuracy"),
    ],
)
def test_training_callback_and_real_copy_based_trial(tmp_path, samples, task, score):
    import torch

    from agg.consolidation import state_hash
    from agg.controller.config import ControllerConfig, MetricGuard
    from agg.controller.core import Controller
    from agg.controller.events import EventLog
    from agg.controller.training import TrainingExecutor, TrainingObserver, TrainingPolicy
    from agg.evaluation import Constraints
    from agg.evaluation.model import make_evaluator
    from agg.ledger import Ledger
    from agg.tasks import hierarchy, modular_addition
    from agg.training import TrainConfig, evaluate, train

    data = modular_addition(5) if task == "modular" else hierarchy(depth=2)
    config = ControllerConfig(
        evaluation_window=4,
        evaluation_samples=samples,
        minimum_gain=0.00001,
        protected_metrics=(MetricGuard("performance.ood_score", 1.0),),
    )
    controller = Controller(
        config,
        EventLog(Ledger(tmp_path / "events.jsonl"), "training"),
        policy=TrainingPolicy(config),
    )
    observer = TrainingObserver(controller, score=score)
    training = TrainConfig(steps=1, eval_every=1, width=8, heads=2, layers=1)
    result = train(data, training, tmp_path / "initial", callback=observer)
    assert observer.latest is not None
    original = state_hash(result.model)
    # A frozen checkpoint evaluated repeatedly has a genuine flat trajectory.
    for step in range(2, 12):
        metrics = {
            name: evaluate(result.model, split)
            for name, split in (("train", data.train), ("id", data.id), ("ood", data.ood))
        }
        observer(step, result.model, metrics)
    proposal = observer.latest
    assert proposal.action.value == "adjust_learning_rate"
    backend = TrainingExecutor(
        result.model,
        data,
        training,
        evaluator=make_evaluator(result.model, data),
        constraints=Constraints(id_tolerance=1, ood_tolerance=1, mechanism_min=0),
        ledger=Ledger(tmp_path / "trials.jsonl"),
        output=tmp_path / "candidates",
        score=score,
    )
    rng = torch.get_rng_state().clone()
    assert controller.start(proposal, backend).status == "pending"
    assert state_hash(result.model) == original
    assert torch.equal(rng, torch.get_rng_state())
    assert backend.evaluate().samples >= samples
    assert controller.finish().status == "committed"
    assert backend.model is not result.model
    assert state_hash(result.model) == original
    assert backend.config.learning_rate > training.learning_rate
    assert (tmp_path / "candidates" / proposal.id / "checkpoint-4.pt").exists()
    strict = TrainingExecutor(
        result.model,
        data,
        training,
        evaluator=make_evaluator(result.model, data),
        constraints=Constraints(require_execution=True),
        ledger=Ledger(tmp_path / "strict-trials.jsonl"),
        output=tmp_path / "strict-candidates",
        score=score,
    )
    with pytest.raises(RuntimeError, match="required execution evidence unavailable"):
        strict.stage(proposal)
    strict.rollback()
    assert strict.model is result.model and state_hash(result.model) == original
    assert Ledger(tmp_path / "strict-trials.jsonl").read()[-1]["accepted"] is False
    from dataclasses import replace

    accepted = backend.model
    with pytest.raises(ValueError):
        backend.stage(replace(proposal, id="next", parameters={"fraction": 0.9}))
    backend.rollback()
    assert backend.model is accepted
    with pytest.raises(ValueError, match="score definitions"):
        backend.stage(replace(proposal, id="wrong-score", baseline_provenance={"score": "other"}))
    backend.rollback()
    assert backend.model is accepted
    with pytest.raises(ValueError, match="accommodate"):
        backend.stage(replace(proposal, id="short-window", evaluation_samples=10))
    backend.rollback()
    assert backend.model is accepted


def test_real_training_trial_cannot_bypass_existing_constraints(tmp_path):
    # The provider rejects unsupported replay instead of reporting training execution.
    from agg.controller.config import ControllerConfig
    from agg.controller.core import Controller
    from agg.controller.demo import replay_observation
    from agg.controller.events import EventLog
    from agg.controller.training import TrainingExecutor
    from agg.evaluation import Constraints
    from agg.evaluation.model import make_evaluator
    from agg.ledger import Ledger
    from agg.tasks import modular_addition
    from agg.training import TrainConfig, train

    data = modular_addition(5)
    config = TrainConfig(steps=1, width=8, heads=2, layers=1)
    trained = train(data, config, tmp_path / "model")
    backend = TrainingExecutor(
        trained.model,
        data,
        config,
        evaluator=make_evaluator(trained.model, data),
        constraints=Constraints(),
        ledger=Ledger(tmp_path / "trials.jsonl"),
        output=tmp_path / "candidates",
    )
    controller = Controller(ControllerConfig(), EventLog(Ledger(tmp_path / "events.jsonl"), "test"))
    for step in range(9):
        proposal = controller.observe(replay_observation(step, 0.9 - step * 0.03))
    original = copy.deepcopy(trained.model.state_dict())
    assert controller.start(proposal, backend).status == "unsupported"
    assert not (tmp_path / "candidates").exists()
    for key, value in trained.model.state_dict().items():
        assert value.equal(original[key])


def test_demo_refuses_overwrite(tmp_path):
    from agg.controller.demo import run_demo

    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
