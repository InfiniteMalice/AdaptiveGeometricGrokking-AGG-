import pytest
import torch

from agg.controller.resources import ResourceProfile, measure_resources


def test_resource_requirements_are_conjunctive_and_missing_fails_closed():
    profile = ResourceProfile(min_id=0.8, min_ood=0.8, max_latency_ms=10, max_model_bytes=1000)
    metrics = {
        "performance.validation_score": 0.9,
        "performance.ood_score": 0.79,
        "resources.latency_ms": 1,
        "resources.model_bytes": 10,
    }
    report = profile.assess(metrics)
    assert report["joint_feasible"] is False
    assert report["requirements"]["min_ood"]["margin"] == pytest.approx(-0.01)
    assert report["requirements"]["max_latency_ms"]["passed"] is True
    metrics["performance.ood_score"] = 0.9
    assert profile.assess(metrics)["joint_feasible"] is True
    del metrics["resources.model_bytes"]
    report = profile.assess(metrics)
    assert not report["joint_feasible"]
    assert report["requirements"]["max_model_bytes"]["margin"] is None
    assert report["requirements"]["max_model_bytes"]["missing_reason"]
    assert not ResourceProfile(max_peak_memory_bytes=10000).assess(metrics)["joint_feasible"]


def test_resource_profile_is_optional_typed_and_round_trips():
    from agg.controller.config import ControllerConfig

    assert ControllerConfig().resources is None
    config = ControllerConfig(resources=ResourceProfile(min_ood=0.8))
    assert ControllerConfig.from_dict(config.to_dict()) == config
    for kwargs in (
        {"min_ood": 1.1},
        {"max_latency_ms": float("nan")},
        {"max_model_bytes": -1},
        {"max_training_updates": True},
    ):
        with pytest.raises(ValueError):
            ResourceProfile(**kwargs)


def test_measured_resources_count_physical_storage_and_preserve_modes():
    from agg.experiments.candidates import ActivationPrecisionModel
    from agg.models import TinyTransformer

    model = TinyTransformer(10, 3, width=8, heads=2, layers=1, max_length=4)
    model.train()
    model.blocks[0].eval()
    modes = [m.training for m in model.modules()]
    x = torch.zeros((2, 4), dtype=torch.long)
    plain = measure_resources(model, x, repeats=3, warmup=1)
    simulated = measure_resources(ActivationPrecisionModel(model, "INT8"), x, repeats=3, warmup=1)
    assert [m.training for m in model.modules()] == modes
    assert (
        plain["metrics"]["resources.model_bytes"] == simulated["metrics"]["resources.model_bytes"]
    )
    assert plain["metrics"]["resources.parameter_count"] == sum(
        p.numel() for p in model.parameters()
    )
    assert len(plain["latency_samples_ms"]) == 3
    assert plain["metrics"]["resources.latency_ms"] > 0
    assert plain["batch_shape"] == [2, 4]
    assert "resources.peak_memory_bytes" in plain["unavailable"]
    assert plain["hardware"]["device"] == "cpu"


@pytest.mark.parametrize(
    "cost,ood,expected",
    [
        (1, 0.8, "committed"),
        (11, 0.8, "rolled_back"),
        (None, 0.8, "rolled_back"),
        (1, 0.5, "rolled_back"),
    ],
)
def test_controller_resource_gate_preserves_anchor(tmp_path, cost, ood, expected):
    from dataclasses import replace

    from test_controller_lifecycle import BudgetBackend, setup_controller

    controller, proposal, ledger = setup_controller(tmp_path)
    controller.config = replace(controller.config, resources=ResourceProfile(max_latency_ms=10))
    metrics = {"continual.retained_performance": 0.85, "performance.ood_score": ood}
    if cost is not None:
        metrics["resources.latency_ms"] = cost
    backend = BudgetBackend(proposal, metrics)
    assert controller.start(proposal, backend).status == "pending"
    outcome = controller.finish()
    assert outcome.status == expected
    if ood == 0.8:
        reports = [
            r["payload"]["assessment"]
            for r in ledger.read()
            if r["payload"].get("status") == "resource_evaluated"
        ]
        assert len(reports) == 1
        assert reports[0]["joint_feasible"] == (expected == "committed")
    if expected == "rolled_back":
        assert "commit" not in backend.calls


@pytest.mark.parametrize("batch_size", [8, 128])
def test_measured_training_provider_records_rejected_copy_and_budget(tmp_path, batch_size):
    from agg.consolidation import state_hash
    from agg.controller.config import ControllerConfig, MetricGuard
    from agg.controller.core import Controller
    from agg.controller.events import EventLog
    from agg.controller.resource_training import MeasuredTrainingExecutor
    from agg.controller.training import TrainingObserver, TrainingPolicy
    from agg.evaluation import Constraints
    from agg.evaluation.model import make_evaluator
    from agg.experiments.independent import CandidateRecorder
    from agg.ledger import Ledger
    from agg.tasks.protocol import make_protocol
    from agg.training import TrainConfig, evaluate, train

    data = make_protocol("retrieval", seed=7, samples=16, length=8, distance=2).development()
    training = TrainConfig(steps=1, width=8, heads=2, layers=1, batch_size=batch_size, eval_every=1)
    initial = train(data, training, tmp_path / "initial").model
    original = state_hash(initial)
    config = ControllerConfig(
        evaluation_window=2,
        evaluation_samples=2,
        minimum_gain=1e-9,
        protected_metrics=(MetricGuard("performance.ood_score", 1),),
        resources=ResourceProfile(max_model_bytes=1),
    )
    controller = Controller(
        config,
        EventLog(Ledger(tmp_path / "events.jsonl"), "resource"),
        policy=TrainingPolicy(config),
    )
    observer = TrainingObserver(controller)
    metrics = {
        name: evaluate(initial, split)
        for name, split in (("train", data.train), ("id", data.id), ("ood", data.ood))
    }
    for step in range(10):
        observer(step, initial, metrics)
    output = tmp_path / "recorded"
    output.mkdir()
    recorder = CandidateRecorder(output, parent_checkpoint=None)
    backend = MeasuredTrainingExecutor(
        initial,
        data,
        training,
        evaluator=make_evaluator(initial, data),
        constraints=Constraints(id_tolerance=1, ood_tolerance=1, mechanism_min=0),
        ledger=Ledger(tmp_path / "trial.jsonl"),
        output=output / "training",
        recorder=recorder,
        measurement_repeats=3,
    )
    assert controller.start(observer.latest, backend).status == "pending"
    assert controller.finish().status == "rolled_back"
    assert state_hash(backend.model) == original
    assert backend.measurement["metrics"]["resources.training_updates"] == 2
    assert (
        backend.measurement["metrics"]["resources.training_tokens"]
        == 2 * min(batch_size, len(data.train.y)) * 8
    )
    records = [r for r in recorder.records() if r["event"] == "completed"]
    assert len(records) == 1 and records[0]["checkpoint"]
    assert records[0]["training_updates_budget"] == 2
    assert backend.last_budget["completed_updates_lower_bound"] == 2


def test_training_budget_requires_preflight_before_execution(tmp_path):
    from dataclasses import replace

    from test_controller_lifecycle import BudgetBackend, setup_controller

    controller, proposal, ledger = setup_controller(tmp_path)
    controller.config = replace(
        controller.config, resources=ResourceProfile(max_training_updates=1)
    )
    backend = BudgetBackend(
        proposal, {"continual.retained_performance": 0.85, "performance.ood_score": 0.8}
    )
    assert controller.start(proposal, backend).status == "rejected"
    assert backend.calls == []


@pytest.mark.parametrize("failure", ["truncated", "unreadable"])
def test_partial_training_accounting_survives_log_failure(tmp_path, monkeypatch, failure):
    from pathlib import Path
    from types import SimpleNamespace

    from agg.controller.resource_training import MeasuredTrainingExecutor
    from agg.controller.training import TrainingExecutor

    provider = object.__new__(MeasuredTrainingExecutor)
    provider.output = tmp_path
    provider.config = SimpleNamespace(batch_size=8)
    provider.data = SimpleNamespace(
        train=SimpleNamespace(x=torch.zeros((20, 4)), y=torch.zeros(20))
    )
    provider.last_budget = {}
    proposal = SimpleNamespace(id="partial", evaluation_window=3)

    def interrupted_stage(self, proposal):
        path = self.output / proposal.id
        path.mkdir()
        (path / "metrics.jsonl").write_text('{"step":0}\n{"step":1}\n{"step":')
        raise RuntimeError("training interrupted after incurred work")

    monkeypatch.setattr(TrainingExecutor, "stage", interrupted_stage)
    if failure == "unreadable":

        def denied(*args, **kwargs):
            raise OSError("log unavailable")

        monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(RuntimeError, match="training interrupted"):
        provider.stage(proposal)
    # Same fallback expression used by the experiment must not turn unknown work into zero.
    budget = provider.last_budget or {"executed_updates": 0}
    assert budget["executed_updates"] is None
    assert budget["completed_updates_lower_bound"] == (1 if failure == "truncated" else 0)
    assert budget["missing_reason"]
