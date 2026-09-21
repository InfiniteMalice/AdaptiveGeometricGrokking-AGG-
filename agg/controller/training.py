"""Adapters for the existing training callback and copy-based consolidation trial.

This provider runs a fresh seeded continuation optimizer. It does not resume or
mutate an in-flight optimizer. Its accepted model changes only at commit.
"""

import copy
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agg.consolidation import Proposal, trial
from agg.evaluation import Constraints, Evaluation
from agg.ledger import Ledger
from agg.tasks import TaskData
from agg.telemetry.controller import Observation, Performance
from agg.training import TrainConfig, TrainingResult, evaluate, train

from .core import Controller
from .diagnosis import Diagnosis, DiagnosisType
from .evaluation import EvaluationWindow
from .policy import Action, InterventionPolicy, InterventionProposal
from .temporal import TemporalSummary


class TrainingObserver:
    """Pass as train(..., callback=observer). This callback only observes/proposes."""

    def __init__(self, controller: Controller, *, score: str = "accuracy"):
        if score not in {"accuracy", "balanced_accuracy"}:
            raise ValueError("score must be accuracy or balanced_accuracy")
        self.controller, self.score = controller, score
        self.latest: InterventionProposal | None = None

    def __call__(self, step: int, model: nn.Module, metrics: dict[str, Any]) -> None:
        self.latest = self.controller.observe(
            Observation(
                step,
                performance=Performance(
                    task_loss=metrics["train"]["loss"],
                    task_accuracy=metrics["train"][self.score],
                    validation_score=metrics["id"][self.score],
                    ood_score=metrics["ood"][self.score],
                ),
                provenance={"source": "agg.training.train callback", "score": self.score},
            )
        )


class TrainingPolicy(InterventionPolicy):
    """Explicit opt-in: a plateau proposes a modest learning-rate trial."""

    def propose(
        self,
        observation: Observation,
        temporal: dict[str, TemporalSummary],
        diagnoses: list[Diagnosis],
        *,
        sequence: int,
    ) -> InterventionProposal:
        proposal = super().propose(observation, temporal, diagnoses, sequence=sequence)
        if (
            proposal.diagnosis.kind == DiagnosisType.PLATEAU
            and proposal.diagnosis.confidence >= self.config.confidence_threshold
            and "performance.task_loss" in observation.metrics()
            and Action.ADJUST_LEARNING_RATE
            not in self.failed_actions.get(proposal.diagnosis.kind, set())
        ):
            return replace(
                proposal,
                action=Action.ADJUST_LEARNING_RATE,
                target_component="training.learning_rate",
                parameters={"fraction": self.config.max_fraction * proposal.diagnosis.confidence},
                target_metric="performance.task_loss",
                higher_is_better=False,
                expected_effect="reduce training loss while preserving held-out capability",
            )
        return proposal


class TrainingExecutor:
    supported_actions = {Action.ADJUST_LEARNING_RATE, Action.ADJUST_REGULARIZATION}

    def __init__(
        self,
        model: nn.Module,
        data: TaskData,
        config: TrainConfig,
        *,
        evaluator: Callable[[nn.Module], Evaluation],
        constraints: Constraints,
        ledger: Ledger,
        output: Path,
        max_fraction: float = 0.15,
        score: str = "accuracy",
    ):
        if not math.isfinite(max_fraction) or not 0 < max_fraction < 1:
            raise ValueError("training provider max_fraction must be in (0,1)")
        if score not in {"accuracy", "balanced_accuracy"}:
            raise ValueError("score must be accuracy or balanced_accuracy")
        self.score = score
        self.model, self.data, self.config = model, data, config
        self.evaluator, self.constraints, self.ledger = evaluator, constraints, ledger
        self.output, self.max_fraction = Path(output), max_fraction
        self.anchor = evaluator(copy.deepcopy(model))
        self._candidate: TrainingResult | None = None
        self._proposal: InterventionProposal | None = None
        self._candidate_config: TrainConfig | None = None
        self._before: tuple[nn.Module, TrainConfig] | None = None

    def stage(self, proposal: InterventionProposal) -> None:
        if self._candidate is not None:
            raise ValueError("training provider already has a candidate")
        # Snapshot this attempt before validation: failure must not undo a prior commit.
        self._before = (self.model, self.config)
        if proposal.action not in self.supported_actions:
            raise ValueError("unsupported training action")
        if proposal.baseline_provenance.get("score") != self.score:
            raise ValueError("training baseline and candidate score definitions must match")
        if not 1 <= proposal.evaluation_samples <= proposal.evaluation_window:
            raise ValueError("training window must accommodate required evaluation samples")
        fraction = proposal.parameters.get("fraction")
        if (
            set(proposal.parameters) != {"fraction"}
            or not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(fraction)
            or not 0 < abs(fraction) <= self.max_fraction
        ):
            raise ValueError("training fraction exceeds provider bounds")
        attribute = (
            "learning_rate" if proposal.action == Action.ADJUST_LEARNING_RATE else "weight_decay"
        )
        if proposal.target_component != f"training.{attribute}":
            raise ValueError("training target does not match action")
        if not proposal.id or Path(proposal.id).name != proposal.id or proposal.id in {".", ".."}:
            raise ValueError("proposal id must be a single filename component")
        self._proposal = copy.deepcopy(proposal)
        self._candidate_config = replace(
            self.config,
            steps=proposal.evaluation_window,
            eval_every=max(1, proposal.evaluation_window // proposal.evaluation_samples),
            **{attribute: getattr(self.config, attribute) * (1 + fraction)},
        )
        result: TrainingResult | None = None

        def apply(candidate: nn.Module) -> nn.Module:
            nonlocal result
            assert self._candidate_config is not None
            result = train(
                self.data, self._candidate_config, self.output / proposal.id, model=candidate
            )
            return result.model

        # train() changes process-level deterministic settings. Restore them as well
        # as the RNG state already restored by the existing trial runner.
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            checked = trial(
                self.model,
                Proposal(
                    proposal.action.value, proposal.target_component, proposal.to_dict(), apply
                ),
                self.evaluator,
                self.constraints,
                self.ledger,
                run_id="controller-training",
                step=proposal.step,
                reference=self.anchor,
            )
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if not checked.accepted or result is None:
            raise RuntimeError(f"existing model constraints rejected candidate: {checked.reason}")
        self._candidate = result

    def evaluate(self) -> EvaluationWindow:
        if self._candidate is None or self._proposal is None:
            raise ValueError("no staged training candidate")
        metrics = self._candidate.history[-1]
        score = self.score
        # Re-measure the staged model so changes outside the training history cannot
        # smuggle stale metrics through the controller's final gate.
        measured = {
            name: evaluate(self._candidate.model, split)
            for name, split in (
                ("train", self.data.train),
                ("id", self.data.id),
                ("ood", self.data.ood),
            )
        }
        return EvaluationWindow(
            self._proposal.step,
            self._proposal.step + metrics["step"],
            {
                "performance.task_loss": measured["train"]["loss"],
                "performance.task_accuracy": measured["train"][score],
                "performance.validation_score": measured["id"][score],
                "performance.ood_score": measured["ood"][score],
            },
            len(self._candidate.history) - 1,
        )

    def commit(self) -> None:
        if self._candidate is None or self._candidate_config is None:
            raise ValueError("no staged training candidate")
        self.model, self.config = self._candidate.model, self._candidate_config
        self._candidate = None

    def rollback(self) -> None:
        if self._before is not None:
            self.model, self.config = self._before
        self._candidate = None
        self._candidate_config = None
        self._proposal = None
