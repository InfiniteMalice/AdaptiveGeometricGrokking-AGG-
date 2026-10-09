"""Measured extension of the real, bounded continuation-training provider."""

import json
import time
from dataclasses import replace
from typing import Any

from .evaluation import EvaluationWindow
from .policy import InterventionProposal
from .resources import measure_resources
from .training import TrainingExecutor


class MeasuredTrainingExecutor(TrainingExecutor):
    def __init__(self, *args: Any, measurement_repeats: int = 9, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if type(measurement_repeats) is not int or measurement_repeats < 3:
            raise ValueError("at least three measurement repeats required")
        self.measurement_repeats = measurement_repeats
        self.measurement: dict[str, Any] | None = None
        self.last_budget: dict[str, Any] = {}

    def stage(self, proposal: InterventionProposal) -> None:
        self.measurement = None
        start = time.perf_counter()
        self.last_budget = {
            "planned_updates": proposal.evaluation_window,
            "completed_updates_lower_bound": 0,
            "executed_updates": None,
            "tokens_per_update": min(self.config.batch_size, len(self.data.train.y))
            * self.data.train.x.shape[1],
            "missing_reason": "execution started; final update count unverified",
            "accounting_error": None,
        }
        succeeded = False
        try:
            super().stage(proposal)
            succeeded = True
        finally:
            # Preserve complete prefix records even after a partial write. Accounting
            # must neither hide the execution error nor relabel unknown work as zero.
            path = self.output / proposal.id / "metrics.jsonl"
            observed = 0
            try:
                if proposal.id and path.parent.parent == self.output and path.exists():
                    for line in path.read_text().splitlines():
                        row = json.loads(line)
                        step = row["step"]
                        if type(step) is not int or not 0 <= step <= proposal.evaluation_window:
                            raise ValueError("invalid completed step in training log")
                        observed = max(observed, step)
            except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                self.last_budget["accounting_error"] = f"{type(exc).__name__}: {exc}"
            if succeeded:
                observed = proposal.evaluation_window
            complete = observed == proposal.evaluation_window
            self.last_budget.update(
                {
                    "completed_updates_lower_bound": observed,
                    "executed_updates": observed if complete else None,
                    "wall_seconds": time.perf_counter() - start,
                    "missing_reason": None
                    if complete
                    else "partial execution: final update count unverified",
                }
            )

    def evaluate(self) -> EvaluationWindow:
        window = super().evaluate()
        assert self._candidate is not None and self._candidate_config is not None
        self.measurement = measure_resources(
            self._candidate.model,
            self.data.id.x[: self.config.batch_size],
            repeats=self.measurement_repeats,
        )
        updates = self._candidate.history[-1]["step"]
        self.measurement["metrics"].update(
            {
                "resources.training_updates": updates,
                "resources.training_tokens": updates
                * min(self._candidate_config.batch_size, len(self.data.train.y))
                * self.data.train.x.shape[1],
            }
        )
        return replace(window, metrics={**window.metrics, **self.measurement["metrics"]})
