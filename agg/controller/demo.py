"""Deterministic synthetic control simulation; no replay learning is implemented here."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agg.ledger import Ledger
from agg.telemetry.controller import Continual, Observation, Performance

from .config import ControllerConfig
from .core import Controller
from .evaluation import EvaluationWindow
from .events import EventLog
from .policy import Action, InterventionProposal


def replay_observation(step: int, retention: float) -> Observation:
    return Observation(
        step,
        performance=Performance(task_accuracy=0.7, ood_score=0.8),
        continual=Continual(retained_performance=retention),
        provenance={"source": "synthetic demo; not measured training performance"},
    )


class SimulatedReplay:
    """Private scalar budget with explicitly supplied synthetic outcome evidence."""

    supported_actions = {Action.INCREASE_REPLAY, Action.INCREASE_REPLAY_DIVERSITY}

    def __init__(self, retention: float, transfer: float, budget: float = 0.2):
        self.retention, self.transfer, self.budget = retention, transfer, budget
        self.before = budget
        self.candidate: float | None = None
        self.proposal: InterventionProposal | None = None

    def stage(self, proposal: InterventionProposal) -> None:
        self.before, self.proposal = self.budget, proposal
        self.candidate = self.budget * (1 + proposal.parameters["fraction"])

    def evaluate(self) -> EvaluationWindow:
        if self.proposal is None:
            raise ValueError("simulation has no candidate")
        return EvaluationWindow(
            self.proposal.step,
            self.proposal.step + self.proposal.evaluation_window,
            {
                "continual.retained_performance": self.retention,
                "performance.ood_score": self.transfer,
            },
            2,
        )

    def commit(self) -> None:
        if self.candidate is None:
            raise ValueError("simulation has no candidate")
        self.budget = self.candidate

    def rollback(self) -> None:
        self.budget, self.candidate = self.before, None


def run_demo(output: Path) -> dict[str, object]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = ControllerConfig()
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    controller = Controller(config, EventLog(Ledger(output / "events.jsonl"), "synthetic-demo"))
    lines = ["Synthetic telemetry and budget simulation; no empirical learning claim."]
    for step in range(9):
        proposal = controller.observe(
            Observation(
                step,
                performance=Performance(task_accuracy=0.3 + 0.04 * step, ood_score=0.8),
                continual=Continual(retained_performance=0.9),
            )
        )
    lines.append(f"step {proposal.step}: {proposal.diagnosis.kind}")
    for step in range(9, 18):
        proposal = controller.observe(replay_observation(step, 0.9 - 0.025 * (step - 8)))
    lines.append(
        f"step {proposal.step}: {proposal.diagnosis.kind}; "
        f"confidence={proposal.diagnosis.confidence:.2f}"
    )
    harmful = SimulatedReplay(0.9, 0.65)
    controller.start(proposal, harmful)
    first = controller.finish()
    lines.append(
        f"step {proposal.step + config.evaluation_window}: {proposal.action}; {first.status}; "
        + "; ".join(first.reasons)
    )
    first_action = proposal.action.value
    for step in range(58, 67):
        proposal = controller.observe(replay_observation(step, 0.7 - 0.01 * (step - 58)))
    safe = SimulatedReplay(0.86, 0.8)
    controller.start(proposal, safe)
    second = controller.finish()
    lines.append(
        f"step {proposal.step + config.evaluation_window}: {proposal.action}; {second.status}"
    )
    summary = {
        "outcomes": [first.status, second.status],
        "actions": [first_action, proposal.action.value],
        "initial_budget": safe.before,
        "accepted_budget": safe.budget,
        "results": [asdict(first), asdict(second)],
        "synthetic": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "timeline.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_demo(args.output)
    print((args.output / "timeline.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
