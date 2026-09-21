"""Synthetic routing ablation; no LLM reasoning or learning improvement claim.

The toy oracle says low/high-entropy evidence is relevant according to the next
synthetic entropy value. Forecasts consume current state, then AGG's fitted slope,
then fitted acceleration. Abrupt regime shifts deliberately violate the smooth
forecast assumption. More dynamics need not win. No model/provider is executed.
"""

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from agg.ledger import Ledger
from agg.telemetry.controller import Observation, Reasoning

from .events import EventLog
from .memory import Abstraction, AbstractionRegistry
from .routing import (
    AuditedEvidenceRouter,
    ControlContext,
    EvidenceValidity,
    FeatureScale,
    ReferenceRouter,
    RoutingConfig,
)
from .routing_diagnostics import diagnose_routing
from .temporal import TemporalTelemetry


def synthetic_entropy(step: int) -> float:
    """Three deterministic regimes, with deliberate trajectory shifts at 20/40."""
    segment, local = divmod(step, 20)
    phase = (0.0, math.pi, math.pi / 2)[min(segment, 2)]
    return 2 + 0.85 * math.sin(local * math.pi / 10 + phase)


def _variant(output: Path, name: str, order: int | None) -> dict[str, Any]:
    registry = AbstractionRegistry()
    for key in ("low", "high"):
        registry.add(
            Abstraction(
                key,
                f"synthetic {key} entropy support",
                "synthetic",
                confidence=1,
                provenance={"synthetic": True},
            )
        )
    config = RoutingConfig(
        features=(FeatureScale("forecast.entropy", 1.0),),
        mode="state_independent" if order is None else "state_conditioned",
    )
    harness = AuditedEvidenceRouter(
        registry,
        ReferenceRouter(config),
        EventLog(Ledger(output / f"{name}.jsonl"), f"synthetic-{name}"),
    )
    temporal = TemporalTelemetry()
    previous = None
    correct_count = unnecessary = compute = 0
    turnovers: list[float] = []
    trace: list[dict[str, Any]] = []
    for step in range(60):
        entropy = synthetic_entropy(step)
        observation = Observation(
            step,
            reasoning=Reasoning(predictive_entropy=entropy),
            provenance={"synthetic": True, "source": "deterministic trigonometric fixture"},
        )
        summaries = temporal.update(observation)
        summary = summaries["reasoning.predictive_entropy"]
        forecast = entropy
        used = 1
        if order is not None and order >= 1 and summary.smoothed_derivative is not None:
            forecast += summary.smoothed_derivative
            used += 1
        if order is not None and order >= 2 and summary.second_derivative is not None:
            forecast += 0.5 * summary.second_derivative
            used += 1
        # Explicit host proxy in its own namespace; it is not observed entropy.
        control = ControlContext.from_observation(observation, summaries)
        control = replace(
            control,
            features=tuple(sorted((*control.features, ("forecast.entropy", forecast)))),
            proxy_metrics=(*control.proxy_metrics, "forecast.entropy"),
            intervention=f"synthetic forecast order={order}; step horizon=1",
        )
        decision = harness.route(
            observation,
            block="synthetic",
            context={},
            validity=EvidenceValidity(("low", "high"), source="synthetic fixture truth"),
            profiles={"low": {"forecast.entropy": 1}, "high": {"forecast.entropy": 3}},
            temporal=summaries,
            control=control,
        )
        oracle = "high" if synthetic_entropy(step + 1) >= 2 else "low"
        correct = decision.selected_ids == (oracle,)
        correct_count += int(correct)
        diagnostic = diagnose_routing(decision, previous)
        if diagnostic.evidence_turnover is not None:
            turnovers.append(diagnostic.evidence_turnover)
        if previous is not None and decision.selected_ids != previous.selected_ids:
            unnecessary += int(trace[-1]["oracle"] == oracle)
        # Explicit cost proxy: coordinate reads + activated evidence items.
        compute += (0 if order is None else used) + len(decision.selected_ids)
        trace.append(
            {
                "step": step,
                "forecast_entropy": forecast,
                "oracle": oracle,
                "selected_ids": list(decision.selected_ids),
                "correct": correct,
            }
        )
        previous = decision
    recovery = []
    for change in (20, 40):
        recovery.append(
            next(
                (
                    t - change
                    for t in range(change, change + 19)
                    if trace[t]["correct"] and trace[t + 1]["correct"]
                ),
                None,
            )
        )
    return {
        "configuration": asdict(config),
        "routing_precision": correct_count / len(trace),
        "routing_stability": 1 - sum(turnovers) / len(turnovers),
        "unnecessary_evidence_changes": unnecessary,
        "compute_usage_proxy": compute,
        "recovery_steps": recovery,
        "protected_regressions": None,
        "unnecessary_interventions": None,
        "trace": trace,
    }


def run_demo(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "synthetic": True,
        "limits": "Routing proxies only; no model, executed intervention, or protected evaluation.",
        "variants": {
            name: _variant(output, name, order)
            for name, order in (
                ("absolute", 0),
                ("first_order", 1),
                ("second_order", 2),
                ("independent", None),
            )
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_demo(args.output)
    print("Synthetic routing demonstration; not scientific validation.")
    for name, metrics in result["variants"].items():
        print(
            f"{name}: precision={metrics['routing_precision']:.3f}, "
            f"stability={metrics['routing_stability']:.3f}, "
            f"compute_proxy={metrics['compute_usage_proxy']}, "
            f"recovery={metrics['recovery_steps']}"
        )


if __name__ == "__main__":
    main()
