"""Versioned, nonmonotonic diagnostic hypotheses. This module grants no rewards."""

import math
from dataclasses import asdict, dataclass
from typing import Any

from agg.probes import retrieval_crossing


@dataclass(frozen=True)
class Measurement:
    value: float | None
    source: str
    verified: bool
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.source
            or type(self.verified) is not bool
            or (
                self.value is not None
                and (type(self.value) not in (int, float) or not math.isfinite(self.value))
            )
        ):
            raise ValueError("measurement needs provenance, verification state and finite value")
        if self.value is None and not self.missing_reason:
            raise ValueError("missing measurements require a reason")


# Closed, explicit diagnostic families, not an optimizer over arbitrary telemetry.
DEFINITIONS: dict[str, dict[str, tuple[float, float]]] = {
    "memorization_gap": {"train_accuracy": (0.9, 1.0), "ood_accuracy": (0.0, 0.6)},
    "retrieval_reuse": {
        "correct_invariance": (0.8, 1.0),
        "required_update": (0.8, 1.0),
        "relevant_flip": (0.8, 1.0),
        "irrelevant_flip": (0.0, 0.1),
    },
    "structural_generalization": {
        "ood_accuracy": (0.8, 1.0),
        "decisive_joint": (0.8, 1.0),
        "correct_invariance": (0.8, 1.0),
    },
    "compression_robustness": {
        "parameters_reduced": (1.0, 1.0),
        "baseline_ood": (0.8, 1.0),
        "compression_id_gain": (-0.01, 1.0),
        "compression_ood_gain": (-0.01, 1.0),
    },
}


def milestone_diagnostics(points: list[dict[str, Any]], *, sustain: int = 3) -> dict[str, Any]:
    steps = [p["step"] for p in points]
    if (
        not steps
        or type(sustain) is not int
        or sustain < 1
        or any(type(s) is not int or s < 0 for s in steps)
        or any(b <= a for a, b in zip(steps, steps[1:], strict=False))
    ):
        raise ValueError("milestones need increasing nonnegative steps and positive sustain")
    result: dict[str, Any] = {}
    for name, requirements in DEFINITIONS.items():
        history: list[dict[str, Any]] = []
        for point in points:
            current = {}
            sources = {}
            missing, conflicts, unverified = [], [], []
            for metric in requirements:
                evidence = point["evidence"].get(metric, [])
                if any(not isinstance(e, Measurement) for e in evidence):
                    raise ValueError("milestone evidence must contain Measurement values")
                sources[metric] = [asdict(e) for e in evidence]
                usable = [e.value for e in evidence if e.verified and e.value is not None]
                current[metric] = usable[0] if usable else None
                if not evidence or all(e.value is None for e in evidence):
                    missing.append(metric)
                elif any(not e.verified for e in evidence):
                    unverified.append(metric)
                if usable and max(usable) - min(usable) > 1e-12:
                    conflicts.append(metric)
                    current[metric] = None
            state = (
                "conflicting"
                if conflicts
                else "missing"
                if missing
                else "unverified"
                if unverified
                else "observed"
            )
            satisfied = (
                all(
                    value is not None
                    and requirements[metric][0] <= value <= requirements[metric][1]
                    for metric, value in current.items()
                )
                if state == "observed"
                else None
            )
            history.append(
                {
                    "step": point["step"],
                    "measurements": current,
                    "previous": history[-1]["measurements"] if history else None,
                    "evidence": sources,
                    "verification_state": state,
                    "satisfied": satisfied,
                    "missing": missing,
                    "conflicts": conflicts,
                    "unverified": unverified,
                }
            )
        crossing = retrieval_crossing(
            steps,
            [0.0] * len(steps),
            [
                float(r["satisfied"]) if r["satisfied"] is not None else float("nan")
                for r in history
            ],
            sustain=sustain,
        )
        confirmed = crossing.confirmation_step
        crossing_record = asdict(crossing)
        if crossing.stable_crossing is not None:
            # Unknown evidence breaks sustained support but cannot prove absence.
            absent = [
                r["step"]
                for r in history
                if r["step"] < crossing.stable_crossing and r["satisfied"] is False
            ]
            crossing_record["interval"] = [absent[-1] if absent else None, crossing.stable_crossing]
        result[name] = {
            "definition_version": "agg.milestone-definitions/1",
            "history": history,
            "crossing": crossing_record,
            "sustain": sustain,
            "contradictions_after_confirmation": [
                r["step"]
                for r in history
                if confirmed is not None
                and r["step"] > confirmed
                and (r["satisfied"] is False or r["verification_state"] == "conflicting")
            ],
            "missing_after_confirmation": [
                r["step"]
                for r in history
                if confirmed is not None
                and r["step"] > confirmed
                and r["verification_state"] in {"missing", "unverified"}
            ],
        }
    return {
        "schema_version": "agg.milestones/1",
        "definitions": {
            name: {metric: list(bounds) for metric, bounds in values.items()}
            for name, values in DEFINITIONS.items()
        },
        "milestones": result,
        "reward_authorized": False,
        "scientific_validity": "unestablished diagnostic hypotheses",
    }
