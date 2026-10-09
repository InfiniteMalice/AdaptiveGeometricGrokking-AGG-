import copy
import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from agg.evaluation import Constraints, Evaluation
from agg.ledger import Ledger


@dataclass
class Proposal:
    kind: str
    component: str
    state: dict[str, Any]
    apply: Callable[[nn.Module], nn.Module]


@dataclass
class TrialResult:
    model: nn.Module
    accepted: bool
    reason: str
    before: Evaluation
    after: Evaluation | None


class TrialRecorder(Protocol):
    def begin(self, model: nn.Module, proposal: Proposal, *, run_id: str, step: int) -> int: ...
    def finish(self, attempt: int, record: dict[str, Any], candidate: nn.Module | None) -> None: ...
    def fail(self, attempt: int, error: Exception) -> None: ...


class InterventionPolicy:
    """Periodic explicit proposal trigger; receives observations, never a model."""

    def __init__(self, interval: int = 100) -> None:
        if interval < 1:
            raise ValueError("interval must be positive")
        self.interval = interval

    def propose(self, step: int, telemetry: dict[str, Any]) -> dict[str, Any] | None:
        if step > 0 and step % self.interval == 0:
            return {"step": step, "reason": "periodic", "signals": telemetry}
        return None


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    # Tensor equality alone is insufficient: fake activation precision and
    # geometry routing are behavioral configuration stored outside state_dict.
    for name, module in model.named_modules():
        metadata: dict[str, Any] = {
            "name": name,
            "type": type(module).__module__ + "." + type(module).__qualname__,
        }
        for attribute in (
            "precision",
            "geometry",
            "dimension",
            "width",
            "num_heads",
            "norm_first",
            "batch_first",
            "eps",
            "p",
            "_forced",
        ):
            value = getattr(module, attribute, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[attribute] = value
        digest.update(json.dumps(metadata, sort_keys=True).encode())
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    return value


def trial(
    model: nn.Module,
    proposal: Proposal,
    evaluator: Callable[[nn.Module], Evaluation],
    constraints: Constraints,
    ledger: Ledger,
    *,
    run_id: str,
    step: int,
    reference: Evaluation | None = None,
    telemetry_before: dict[str, Any] | None = None,
    observer: Callable[[nn.Module], dict[str, Any]] | None = None,
    recorder: TrialRecorder | None = None,
    selection_gain_floor: float | None = None,
) -> TrialResult:
    """Evaluate a copy. Rejection never changes the original object or optimizer.

    Pass a fixed reference across a sequential sweep to prevent tolerance drift.
    Evaluators should be deterministic and restore train/eval mode.
    """
    if selection_gain_floor is not None and (
        type(selection_gain_floor) not in (int, float)
        or not math.isfinite(selection_gain_floor)
        or selection_gain_floor < 0
    ):
        raise ValueError("selection gain floor must be a finite nonnegative number")
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    digest_before = state_hash(model)
    after = None
    candidate = None
    accepted = False
    observed_before = copy.deepcopy(telemetry_before)
    observed_after = None
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    attempt = None
    try:
        if recorder is not None:
            attempt = recorder.begin(model, proposal, run_id=run_id, step=step)
        # Reference failures are genuine experiment failures, not candidate rejections.
        # A fresh local measurement is needed even with a fixed acceptance anchor.
        before = evaluator(copy.deepcopy(model))
        anchor = reference if reference is not None else before
        try:
            if observer is not None:
                observed_before = copy.deepcopy(observer(copy.deepcopy(model)))
            candidate = proposal.apply(copy.deepcopy(model))
            after = evaluator(candidate)
            if observer is not None:
                observed_after = copy.deepcopy(observer(copy.deepcopy(candidate)))
            reasons = constraints.violations(anchor, after)
            if selection_gain_floor is not None:
                for name in ("id_accuracy", "ood_accuracy"):
                    gain = getattr(after, name) - getattr(before, name)
                    if not math.isfinite(gain) or gain <= selection_gain_floor:
                        reasons.append(f"selection gain does not exceed floor: {name}")
            accepted = not reasons
            reason = "; ".join(reasons) if reasons else "all configured constraints satisfied"
        except Exception as exc:
            reason = f"candidate error: {type(exc).__name__}: {exc}"
    except Exception as exc:
        if recorder is not None and attempt is not None:
            recorder.fail(attempt, exc)
        raise
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    retained = candidate if accepted and candidate is not None else model
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "checkpoint": step,
        "component": proposal.component,
        "intervention_type": proposal.kind,
        "telemetry_before": observed_before,
        "telemetry_after": observed_after,
        "proposed_state": proposal.state,
        "accepted": accepted,
        "reason": reason,
        "evaluation_before": asdict(before),
        "evaluation_reference": asdict(anchor),
        "evaluation_after": asdict(after) if after else None,
        "state_before": digest_before,
        "state_retained": state_hash(retained),
        "state_candidate": state_hash(candidate) if candidate is not None else None,
        "id_delta": after.id_accuracy - before.id_accuracy if after else None,
        "ood_delta": after.ood_accuracy - before.ood_accuracy if after else None,
        "mechanism_score": after.mechanism_score if after else None,
        "cost_delta": after.cost - before.cost if after else None,
        "gate_strength": proposal.state.get("gate"),
        "memorization_delta": None,
        "retrieval_delta": None,
        "generalization_delta": None,
        "attribution_delta": None,
        "geometry_topology_delta": None,
        "description_length_proxy_delta": None,
        "parameter_delta": None,
        "sparsity_delta": None,
        "precision_delta": None,
        "zero_density": None,
        "storage_format": None,
        "serialized_byte_delta": None,
        "hardware": None,
        "latency_delta": None,
        "execution_evidence": None,
        "unavailable_reason": "null quantities were not measured by this evaluator",
    }
    record.update(_observation_deltas(observed_before, observed_after, proposal))
    ledger.append(_finite_json(record))
    if recorder is not None and attempt is not None:
        recorder.finish(attempt, _finite_json(record), candidate)
    return TrialResult(retained, accepted, reason, before, after)


def _numeric_delta(before: Any, after: Any) -> Any:
    """Only subtract corresponding finite measured numbers; labels stay in raw data."""
    if (
        type(before) in (int, float)
        and type(after) in (int, float)
        and math.isfinite(before)
        and math.isfinite(after)
    ):
        return after - before
    if isinstance(before, dict) and isinstance(after, dict):
        result = {key: _numeric_delta(before[key], after[key]) for key in before.keys() & after}
        return {key: value for key, value in result.items() if value is not None} or None
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [_numeric_delta(b, a) for b, a in zip(before, after, strict=True)]
    return None


def _category(observation: dict[str, Any] | None, category: str) -> Any:
    return observation.get(category) if observation is not None else None


def _sparsity(observation: Any) -> float | None:
    if not isinstance(observation, dict):
        return None
    parameters, active = observation.get("parameters"), observation.get("active_parameters")
    if (
        isinstance(parameters, (int, float))
        and isinstance(active, (int, float))
        and not isinstance(parameters, bool)
        and not isinstance(active, bool)
        and math.isfinite(parameters)
        and math.isfinite(active)
        and 0 <= active <= parameters
        and parameters > 0
    ):
        return 1 - active / parameters
    return None


def _observation_deltas(
    before: dict[str, Any] | None, after: dict[str, Any] | None, proposal: Proposal
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for category in ("memorization", "retrieval", "generalization"):
        change = _numeric_delta(_category(before, category), _category(after, category))
        deltas[f"{category}_delta"] = change.get("score") if isinstance(change, dict) else None
    for field, category in (
        ("attribution_delta", "attribution"),
        ("description_length_proxy_delta", "complexity"),
        ("dimension_delta", "dimension"),
    ):
        deltas[field] = _numeric_delta(_category(before, category), _category(after, category))
    geometry = {
        category: _numeric_delta(_category(before, category), _category(after, category))
        for category in ("geometry", "topology")
    }
    deltas["geometry_topology_delta"] = {
        key: value for key, value in geometry.items() if value is not None
    } or None
    sparsity_before, sparsity_after = _category(before, "sparsity"), _category(after, "sparsity")
    counts = _numeric_delta(sparsity_before, sparsity_after)
    deltas["parameter_delta"] = counts.get("parameters") if isinstance(counts, dict) else None
    deltas["active_parameter_delta"] = (
        counts.get("active_parameters") if isinstance(counts, dict) else None
    )
    deltas["sparsity_delta"] = _numeric_delta(_sparsity(sparsity_before), _sparsity(sparsity_after))
    precision = {
        key: proposal.state[key]
        for key in ("weight", "activation", "compute", "accumulator")
        if key in proposal.state
    }
    deltas["precision_delta"] = (
        {
            "before": _category(before, "precision"),
            "after": precision,
            "source": "proposed logical precision; candidate acceptance recorded separately",
        }
        if precision
        else None
    )
    return deltas
