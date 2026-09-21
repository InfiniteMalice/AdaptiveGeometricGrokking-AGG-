"""Interventions on telemetry-derived control, never on observed evidence.

Coordinate permutation shuffles each named coordinate across samples, not across
incompatible units/domains. Temporal scrambling moves complete control contexts.
Derivative features for changed coordinates are removed, not silently reused.
Hosts recompute temporal summaries if their experiment requires consistent dynamics.
"""

import random
from dataclasses import replace
from typing import Any

from agg.telemetry.controller import Observation

from .routing import AuditedEvidenceRouter, ControlContext, EvidenceValidity, RoutingDecision

STATE_COORDINATES = (
    "representation_dispersion",
    "state_displacement",
    "trajectory_cosine",
    "predictive_entropy",
)
DERIVATIVES = (".first_derivative", ".smoothed_derivative", ".second_derivative")


def _coordinates(names: tuple[str, ...]) -> None:
    if len(set(names)) != len(names) or any(name not in STATE_COORDINATES for name in names):
        raise ValueError("ablation coordinates must be unique compact-state field names")


def _changed(
    control: ControlContext, values: dict[str, float | None], label: str
) -> ControlContext:
    state = replace(control.state, **values)
    features = {
        key: value
        for key, value in control.features
        if not any(
            key == f"reasoning.{name}" or key.startswith(f"reasoning.{name}.") for name in values
        )
    }
    features.update(
        {f"reasoning.{name}": value for name, value in values.items() if value is not None}
    )
    return replace(
        control, state=state, features=tuple(sorted(features.items())), intervention=label
    )


def zero_coordinates(control: ControlContext, names: tuple[str, ...]) -> ControlContext:
    """Zero measured coordinates only; missing values remain None."""
    _coordinates(names)
    return _changed(
        control,
        {name: 0.0 if getattr(control.state, name) is not None else None for name in names},
        f"zero_coordinates:{','.join(names)}",
    )


def dynamics_disabled(control: ControlContext) -> ControlContext:
    """Remove displacement and trajectory direction (and their derivatives)."""
    return _changed(
        control, {"state_displacement": None, "trajectory_cosine": None}, "dynamics_disabled"
    )


def absolute_state_only(control: ControlContext) -> ControlContext:
    """Remove transition coordinates and all AGG temporal derivative features."""
    result = dynamics_disabled(control)
    return replace(
        result,
        features=tuple((k, v) for k, v in result.features if not k.endswith(DERIVATIVES)),
        intervention="absolute_state_only",
    )


def permute_coordinates(
    controls: tuple[ControlContext, ...], names: tuple[str, ...], *, seed: int
) -> tuple[ControlContext, ...]:
    _coordinates(names)
    rng = random.Random(seed)
    columns = {name: [getattr(control.state, name) for control in controls] for name in names}
    for column in columns.values():
        rng.shuffle(column)
    return tuple(
        _changed(
            control,
            {name: values[i] for name, values in columns.items()},
            f"permute_coordinates:{','.join(names)};seed={seed}",
        )
        for i, control in enumerate(controls)
    )


def scramble_temporal(
    controls: tuple[ControlContext, ...], *, seed: int
) -> tuple[ControlContext, ...]:
    indices = list(range(len(controls)))
    random.Random(seed).shuffle(indices)
    return tuple(
        replace(controls[i], intervention=f"temporal_scramble:source_index={i};seed={seed}")
        for i in indices
    )


def counterfactual_replay(
    harness: AuditedEvidenceRouter,
    observation: Observation,
    perturbed: ControlContext,
    *,
    block: str,
    context: dict[str, Any],
    validity: EvidenceValidity,
    profiles: dict[str, dict[str, float]],
) -> RoutingDecision:
    """Replay a fixed evidence store with changed control, logging both states.

    The caller retains the same registry version/profiles/validity for a paired
    comparison. Replaying with a changed store is a different experiment.
    """
    if not perturbed.intervention:
        raise ValueError("counterfactual replay requires a labeled control intervention")
    return harness.route(
        observation,
        block=block,
        context=context,
        validity=validity,
        profiles=profiles,
        control=perturbed,
    )
