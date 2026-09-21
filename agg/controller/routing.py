"""Opt-in activation of externally valid evidence; no knowledge mutation authority.

Reference profiles express relevance, never truth. The host supplies validity
attestations independently. Router implementations receive immutable values, not
the registry/model. Trusted Python plugins still require host process isolation
if malicious code is in scope. Every exposed selection is logged first.
"""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from agg.telemetry.controller import Observation, Reasoning

from .events import EventLog, EventType
from .memory import AbstractionRegistry
from .temporal import TemporalSummary


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _pairs(values: tuple[tuple[str, float], ...]) -> None:
    if len({k for k, _ in values}) != len(values):
        raise ValueError("feature names must be unique")
    for name, value in values:
        if not name or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("named finite numeric features required")


@dataclass(frozen=True)
class ControlContext:
    """Selected views of AGG context remain named, not an implicit learned vector.

    `intervention` labels counterfactual control; the observed state is preserved
    separately in the audit. Derivative suffixes always use Observation.step.
    """

    state: Reasoning
    features: tuple[tuple[str, float], ...]
    proxy_metrics: tuple[str, ...] = ()
    intervention: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple((key, value) for key, value in self.features))
        object.__setattr__(self, "proxy_metrics", tuple(self.proxy_metrics))
        _pairs(self.features)
        if not isinstance(self.state, Reasoning):
            raise ValueError("control state must be Reasoning")
        for name, value in self.features:
            if name.startswith("reasoning.") and name.count(".") == 1:
                if getattr(self.state, name.split(".")[1], None) != value:
                    raise ValueError("control state and reasoning features disagree")

    @classmethod
    def from_observation(
        cls, observation: Observation, temporal: dict[str, TemporalSummary] | None = None
    ) -> "ControlContext":
        features = observation.metrics()
        for name, summary in (temporal or {}).items():
            # Missing current values cannot resurrect stale temporal measurements.
            if name not in features:
                continue
            for coordinate in ("first_derivative", "smoothed_derivative", "second_derivative"):
                value = getattr(summary, coordinate)
                if value is not None:
                    features[f"{name}.{coordinate}"] = value
        return cls(
            observation.reasoning, tuple(sorted(features.items())), observation.proxy_metrics
        )


@dataclass(frozen=True)
class EvidenceValidity:
    valid_ids: tuple[str, ...] = ()
    invalid_ids: tuple[str, ...] = ()
    unavailable_ids: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        ids = self.valid_ids + self.invalid_ids + self.unavailable_ids
        if not self.source or any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("validity requires source and unique, disjoint named IDs")


@dataclass(frozen=True)
class EvidenceCandidate:
    id: str
    profile: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence ID must be named")
        _pairs(self.profile)


@dataclass(frozen=True)
class FeatureScale:
    name: str
    scale: float

    def __post_init__(self) -> None:
        if (
            not self.name
            or type(self.scale) not in (int, float)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("feature scale must be explicitly positive and finite")


@dataclass(frozen=True)
class RoutingConfig:
    max_active: int = 1
    features: tuple[FeatureScale, ...] = (FeatureScale("reasoning.predictive_entropy", 1.0),)
    minimum_score: float = 0.0
    mode: str = "state_conditioned"
    seed: int = 0
    fixed_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.max_active) is not int or self.max_active < 0 or type(self.seed) is not int:
            raise ValueError("nonnegative integer max_active and integer seed required")
        if (
            type(self.minimum_score) not in (int, float)
            or not math.isfinite(self.minimum_score)
            or not 0 <= self.minimum_score <= 1
        ):
            raise ValueError("minimum_score must be in [0,1]")
        if self.mode not in {
            "state_conditioned",
            "random",
            "state_independent",
            "fixed",
            "full_history",
        }:
            raise ValueError("unknown routing mode")
        if not self.features or len({f.name for f in self.features}) != len(self.features):
            raise ValueError("routing requires uniquely named feature scales")
        if len(set(self.fixed_ids)) != len(self.fixed_ids) or any(not x for x in self.fixed_ids):
            raise ValueError("fixed IDs must be unique and named")


@dataclass(frozen=True)
class RoutingSelection:
    selected_ids: tuple[str, ...]
    scores: tuple[tuple[str, float | None], ...]
    rationale: tuple[tuple[str, str], ...]


class EvidenceRouter(Protocol):
    """Interface for rules, learned selectors, JEV or experimental neural routers."""

    def configuration(self) -> dict[str, Any]: ...
    def select(
        self, control: ControlContext, eligible: tuple[EvidenceCandidate, ...]
    ) -> RoutingSelection: ...


class ReferenceRouter:
    """Deterministic scaled profile similarity, not the paper's learned controller.

    Score = 1 / (1 + Euclidean distance in explicitly scaled selected coordinates).
    Missing required coordinates abstain, rather than shrinking the dimension or
    inventing a zero. Ties use stable evidence IDs. The random baseline is locally
    seeded per call, so counterfactual state changes hold its draw fixed.
    """

    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or RoutingConfig()

    def configuration(self) -> dict[str, Any]:
        return {"implementation": "agg.reference-router/1", **asdict(self.config)}

    def select(
        self, control: ControlContext, eligible: tuple[EvidenceCandidate, ...]
    ) -> RoutingSelection:
        c = self.config
        values = dict(control.features)
        ordered = sorted(eligible, key=lambda e: e.id)
        scores: dict[str, float | None] = {}
        reasons: dict[str, str] = {}
        for candidate in ordered:
            profile = dict(candidate.profile)
            if c.mode != "state_conditioned":
                scores[candidate.id] = None
                reasons[candidate.id] = f"{c.mode} ablation; no state similarity score"
            elif any(f.name not in values or f.name not in profile for f in c.features):
                scores[candidate.id] = None
                reasons[candidate.id] = "missing required control/profile coordinate; abstain"
            else:
                distances = [(values[f.name] - profile[f.name]) / f.scale for f in c.features]
                distance = math.hypot(*distances)
                if not math.isfinite(distance):
                    raise ValueError("scaled profile distance exceeds finite range")
                scores[candidate.id] = 1 / (1 + distance)
                reasons[candidate.id] = "scaled profile similarity; no validity inference"
        ids = [item.id for item in ordered]
        if c.mode == "state_conditioned":
            ranked = [
                (key, score)
                for key, score in scores.items()
                if score is not None and score >= c.minimum_score
            ]
            ids = [key for key, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))]
        elif c.mode == "random":
            random.Random(c.seed).shuffle(ids)
        elif c.mode == "fixed":
            ids = [key for key in c.fixed_ids if key in ids]
        # Full history is an explicit baseline: it intentionally ignores max_active.
        selected = ids if c.mode == "full_history" else ids[: c.max_active]
        return RoutingSelection(tuple(selected), tuple(scores.items()), tuple(reasons.items()))


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable canonical audit snapshot; decoded views cannot rewrite the record."""

    id: str
    step: int
    component: str
    eligible_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    audit_json: str

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(self.audit_json))


class AuditedEvidenceRouter:
    """Host boundary: attest validity, validate selection, log, then expose IDs.

    Calls do not activate anything inside the model. Hosts use selected IDs as
    ephemeral context. Store original record versions in a host evidence store;
    the audit's version/hash identifies but does not retain their payloads.
    """

    def __init__(self, registry: AbstractionRegistry, router: EvidenceRouter, events: EventLog):
        self.registry, self.router, self.events = registry, router, events

    def route(
        self,
        observation: Observation,
        *,
        block: str,
        context: dict[str, Any],
        validity: EvidenceValidity,
        profiles: dict[str, dict[str, float]],
        temporal: dict[str, TemporalSummary] | None = None,
        control: ControlContext | None = None,
    ) -> RoutingDecision:
        observed_control = ControlContext.from_observation(observation, temporal)
        selected_control = control or observed_control
        if selected_control != observed_control and not selected_control.intervention:
            raise ValueError("changed control requires an explicit intervention label")
        records = self.registry.snapshot()
        known = {r.id for r in records}
        applicable = {r.id for r in self.registry.retrieve(block, context)}
        valid, invalid = set(validity.valid_ids), set(validity.invalid_ids)
        unavailable = set(validity.unavailable_ids) | ((valid | invalid | set(profiles)) - known)
        eligible_ids = tuple(sorted(applicable & valid - unavailable - invalid))
        # Validate all profiles, including currently ineligible evidence, before audit.
        candidates = {
            key: EvidenceCandidate(key, tuple(sorted(profile.items())))
            for key, profile in profiles.items()
        }
        eligible = tuple(candidates.get(key, EvidenceCandidate(key)) for key in eligible_ids)
        payload: dict[str, Any] = {
            "schema_version": "agg.evidence-routing/1",
            "step": observation.step,
            "component": observation.component,
            "observation": observation.to_dict(),
            "control": asdict(selected_control),
            "eligible_ids": eligible_ids,
            "invalid_ids": tuple(sorted(invalid)),
            "unknown_ids": tuple(sorted(known - valid - invalid - unavailable)),
            "unavailable_ids": tuple(sorted(unavailable)),
            "ineligible_ids": tuple(sorted(known - applicable)),
            "evidence_refs": [(r.id, r.version, _hash(asdict(r))) for r in records],
            "profiles": profiles,
            "validity": asdict(validity),
            "configuration": self.router.configuration(),
            "retrieval": {
                "block": block,
                "context": context,
                "activation_threshold": self.registry.activation_threshold,
            },
            "provenance": observation.provenance,
            "temporal": {k: asdict(v) for k, v in (temporal or {}).items()},
        }
        # Freeze host inputs/config before invoking any plugin.
        payload = json.loads(_json(payload))
        selection = None
        try:
            selection = self.router.select(selected_control, eligible)
            self._validate(selection, set(eligible_ids))
        except Exception as exc:
            # Invalid output may contain NaN, non-JSON values or malformed types.
            # Preserve its representation without poisoning the rejection audit.
            payload["invalid_selection_repr"] = repr(selection)
            payload.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            self.events.emit(EventType.EVIDENCE_ROUTING, observation.step, payload)
            raise
        dropped = tuple(sorted(set(eligible_ids) - set(selection.selected_ids)))
        payload.update(
            status="selected",
            selected_ids=selection.selected_ids,
            dropped_ids=dropped,
            scores=selection.scores,
            rationale=selection.rationale,
        )
        identity = _hash(payload)
        payload["id"] = identity
        encoded = _json(payload)
        self.events.emit(EventType.EVIDENCE_ROUTING, observation.step, json.loads(encoded))
        return RoutingDecision(
            identity,
            observation.step,
            observation.component,
            eligible_ids,
            selection.selected_ids,
            dropped,
            encoded,
        )

    @staticmethod
    def _validate(selection: RoutingSelection, eligible: set[str]) -> None:
        # A duck-typed property could change IDs between validation and logging.
        # Require the plain value type, rather than rereading executable getters.
        if type(selection) is not RoutingSelection:
            raise ValueError("router must return a plain RoutingSelection value")
        selected = selection.selected_ids
        if type(selected) is not tuple or any(type(key) is not str for key in selected):
            raise ValueError("router must return an immutable tuple of evidence IDs")
        if len(selected) != len(set(selected)) or not set(selected) <= eligible:
            raise ValueError("router may only select unique eligible IDs")
        for entries in (selection.scores, selection.rationale):
            if type(entries) is not tuple or any(type(entry) is not tuple for entry in entries):
                raise ValueError("router scores and rationale must be immutable tuples")
            if len(entries) != len(eligible) or {k for k, _ in entries} != eligible:
                raise ValueError("every eligible ID requires score and selection rationale")
        for _, score in selection.scores:
            if score is not None and (type(score) not in (float, int) or not math.isfinite(score)):
                raise ValueError("routing scores must be finite or None")
        if any(not isinstance(reason, str) or not reason for _, reason in selection.rationale):
            raise ValueError("every eligible ID requires a named rationale")

    def record_downstream(
        self, decision: RoutingDecision, outcome: dict[str, Any], *, proposal_id: str | None = None
    ) -> None:
        """Append a linkage; never rewrite the earlier evidence decision."""
        self.events.emit(
            EventType.ROUTING_OUTCOME,
            decision.step,
            {
                "routing_id": decision.id,
                "component": decision.component,
                "outcome": json.loads(_json(outcome)),
            },
            proposal_id,
        )
