"""Scoped procedural knowledge, retrieval evidence and selective composition tests.

No applicable abstraction is valid. Retrieval failure is not knowledge failure;
individually successful skills are not assumed to compose successfully.
"""

import copy
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from .events import EventLog, EventType


@dataclass(frozen=True)
class Abstraction:
    id: str
    proposition: str
    block: str
    conditions: dict[str, Any] = field(default_factory=dict)
    exclusions: dict[str, Any] = field(default_factory=dict)
    source_episodes: tuple[str, ...] = ()
    supporting_episodes: tuple[str, ...] = ()
    contradicting_episodes: tuple[str, ...] = ()
    parent: str | None = None
    children: tuple[str, ...] = ()
    confidence: float = 0.0
    coverage: float | None = None
    precision: float | None = None
    transfer_utility: float | None = None
    false_applicability_rate: float | None = None
    composition_history: tuple[str, ...] = ()
    status: str = "active"
    superseded_by: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if not self.id or not self.block or not self.proposition or self.version < 1:
            raise ValueError(
                "abstraction requires identity, scope, proposition and positive version"
            )
        if self.status not in {"active", "experimental", "retired", "superseded"}:
            raise ValueError("invalid abstraction status")
        for value in (
            self.confidence,
            self.coverage,
            self.precision,
            self.false_applicability_rate,
        ):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("abstraction rates must be finite in [0,1]")
        json.dumps(asdict(self), allow_nan=False)


class AbstractionRegistry:
    def __init__(self, activation_threshold: float = 0.7, events: EventLog | None = None):
        if not math.isfinite(activation_threshold) or not 0 <= activation_threshold <= 1:
            raise ValueError("activation threshold must be in [0,1]")
        self.activation_threshold, self.events = activation_threshold, events
        self._records: dict[str, Abstraction] = {}
        self._negative_controls: dict[str, tuple[int, int]] = {}

    def add(self, record: Abstraction, *, step: int = 0) -> None:
        if record.id in self._records:
            raise ValueError("abstraction id already exists")
        if record.parent is not None and record.parent not in self._records:
            raise ValueError("parent abstraction does not exist")
        self._write(record, step)

    def _write(self, record: Abstraction, step: int) -> None:
        if self.events:
            self.events.emit(EventType.ABSTRACTION, step, asdict(record))
        self._records[record.id] = copy.deepcopy(record)

    def get(self, identity: str) -> Abstraction:
        return copy.deepcopy(self._records[identity])

    def snapshot(self) -> tuple[Abstraction, ...]:
        """Read-only routing/audit input, including inactive and conflicting records.

        Returning private copies prevents a selector from rewriting registry
        knowledge through nested conditions or provenance dictionaries.
        """
        return tuple(copy.deepcopy(self._records[key]) for key in sorted(self._records))

    def retrieve(self, block: str, context: dict[str, Any]) -> list[Abstraction]:
        # Conditions are conjunctions. Any matched exclusion vetoes application.
        return [
            copy.deepcopy(a)
            for a in self._records.values()
            if a.block == block
            and a.status == "active"
            and a.confidence >= self.activation_threshold
            and all(key in context and context[key] == value for key, value in a.conditions.items())
            and not any(
                key in context and context[key] == value for key, value in a.exclusions.items()
            )
        ]

    def retire(self, identity: str, *, superseded_by: str | None = None, step: int = 0) -> None:
        if superseded_by is not None and (
            superseded_by == identity or superseded_by not in self._records
        ):
            raise ValueError("superseding abstraction must be a different existing record")
        a = self.get(identity)
        self._write(
            replace(
                a,
                status="superseded" if superseded_by else "retired",
                superseded_by=superseded_by,
                version=a.version + 1,
            ),
            step,
        )

    def record_applicability(
        self, identity: str, *, should_apply: bool, selected: bool, step: int = 0
    ) -> None:
        a = self.get(identity)
        false, total = self._negative_controls.get(identity, (0, 0))
        if not should_apply:
            total += 1
            false += int(selected)
        self._write(
            replace(
                a, false_applicability_rate=false / total if total else None, version=a.version + 1
            ),
            step,
        )
        self._negative_controls[identity] = (false, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "agg.abstractions/1",
            "threshold": self.activation_threshold,
            "records": [asdict(a) for a in self._records.values()],
            "negative_controls": dict(self._negative_controls),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AbstractionRegistry":
        if raw.get("schema_version") != "agg.abstractions/1":
            raise ValueError("unsupported abstraction schema")
        registry = cls(raw["threshold"])
        for value in raw["records"]:
            fields = dict(value)
            for name in (
                "source_episodes",
                "supporting_episodes",
                "contradicting_episodes",
                "children",
                "composition_history",
            ):
                fields[name] = tuple(fields.get(name, ()))
            registry.add(Abstraction(**fields))
        registry._negative_controls = {k: tuple(v) for k, v in raw["negative_controls"].items()}
        return registry


@dataclass(frozen=True)
class RetrievalEvidence:
    correct_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...] = ()
    composition_success: bool | None = None
    low_separation: bool = False
    poor_specificity: bool = False
    poor_faithfulness: bool = False
    ambiguous_match: bool = False


def diagnose_retrieval(evidence: RetrievalEvidence) -> tuple[str, ...]:
    correct, retrieved = set(evidence.correct_ids), set(evidence.retrieved_ids)
    findings = []
    if not correct:
        findings.append("uncovered_query")
    if correct - retrieved:
        findings.append("miss")
    if retrieved - correct:
        findings.append("false_positive")
    relevant_invalid = set(evidence.invalid_ids) & (correct | retrieved)
    if relevant_invalid:
        findings.append("bad_knowledge")
    if correct and correct <= retrieved and not relevant_invalid:
        if evidence.composition_success is False:
            findings.append("composition_failure")
    for name in ("low_separation", "poor_specificity", "poor_faithfulness", "ambiguous_match"):
        if getattr(evidence, name):
            findings.append(name)
    return tuple(findings)


class RetrievalIndexProvider(Protocol):
    """Key revisions change index metadata, not the registry's knowledge content."""

    def retrieve(self, query: str) -> tuple[str, ...]: ...
    def revise_keys(self, abstraction_id: str, keys: tuple[str, ...]) -> None: ...


@dataclass(frozen=True)
class CompositionResult:
    ordered_ids: tuple[str, ...]
    individual_success: dict[str, bool]
    composed_success: bool
    interface_failures: tuple[str, ...] = ()
    order_dependent: bool = False
    nonlocal_dependency_failure: bool = False

    @property
    def composition_failure(self) -> bool:
        return (
            bool(self.ordered_ids)
            and all(self.individual_success.get(a, False) for a in self.ordered_ids)
            and not self.composed_success
        )


class CompositionEvaluator(Protocol):
    def evaluate(self, ordered_ids: tuple[str, ...]) -> CompositionResult: ...


def prioritize_compositions(
    dependencies: list[tuple[str, ...]], coactivation: dict[tuple[str, ...], int], *, budget: int
) -> list[tuple[str, ...]]:
    if budget < 0:
        raise ValueError("composition budget must be nonnegative")
    candidates = list(
        dict.fromkeys(
            dependencies + sorted(coactivation, key=lambda pair: (-coactivation[pair], pair))
        )
    )
    return [pair for pair in candidates if len(pair) >= 2][:budget]
