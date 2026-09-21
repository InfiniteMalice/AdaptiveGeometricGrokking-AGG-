"""Structured controller events share the existing append-only Ledger transport."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from agg.ledger import Ledger


class EventType(StrEnum):
    TELEMETRY = "TelemetrySnapshot"
    DIAGNOSIS = "DiagnosisEvent"
    PROPOSAL = "InterventionProposal"
    EXECUTION = "InterventionExecution"
    OUTCOME = "InterventionOutcome"
    ROLLBACK = "RollbackEvent"
    ABSTRACTION = "AbstractionEvent"
    CURRICULUM = "CurriculumEvent"
    REGRESSION = "RegressionEvent"
    EVIDENCE_ROUTING = "EvidenceRoutingEvent"
    ROUTING_OUTCOME = "EvidenceRoutingOutcome"


@dataclass(frozen=True)
class Event:
    kind: EventType
    step: int
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    proposal_id: str | None = None
    schema_version: str = "agg.events/1"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Event":
        fields = dict(value)
        if fields.get("schema_version") != "agg.events/1":
            raise ValueError("unsupported event schema")
        fields["kind"] = EventType(fields["kind"])
        return cls(**fields)


class EventLog:
    def __init__(self, ledger: Ledger, run_id: str):
        if not run_id:
            raise ValueError("run_id must be named")
        self.ledger, self.run_id = ledger, run_id

    def emit(
        self, kind: EventType, step: int, payload: dict[str, Any], proposal_id: str | None = None
    ) -> None:
        self.ledger.append(asdict(Event(kind, step, self.run_id, payload, proposal_id)))
