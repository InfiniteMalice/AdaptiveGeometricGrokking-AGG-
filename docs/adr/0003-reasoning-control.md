# ADR 0003: Separate reasoning state, evidence activation and authority

Status: accepted for implementation under the supplied task specification.

Use a dedicated optional `Reasoning` category and a versioned Observation loader.
Keep AGG context in typed categories and temporal summaries; do not concatenate
all telemetry into a learned vector. Existing registry retrieval determines
applicability; external attestations determine validity; the router determines
activation among valid applicable records. None implies another.

Routing accepts immutable value objects and returns IDs. It has no model,
registry mutation, evaluator or reward authority. A host harness appends complete
decision records through EventLog before exposing selections. This preserves
non-selected evidence in audit without copying large payloads into every event.

The inner model/depth loop, reasoning/evidence loop, and harness/audit loop remain
separate. The optional compute policy implements the existing protocol and uses
the existing proposal/stage/evaluate/commit-or-rollback path. Unsupported provider
actions never become successes. See the [guide](../state-of-thought.md) for units
and limits, and the [specification](../specs/state-of-thought.md) for acceptance scope.
