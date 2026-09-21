# ADR 0002: Separate temporal control from model consolidation

Status: accepted for this implementation of the user-specified controller.

The existing periodic consolidation trigger cannot diagnose temporal changes.
Replacing it would alter historical experiments. Extending its model-only
proposal with replay, retrieval and curriculum mutation would couple unrelated
execution environments.

Add a separate opt-in controller that consumes typed observations. Reuse legacy
telemetry through explicit conversion, the existing JSONL ledger for events and
the existing model trial for model candidates. Temporal features do not reuse
depth differences: their independent variable is training step.

Providers stage candidates privately and expose evaluation and commit/rollback.
Rollback snapshots remain available until the controller records the committed
outcome. It then calls the provider's `finalize` method to release transaction
state, preventing a later rollback from undoing an already accepted transaction.
Custom providers must implement this method; cleanup failure blocks the controller
and propagates to the host without undoing the audited commit.
The controller owns protected metrics and fixed reference anchors. Model trials
also enforce the original ID/OOD/mechanism constraints. The initial training
provider runs a fresh seeded optimizer continuation, preserving accepted objects.
No online optimizer mutation is introduced.

Consequences: old experiments remain reproducible; absent mechanisms require
providers; confidence is explicitly heuristic; malicious in-process providers
still require external process isolation. No new reward terms are added.
