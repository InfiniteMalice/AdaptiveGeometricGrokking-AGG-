# Resource-constrained controller plan

Spec: PR-5 of the approved research integration contract. Existing controller
anchors, target improvement, copy trials and rollback remain authoritative.

## Design

Add an optional typed ResourceProfile to ControllerConfig. Absolute capability
floors and upper resource limits are a conjunction, with named margins and
explicit missing evidence. An absent profile preserves legacy behavior. Measure
CPU forward wall latency and process CPU time on declared batches with repeated
samples; count actual unique tensor storage and parameters. Model tensor bytes
are not peak runtime memory; peak memory, energy and FLOPs remain unavailable.
A configured unavailable requirement fails closed.

A measured TrainingExecutor extends the existing LR/regularization provider,
retaining its original capability/mechanism anchor and transaction semantics.
Add optional existing TrialRecorder integration so rejected attempts also persist.
Training budgets use observed completed loop steps; log partial failures separately.
Unsupported depth/routing/precision/reasoning actions stay unsupported.

The benchmark compares fixed continuation, the existing controller, a joint
resource controller, and a controller guided by earlier selection-only experience.
All arms start from the same per-cell initial model, have the same intervention
step cap, and are remeasured on target hardware. Successful or failed experience
records conditions and cannot authorize acceptance. Freeze all cells before a
separate independent audit. Never use audit/final to propose or rank policies.

## Tasks and interfaces

1. ResourceProfile produces a conjunctive assessment consumed by Controller.finish.
   Add CPU measurements. Test missing constraints, non-compensation, actual storage,
   fake precision, malformed limits, defaults and fixed-anchor regression.
   Observe failures, implement, run focused/full checks, commit.
2. Measured provider consumes existing TrainingExecutor and TrialRecorder. Add
   provenance and actual budgets. Test real rollback, immutable original, recorded
   rejected copies, unsupported proposals and no protected-data access.
3. CLI resource benchmark consumes profile/provider, produces frozen conditions,
   all candidate outcomes/measurements and audit-only task/seed comparisons. Test
   four arms, target remeasurement, experience provenance, hashes and one-use audit.
   Run CPU smoke across independent tasks/two seeds, full checks, one fresh branch
   review, one regression fix pass if needed, publish stacked PR without merge.

## Review focus

Missing measurements must not pass. Savings cannot buy a protected regression.
Model storage must not be labeled peak memory or fake precision savings. Timing
units/batch/hardware/sample count must be explicit. Experience must not smuggle
prior hardware measurements into target acceptance. Rejected provider trials and
partial training cost must not disappear. Audit cannot affect any selection.
