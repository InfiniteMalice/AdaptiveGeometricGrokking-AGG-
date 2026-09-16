# AGG v0.1 implementation specification

The accompanying approved-research-brief.md is the scientific authority. This
implementation provides a CPU-runnable research harness, not evidence that its
hypotheses are true. No smoke experiment establishes grokking.

## Design

Use a small causal Transformer and deterministic modular addition, tree ancestry,
and associative retrieval generators. Train with AdamW, recording checkpoints,
held-out behavioral metrics and representation diagnostics on a shared step axis.
All interventions operate on independent model copies; rejection retains the exact
previous model and optimizer. Policies propose, runners execute, evaluators decide.

Telemetry is a versioned JSON-compatible record with raw, derived, availability,
and proxy annotations. Missing observations remain explicitly unavailable. ID/OOD
and mechanism constraints are independent and fail closed for nonfinite values.
Logical precision simulates rounding separately from physical encoders; benchmark
decode-plus-dense operations honestly, without claiming accelerated sparse kernels.

## Public boundaries

- `agg.tasks`: `TaskData` with train/ID/OOD token-label tensors and metadata.
- `agg.models`: `TinyTransformer.forward(tokens, return_hidden=False)`.
- `agg.telemetry`: `TelemetrySnapshot`, `TelemetryHistory`, `TelemetryCollector`.
- `agg.evaluation`: `Evaluation` and `Constraints` consumed by transactional trials.
- `agg.consolidation`: proposal-only `InterventionPolicy` and explicit trial runner.
- `agg.storage`: lossless ternary codecs independent of logical precision.
- `agg.supervision`, `agg.credit`: immutable execution evidence, source priorities,
  confidence, bounded mixing, and gamma return-to-go.

## Acceptance and risks

Test analytic derivatives, degenerate spectra, seeded task splits, independent
retrieval factors, persistent crossings, gate endpoints, rollback on exception and
constraint failure, invalid numerics, codec round trips and metadata costs, conflicting
supervision, gamma endpoints, and real flag removal. Run small end-to-end examples
for each family and retain actual outputs. Full multi-seed experiments remain
unestablished until their artifacts exist. Optional manifold/topology dependencies
must fail with actionable installation guidance; no silent Euclidean fallback.

Public configs and schemas are new interfaces at version 0.1. Risks include proxy
construct validity, small-sample geometry, data leakage, simulated precision versus
real kernels, and selection overfitting. Record these in research documentation.
