# SoT-inspired extension implementation plan

**Goal:** Integrate compact control telemetry and auditable evidence activation.
**Architecture:** Dedicated telemetry, pure extraction, immutable routing inputs,
EventLog harness, existing compute policy and lifecycle.
**Tech stack:** Python 3.12, existing torch/numpy/pytest/ruff/mypy; no dependencies.
**Spec:** ../../specs/state-of-thought.md

## Constraints and review focus

Preserve None, proxy labels, protected metrics, provider allow-list, copy trials
and rollback. Do not change TinyTransformer or rewards. Explicitly test malformed
v1 payloads, zero-vector direction, router selecting invalid IDs, failed audit
writes, and perturbed state leaking into observed telemetry.

## Tasks (execute inline)

- [x] Add failing tests in test_controller_reasoning.py for v2 migration and
  extraction (`state_displacement([0,0],[3,4]) == 5`, uniform binary entropy
  equals log(2), zero direction is None). Implement telemetry/reasoning.py and
  update telemetry/controller.py. Run focused tests.
- [x] Add routing tests for changed-state selection, invalid/unavailable evidence,
  immutable registry, complete drops audit and log failure. Implement eligibility
  snapshots in memory.py, routing.py and routing diagnostics. Run focused tests.
- [x] Add ablation tests for missingness-preserving zeroing/permutation, temporal
  scramble, dynamics-disabled and replay. Implement reasoning_ablations.py and
  reference random/independent/fixed/full routing modes. Run focused tests.
- [x] Add policy and lifecycle tests for opt-in state response, risk/proxy gates,
  unsupported actions and protected rollback. Extend config.py and policy.py;
  keep Controller transaction implementation unchanged. Run controller tests.
- [x] Add deterministic synthetic fixture using TemporalTelemetry and reference
  routing with A/B/C/D contexts; assert reproducibility, measured metrics and
  deliberate-change recovery. Implement reasoning_demo.py and run its CLI.
- [x] Document formulas, trust boundaries, examples, three timescales, synthetic
  limits and citation; update related docs. Run full tests/lint/types, review
  diff and repair actionable findings. Record exact results in delivery report.

## Execution record

Baseline main `b1da38e`: 207 tests passed before behavior changes. Each new task
began with failing feature tests. Review found mutable plugin output/control and
large-vector centroid roundoff; regression tests preceded their fixes. A second
review found changing-property selector output; exact result-type validation and
a regression closed that path. Independent review then reported no blockers.

Design rulings: retain existing task-gain acceptance (budget savings alone cannot
authorize a candidate); use across-sample coordinate permutation to preserve
units/domains; report model-based synthetic metrics as unavailable instead of
simulating providers. Execution remains a host responsibility. No existing tests
were weakened, no dependencies added, and core transaction code was unchanged.
