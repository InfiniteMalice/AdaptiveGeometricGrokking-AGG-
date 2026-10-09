# Empowerment reference implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkboxes.

**Goal:** A bounded, numerically checked finite-state empowerment reference.
**Architecture:** Pure environment/estimator modules plus an opt-in CLI report.
**Tech Stack:** Python 3.12, NumPy, existing pytest/Ruff/mypy.
**Spec:** `docs/specs/empowerment-play.md`.

## Global constraints

- Deterministic CPU environments; nats internally; occupancy includes t=0.
- Explicit exhaustive cap; restricted capacity never implies full potential.
- No changes to legacy defaults, controller acceptance, precision or checkpoints.
- No new dependency, learned policy, large factorial, merge or final release.

## Review focus

- Degenerate channels/zero priors must be finite without fabricating observations.
- Failed convergence must remain visible, with restricted upper-bound scope.
- Enumeration must reject oversized spaces before allocation.
- Mutating input arrays must not change environment identity.
- Unreachable distances must serialize as unavailable rather than infinity or zero.

## Task 1: finite mathematics and fixtures

Files: `agg/empowerment/{__init__,environments,estimators}.py`,
`tests/test_empowerment.py`.
Interfaces: `FiniteEnvironment.step`, `distances`, `goal_policies`;
`discounted_occupancy(env,policy,start,gamma)`, `effective_information(channel,prior)`,
`channel_capacity(channel,tolerance,max_iterations)`,
`potential_empowerment(env,start,gamma,mode,max_policies)`.

- [ ] Write tests: fork MI = gamma*ln(2); alternating-chain analytic occupancy;
  gamma=0; invalid probability/indices; immutable transitions; disconnected BFS;
  asymmetric channel capacity ln(1.25), redundant rows, nonconvergence and cap.
- [ ] Run `python -m pytest tests/test_empowerment.py -q`; observe missing module.
- [ ] Implement immutable deterministic table, grid/gateway/central/distractor
  fixtures, BFS and graph betweenness, float64 resolvent and channel solver.
- [ ] Run focused tests, Ruff/mypy, and commit this independently testable unit.

## Task 2: opt-in report and documentation

Files: `agg/experiments/empowerment.py`, CLI, example config, tests, references,
`docs/empowerment.md`.
Interface: `EmpowermentConfig.from_dict`, `run_empowerment(config, output)`.

- [ ] Write report tests: schema roundtrip, deterministic numeric replay, unavailable
  unreachable distances, actual protocol/cost labels, reward anisotropy, no overwrite.
- [ ] Observe failing tests, then implement strict config/report with hashes,
  occupancies, priors, reward comparisons and capacity bounds. No training metrics.
- [ ] Run new CPU smoke twice, compare numerical results, then full pytest/Ruff/mypy.
- [ ] One fresh whole-branch review; fix material findings with failing regression
  tests, update actual result documentation, commit and publish stacked PR 1.

Self-review: all PR 1 requirements mapped; optional Gaussian width omitted and
explicitly scoped; six-stage requirements retained in the linked spec.
