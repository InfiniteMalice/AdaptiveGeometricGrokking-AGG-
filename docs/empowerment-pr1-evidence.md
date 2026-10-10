# Empowerment PR 1 evidence and handoff

Scope: deterministic finite environments, discounted occupancy, effective MI,
exhaustive/restricted potential estimates, graph diagnostics, reward controls and
an opt-in CLI. Implementation details: [empowerment.md](empowerment.md).
The [integrated design](specs/empowerment-play.md) preserves all six PR stages.

## Scientific status

Supported implementation checks: the absorbing fork gives `gamma * ln(2)`;
the alternating chain agrees with its analytic infinite-horizon occupancy;
the asymmetric channel's optimized capacity is `ln(1.25)` with nonuniform prior.
Duplicate channel rows do not increase capacity. These are analytic calibration
checks, not newly discovered scientific results.

Constructed negative result: the distractor-only skill subset has approximately
1.40464 nats and zero task reward. Including the rewarding skill permits 0.9
normalized discounted reward. Constant reward yields zero adaptation advantage.
Neither finding establishes a relationship in learned representations.

Unsupported: better generalization, exploration-driven weight learning, memory
benefits, geometry superiority, compression robustness, or reproductions of the
two source papers. No agent training occurred in this stage.

## Validation protocol

Commands (Python 3.12.13, NumPy 2.5.3, PyTorch 2.14.1 CPU; one BLAS/OpenMP thread):

```text
python -m pytest -q
python -m ruff check agg tests
python -m ruff check .
python -m mypy agg
python -m agg.experiments.cli empowerment --config configs/empowerment-smoke.json --output NEW_DIRECTORY
```

Two independent output directories using the fixed config give identical numeric
records for all 38 starts. Each uses 366 occupancy solves and 813 capacity
iterations; all capacity solvers converge. No executed environment actions or
training updates are counted. Runtime is observational, not a speedup claim.

The 365 pre-existing tests remain part of the full suite. New tests cover analytic
math, nonconvergence, exhaustive caps, invalid inputs, immutability, reward
controls, JSON roundtrip, no overwrite, missing distances and RNG preservation.
The CLI import check confirms disabled empowerment performs no new import or
calculation. Existing checkpoint formats and telemetry versions are unchanged.

One independent whole-branch review found two numerical boundary bugs: marginal
product underflow and reciprocal overflow. Both were reproduced with failing
tests and fixed in one pass. The 28 focused tests pass after repair. Final complete
suite result: **393 passed, 1 upstream Geoopt/TorchScript deprecation warning in
118.43 seconds**. Ruff (`agg tests` and repository-wide `.`) passes; mypy reports
no issues in 76 source files. Raw output is included in the accompanying evidence
package; no second review is represented as having occurred.

## Integration and next PR

Reused: experiment CLI, shared JSON writer, package/dependency/test conventions.
The audit maps future reuse of GeometryAdapter, AbstractionRegistry, audited
routing, execution evidence, ledger and copy-based intervention trials. None of
those mechanisms has been duplicated or changed in this PR.

Baselines: fixed uniform prior versus optimized source, exhaustive versus
explicitly restricted policies, redundant outcome labels, rewarding versus
irrelevant outcome axes, and constant reward. Stochastic environments, learned
policies, Gaussian width and functional compression are out of scope here.

PR 2 should test latent/temporal/reachability relationships, outcome separation,
equivalent affordances, bottleneck/path removal and functional preservation with
matched geometry budgets. It must include both directions of the representation
similarity/behavior dissociation. Use held-out layouts and report compute and
sampling uncertainty separately from deterministic numerical error.

The previous research-integration final-test release remains unauthorized and
unexecuted. No pull request is merged by this stage.
