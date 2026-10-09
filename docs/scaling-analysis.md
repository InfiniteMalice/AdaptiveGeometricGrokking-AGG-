# Controlled scaling and held-out extrapolation

This opt-in grid tests parameter, geometry and training-data effects under declared
training budgets. It adapts RoboJEPA's controlled fit/extrapolate methodology; it
does not adopt a robotics model, fitted law, fixed pretrained encoder or planning
claim. AGG's existing Transformer and geometry adapter train together.

```sh
python -m agg.experiments.cli scaling --config configs/scaling-smoke.json --output runs/scaling
python -m agg.experiments.cli scaling-audit --run runs/scaling --seed 43
```

## Controls and accounting

Default fit widths: 8, 12, 16, 20. Width 24 is reserved for extrapolation. Cross
Euclidean/hyperbolic/product geometry, model seeds 17/19, nested training volumes
16/64, and budgets 4/8 updates. Batch 16, context 8, adapter dimension 4, gate 0.5
and learned scalar controls stay fixed. There are 120 configurations. Curved
geometries learn curvature; the Euclidean control learns its active bandwidth
scalar. The small test verifies matched trainable parameter counts at each width.
Actual counts remain in every result; parameter effects and within-width geometry
contrasts are reported separately.

One data seed (149) fixes the four structural roles and all selection/audit rows.
Only the training-role subset changes with volume; nested subsets come from a
predeclared shuffled order. All cells at the same width/seed start identically
before the geometry-specific forward rule. Training updates and token exposures
match within each volume/budget. Equal steps/tokens are **not equal FLOPs**.

Reports include actual completed updates, tokens, available/unique training rows,
a lower bound on observed distinct training rows, wall execution, model storage,
parameters and scoped CPU inference measurements. Failed configurations retain
reasons and attested lower bounds; unknown final work is null. Operator-complete
training FLOPs remain uninstrumented, so this is not a compute-optimal scaling law.
Wall measurements include training logging/checkpoint work and resource probes.

Training/selection completes and every artifact freezes before the independent
audit. Larger configurations are trained and reported, but their outcomes never
enter fitting, bootstrap resampling or form selection. Final data stay untouched.

## Fits and uncertainty

Two forms are declared in advance, separately for each geometry/volume/budget:

- Power error: `error = exp(a) * parameters**b`.
- Log-linear error: `error = a + b * log(parameters)`.

The response is independent audit OOD error. Fit all valid smaller-model rows;
require at least three distinct small sizes. Never silently replace zero error
with an arbitrary floor: the power form becomes unavailable. Preserve failed
measurements and unavailable reasons. Out-of-domain or nonfinite extrapolations
are flagged and never clipped to resemble success.

Whole-model-seed resampling preserves dependence across fitted widths. Reports
include coefficient intervals, fit residuals/RMSE, held-out predictions/residuals
and extrapolation RMSE. Prediction intervals describe uncertainty of the fitted
mean, not full future-observation coverage. Fewer than two seeds means no interval;
the two-seed smoke remains sparse and uncalibrated. These intervals condition on
one fixed data partition and a chosen form. They do not cover data-seed variation.

Within-width geometry contrasts use paired structural-cluster comparisons at the
same model seed, volume and budget. Volume contrasts pair model seeds and report
accuracy gains with bootstrap intervals. Flat responses and near-perfect fitted
observations are recorded. A small/zero volume effect does not prove saturation;
short training, weak representation or insufficient data variation can explain it.

## Horizon, causal and compression measurements

The audit locates each query's unique relevant memory, moves it to sampled distances
1/2/4/7, and independently verifies that the oracle label is unchanged. The existing
horizon helper reports the maximum sampled distance reaching 0.8 accuracy, with no
interpolation. If none qualifies, horizon is null with a reason, never zero.

PR-2 invariant/decisive measurements on audit OOD examples retain joint correctness,
correct required updates and missing conditional denominators. Geometry or phase
indicators are not thereby established as causes of generalization.

FP16/INT8 activation-rounding copies measure paired ID/OOD tolerance. The report
separates being within 0.02 accuracy tolerance from establishing baseline capability
(ID and OOD both at least 0.8). A consistently wrong model is not compression evidence.
This is simulated activation precision; physical savings are unavailable. No altered
copy is selected, retained or fed into the fitting pipeline. Each audit row counts
actual model forward calls/token inputs, including diagnostic copies and repeats.

## Conditional latent predictor

PR-6B is deferred. `ToyEnvironment` already executes and verifies integer state
transitions; a new predictor requires an independently justified AGG hypothesis
beyond that existing supervision. The scaling smoke does not supply such evidence.
No world-model training or robotics generalization is claimed. See [references](references.md)
and the [implementation plan](specs/scaling-analysis.md).

## CPU smoke outcome

All 120 declared configurations completed: 720 optimizer updates, 92,160 training
tokens, and parameter counts 1,457 / 2,557 / 3,913 / 5,525 / 7,393 across increasing
widths. Summed training/logging/probe wall time was 32.35 seconds while verification
also ran on the CPU; this is not a hardware performance comparison. Audit used
2,520 model forward calls and 983,040 token inputs for all 120 cells and diagnostics.

Audit OOD accuracy ranged from 0 to 0.53125. No model reached the sampled horizon
threshold, so all 120 effective horizons are explicitly unavailable. No baseline
established the required capability floor for compression evidence. Within-width
geometry OOD differences were -0.03125, 0 or +0.03125. Seed-paired gains from more
training rows ranged -0.171875 to 0 at the declared short budgets; extra data did
not establish a benefit here.

Across 12 geometry/volume/budget groups and two predeclared forms, held-out larger-
model error RMSE ranged 0.0245–0.2103. All fits were numerically available, but two
seeds and one larger width do not establish a scaling law, calibrated interval
coverage, saturation, or reliable extrapolation. Forms were not selected based on
these errors. No predictor or scientific final-test evaluation was authorized by
these measurements.

Verification: 353 tests pass (one upstream geoopt/Torch warning); Ruff clean;
mypy clean across 69 source files.


Fresh review found two important failure-isolation defects. Construction now runs
inside per-cell accounting and retains unavailable parameter counts; later cells
continue. Primary scores are persisted before optional diagnostics. Failed precision,
horizon, causal, resource or comparison measurements retain separate reasons and
cannot remove valid OOD responses from fitting. Four regression cases failed before
the fixes and pass afterward. No critical or minor findings; CodeRabbit was not run.
