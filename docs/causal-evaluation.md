# Independent causal evaluation

The opt-in independent runner now saves versioned inference artifacts at every
existing evaluation checkpoint. Training checkpoints keep their original schema.
After selection freezes, run:

```sh
python -m agg.experiments.cli run --config configs/causal-smoke.json --output runs/causal
python -m agg.experiments.cli causal --run runs/causal --seed 19
```

The second command consumes audit data only. It writes a one-use attempt marker,
`causal-report.json`, and failure records if reporting fails. It refuses changed
frozen artifacts and never changes selected weights. Legacy frozen runs without
inference snapshots can report their selected model, with missing trajectory.
Final data remain untouched. Keep the seed and transform definition fixed before
examining this report; later adaptation requires fresh audit structures.

## Definitions and scope

Every pair is independently decoded by the task oracle before model inference.
Retrieval permutes irrelevant positions (holding the relevant distance fixed) or
replaces the relevant value. Modular invariance swaps operands; decisive pairs
replace one operand. Hierarchy pairs replace one queried node, with an unchanged
or changed ancestry answer. Non-retrieval replacements must exist in the same
private audit stratum. The hierarchy model has no edge-list input, so arbitrary
edge edits and graph renaming are unsupported. Context extension is also outside
this benchmark's fixed positional capacity. No-op rows are missing, never successes.

All rates record numerator, denominator, clustered uncertainty and missingness:

| Measurement | Definition |
| --- | --- |
| Correct invariance | Both original and invariant-transformed answers correct / valid invariant pairs |
| Spurious flip | Prediction changes / valid invariant pairs |
| Required update | Decisive-transformed correct / originally correct decisive pairs |
| Joint correctness | Both answers correct / valid pairs |
| Structural accuracy | Correct predictions on the original private ID or OOD stratum |
| Relevant-feature dependence | Prediction changes / valid decisive pairs; correctness reported separately |
| Distractor dependence | Retrieval prediction changes / valid irrelevant-context permutations |
| Causal sensitivity | Decisive flip rate minus invariant flip rate; diagnostic, no causal-mechanism claim |
| Failure recovery | Transformed correct / originally wrong pairs; counterfactual recovery, not an iterative retry |

The report retains each original/transformed label and prediction. A constant
wrong model cannot receive correct-invariance credit. Correct decisive changes
are never penalized as spurious flips. Empty conditional denominators remain
null. Rates are row-weighted; hierarchy prevalence is naturally imbalanced.
Interpret its raw structural accuracy together with class-aware independent
reports, not as balanced accuracy.

Uncertainty uses the PR-1 paired cluster bootstrap. Reused transformation endpoints
merge their source clusters into connected components before resampling. A single
connected component yields no interval. Few key clusters cannot establish strong
statistical claims; descriptive intervals do not adjust for post-report adaptation.
Sensitivity reports both support counts; joint uncertainty for that contrast is
explicitly unavailable. Marginal intervals must not be subtracted to infer it.

## Phase linkage

All saved checkpoint measurements are retained. If the existing sustained M/R
proxy crossing exists, the report links the last available checkpoint before
onset, the first within onset/confirmation, and the first after confirmation.
Failed checkpoint measurements remain in the trajectory but cannot supply a phase link.
The original observed onset bracket and censoring are preserved. Absent crossings
or checkpoints produce null phase links, not an invented onset at step zero.
The existing kNN and attribution quantities remain proxies. Correlation with
independent causal robustness does not establish that geometry caused learning.

## Executed smoke

`configs/causal-smoke.json`, data/model seeds 73/11, six baseline updates,
checkpoints 0/2/4/6, two candidates (one distillation update and pruning), seed19
for transformations/bootstrap. Audit ID/OOD: 64/32 rows, four key clusters.
All selections froze before the causal audit. OOD structural accuracy and correct
invariance were .34375 at all four checkpoints. Spurious flips and required
updates were both zero. ID accuracy was .3125 at step0 and .328125 thereafter.
M/R never crossed: onset is right-censored at step6; all three phase links are
unavailable. Stable predictions therefore did not demonstrate causal competence.
This is a software smoke negative finding, not a grokking experiment.

Validation covers oracle agreement for all three tasks/strata, protected support,
ambiguous retrieval, no-op missingness, wrong-answer consistency, decisive updates,
cluster dependence, censoring, real checkpoint reports and failed-checkpoint
continuation. Research motivation: Beucler et al. (2026), with the adaptation
boundary in [references](references.md). See [design](specs/causal-evaluation.md).

After fresh review and regression fixes: 312 tests passed (one upstream geoopt
warning), Ruff clean and mypy clean across 60 source files. Review fixes exclude
failed checkpoints from phase links and add full-stratum statistical accounting.
The smoke was rerun against the corrected report. At step0, .03125 of decisive
OOD predictions changed, but none of the originally correct predictions updated
correctly; raw sensitivity alone would therefore be misleading.
