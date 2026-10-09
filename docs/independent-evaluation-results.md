# Independent evaluation verification

These are CPU software smoke measurements, not evidence of grokking or a
superior selection rule. The hypothesis from Hu and Tang (2026) is that adaptive
selection may overstate independent improvement; eight tiny runs cannot test it
reliably. See [references](references.md) and the [protocol](independent-evaluation.md).

## Executed checks

Python 3.12.13, PyTorch 2.14.1+cpu, NumPy 2.5.3; Windows; one OpenMP/MKL thread.
`python -m pytest -q`: 304 passed, one upstream geoopt TorchScript warning.
`python -m ruff check .`: clean. `python -m mypy agg`: 57 source files clean.
The final statistics-only regression run passed all nine tests after a typing
normalization. The pre-change baseline had 283 passing tests.

Fresh branch review identified sequential-parent provenance, missing class
support, nonfinite candidate evaluation, and baseline failure logging defects.
Each was reproduced in a failing regression test and repaired. Existing defaults
and trial rollback remain covered by the complete suite.

## Bounded comparison

All arms use `configs/independent-smoke.json`: retrieval, width 8, one layer,
two heads, three baseline updates, one fitted distillation update and one pruning
candidate. Each run considers two candidates. For seed pairs (data/model)
71/11 and 72/12, `selection_comparison` creates four arms. All eight selections
were frozen before any audit. Bootstrap seed: 401; 1,000 paired cluster draws.
Audit ID/OOD contain 64/32 rows and four independent key clusters each.

| Data/model seed | Arm | Selection ID | Selection OOD | Audit ID | Audit OOD | Accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 71/11 | Existing | .187500 | .000000 | .359375 | .562500 | 2 |
| 71/11 | Larger selection | .270833 | .000000 | .359375 | .562500 | 2 |
| 71/11 | Audit reporting | .187500 | .000000 | .359375 | .562500 | 2 |
| 71/11 | Conservative | .187500 | .000000 | .328125 | .312500 | 0 |
| 72/12 | Existing | .187500 | .125000 | .156250 | .187500 | 0 |
| 72/12 | Larger selection | .250000 | .062500 | .156250 | .187500 | 0 |
| 72/12 | Audit reporting | .187500 | .125000 | .156250 | .187500 | 0 |
| 72/12 | Conservative | .187500 | .125000 | .156250 | .187500 | 0 |

Larger selection changes only that role's rows (ID 64 to 96 and OOD 32 to 64),
with identical training/audit/final data and search budgets. Conservative requires
both local-parent ID and OOD gains greater than .01 as well as all original
anchor gates. Audit reporting intentionally has the same selection behavior as
Existing; it changes reporting, not the optimizer. Every arm receives a post hoc
audit for this comparison.

For 71/11 Existing, paired audit ID gain is .03125 with descriptive 95% interval
[0, .113208]. The OOD gain is .25. Four clusters and two seed pairs are inadequate
for a strong uncertainty claim. Different key/value support can make selection
gaps negative. No final scientific data were released. No training run here was
long enough to establish a phase transition; absence of onset is censored.

## Reproduction

Run the documented CLI smoke and audit first. For the matrix, load the smoke
config with `ExperimentConfig.from_dict`, use `dataclasses.replace` for the two
seed pairs, then iterate `selection_comparison(config).items()` and call
`run_experiment(arm, fresh_output_directory)`. After **all** calls return frozen
runs, call `report_run(directory, role="audit", bootstrap_seed=401)` for each.
Do not reuse output directories or tune this frozen comparison from its audit.

Candidate JSONL and reports include selection/audit gaps, paired intervals,
missingness, candidate/reuse counts, acceptance reasons and measured fitting wall
time. Compute FLOPs and deployment memory are not measured by this stage and
remain null; active parameters are not physical savings. Final-data authorization
in unit tests applies only to disposable test fixtures.
