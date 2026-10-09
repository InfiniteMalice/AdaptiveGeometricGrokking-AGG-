# Research integration contract

Status: staged implementation contract, 2026-10-08. Governing authority: the
user's research integration request and `approved-research-brief.md`.
Baseline revision: `c50fbe4aacf302d2c146456a55d0decdfb12c951`.

## Purpose and invariants

Extend the existing experimental harness to distinguish development improvement
from independent behavioral improvement. Preserve **Understand → Restructure →
Compress → Verify**. Software fixtures establish implementation behavior only.
All new mechanisms are opt-in. Existing configurations retain their behavior.

The trainer and candidate selector may consume training and selection data only.
Independent audit results are reporting evidence after selection is frozen.
Final evaluation requires explicit authorization for frozen checkpoints. Neither
audit nor final results may feed back into the current experiment's selection.
If a researcher adapts a later experiment to an audit, label that reuse and obtain
fresh confirmatory data. Python object boundaries are not a security sandbox.

Keep protected ID/OOD/mechanism constraints and the original capability anchor.
Never trade a protected regression for smaller size or faster execution. Preserve
copy-based trials, immutable checkpoint artifacts, RNG restoration and rollback.
Missing measurements remain null, including runtime or memory not actually measured.
Version new serialized records; do not silently reinterpret existing schemas.

## Implementation map and duplication review

| Existing component | Extension seam | Do not duplicate |
|---|---|---|
| `tasks/synthetic.py`: `Split`, `TaskData`, task oracles | Structurally grouped four-role protocol returning an ordinary development `TaskData` | Trainer/model/task tensor interfaces |
| `experiments/runner.py`, `candidates.py` | Optional data seed and evaluation protocol; freeze selection before audit | Candidate enumeration or staged selection policy |
| `consolidation/trials.py` | Pre-attempt accounting and optional candidate artifact callback | Copy, constraint, rollback and RNG semantics |
| `evaluation/model.py`, `training.evaluate` | Reuse score definitions for independent paired predictions | Second classifier evaluator with different accuracy semantics |
| `probes`, `experiments/diagnostics.py` | Oracle-validated paired counterfactual measurements | M/R/G or existing sustained-crossing calculation |
| `telemetry`, `probes.retrieval_crossing` | Optional versioned milestone evidence summaries | Learned reward or universal stage sequence |
| `controller.memory`, `curriculum`, `routing`, `events` | Lagged transfer updates on existing experimental abstractions | Another knowledge store or evidence validity system |
| `controller.core`, `training`, `evaluation` | Additional conjunctive resource feasibility gates and actual provider measurements | Alternative model mutator or compensatory fitness score |
| `experiments.analysis`, `matrix` | Controlled fit/extrapolation and reproducible comparison manifests | A claim that a borrowed scaling law already holds |
| `supervision.ToyEnvironment`, `credit` | Optional independently tested predictors/credit objectives | Relabeling discounted returns as a trained RL algorithm |

## Stages and evidence gates

Each row is a separate PR. Later PRs target the preceding branch until its merge;
no automatic merges. Each implemented mechanism needs deterministic unit tests,
a small end-to-end fixture, Ruff, mypy, full pytest, and a limitations report.

| PR | Bounded deliverable | Gate before dependent work |
|---|---|---|
| 0 | Baseline inventory, protocol, source registry and implementation plan | Execute and record baseline checks; resolve conflicting interfaces |
| 1 | Four data roles, separate task/model seeds, complete candidate accounting, frozen audit/final reports, paired cluster uncertainty | Structural overlap tests, no audit/final selector access, unchanged rollback, actual small training fixture |
| 2 | Invariant and decisive transformations with independent semantic validators | Separate original/transformed correctness; constant-wrong and change-required controls; failed/inconclusive records |
| 3 | Versioned nonmonotonic milestone diagnostics | Provenance, missing/conflicting evidence, sustained confirmation and censoring tests; no optimizer coupling |
| 4 | Block summaries in `AbstractionRegistry` with lagged utility | Future-block matched comparisons, provenance, negative/obsolete/out-of-scope controls; summary text never authorizes mutation |
| 5 | Joint resource constraints using actual providers | All protected gates mandatory; measured margins and hardware; unsupported actions remain unsupported |
| 6A | Scaling fit, residuals, uncertainty and excluded larger-run extrapolation | Honest parameter/data/compute accounting and matched geometry controls |
| 6B | Optional action-conditioned latent predictor | Proceed only after 6A and a distinct, independently justified AGG hypothesis; actual environment targets |
| 7 | Optional k-step discrete-policy experiment | Verify numerical objective against exhaustive small MDP; SCP only with verified assumptions and available solver |
| 8 | Integrated ablation manifest and independent reporting | Required prerequisites tested; freeze hypotheses/budgets/seeds before final evaluation |

## PR-1 design

Prefer a protocol envelope around existing `TaskData` over adding audit/final
fields to `TaskData`. The latter would expose protected data to every training
callback and candidate factory. A wholly separate training runner would duplicate
acceptance and rollback. The envelope keeps the existing runner and evaluator.

Partition underlying structures before constructing development views. Modular
addition groups commuted operand pairs. Retrieval groups all variants of a query
key (a stronger boundary than query-value binding alone), including distractor
bindings; protected-role keys do not appear as distractors in development data.
Hierarchy holds out whole subtrees; related ancestor/descendant pairs and sibling
variants stay within their assigned structural group. Reject configurations with
too few nonempty groups; do not silently fall back to random row splitting.

Keep ID and structural OOD strata within selection, audit and final roles.
Record partition algorithm/version, task-construction seed, group identities,
sample counts and fingerprints. Model initialization and training seeds remain
independent configuration. Default legacy runs remain three-way development runs.

Append proposal intent before execution, then success/rejection/failure outcomes
with family, parent/model hashes, checkpoint and optimizer provenance, candidate
index, selection reuse count, budgets, selection scores and protected regressions.
Candidate fit optimizers are fresh, not inherited; record that distinction.
Preserve rejected/failed attempts even if evaluation fails. Audit every available
candidate checkpoint only after selection closes; failed candidates retain null
scores and a reason. Audit reporting cannot alter accepted checkpoints.

Freeze a manifest binding configuration, partitions, candidate history and selected
checkpoint. A separate final evaluator validates that manifest and requires a
named authorization. Final data/outcomes never enter proposal generation or
acceptance. On conflicting or stale artifacts, fail before evaluating final data.

Report absolute selection/audit scores and their gap, plus paired candidate-minus-
reference effects. Bootstrap independent structural groups, preserving paired
rows and task strata; report cluster count, seed, method and uncertainty. A single
cluster has unavailable uncertainty, not zero uncertainty. Adaptive selection
intervals are descriptive and cannot be labeled selection-adjusted confidence.

Compare legacy selection, larger development selection, independent audit reporting,
and experimental conservative selection with matched model/training/search budgets.
Conservative rules apply only to selection evidence and remain experimental;
they do not consume audits or replace protected gates. No superiority is assumed.

## Scientific reporting contract

Every report identifies hypothesis, manipulated variable, controls, seeds, examples,
training/evaluation/search budgets, measurement definitions, baselines/ablations,
ID/OOD outcomes, uncertainty, missing/censored runs, and rejected/failed proposals.
Report observed onset brackets separately from statistical intervals. Compression
reports separate logical precision, physical bytes, decoder overhead, runtime,
numerical error, behavior and mechanism proxies. Keep negative results.

PR-3 rewards, PR-6B prediction, and PR-7 SCP are conditional research work. The
existence of telemetry, environment traces or a numerical prototype does not
satisfy their scientific prerequisite. Record an unmet gate explicitly.

## Verification and known risks

Use `python -m pytest -q`, `python -m ruff check agg tests`, and
`python -m mypy agg`. Also run the contributor's broader Ruff check before PRs.
The independent-evaluation fixture must train a small real model, run accepted,
rejected and failed candidates, freeze selection, and exercise explicit final
authorization. Tests inspect model hashes and audit access order.

The strongest risk is structural leakage disguised by disjoint row hashes.
Other risks are audit-driven adaptation, tiny cluster counts, differing metric
definitions, selection-cost imbalance and treating reference-codec timings as
deployment savings. Unit tests cover protocol mechanics; scientific claims need
new seeds, matched budgets and independently retained evidence.
