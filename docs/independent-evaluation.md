# Independent candidate evaluation

PR-1 adds an opt-in evaluation protocol inspired by Hu and Tang's selection-bias
study. It does not reproduce their acceptance rules or establish improvement.
See [references](references.md) and the [integration contract](specs/research-integration.md).

## Run and audit

```bash
python -m agg.experiments.cli run --config configs/independent-smoke.json --output runs/independent
python -m agg.experiments.cli independent --run runs/independent --role audit
```

Use a new output directory. `independent_evaluation` defaults to false. Enabling
it requires an explicit `data_seed`; `training.seed` controls model initialization,
batches and optimizer randomness. A data seed can also be set for legacy runs.

Training and proposals receive ordinary `TaskData` with training and selection
ID/OOD only. The runner freezes selection before any independent report. It
neither reads audit scores nor calls final evaluation. Independent evaluation is
a separate command which validates the frozen artifact hashes first.

## Partition contract

| Family | Protected group | Selection/audit/final ID and OOD |
|---|---|---|
| Modular | Unordered operand pair; both commuted variants share a role | Non-corner versus upper-operand-corner; at least three corner groups required |
| Hierarchy | Entire depth-two subtree, including both node positions; shared ancestors and cross-subtree pairs excluded | Shallower versus deepest descendants within each role; depth at least 4 to retain both ID classes |
| Retrieval | Query and distractor keys assigned to disjoint role vocabularies | Configured versus longer relevant distance; OOD uses the existing generator's reserved values |

The hierarchy protocol measures held-out subtrees of a regular binary tree, not
transfer to unseen topology. It omits shared-root relations to protect lineage.
Each hierarchy role currently has one subtree, so cluster uncertainty is unavailable.
Retrieval uses four local keys per role mapped into sixteen global keys. All
variants of a binding remain within one role; no protected key appears as a
development distractor. This is harder than the legacy binding holdout and can
confound unseen-token exposure with rule generalization. Report that limitation.
Modular corner strata may have few groups; do not infer reliable uncertainty from
the existence of a bootstrap interval.

The partition manifest includes schema, algorithm, seed, per-stratum row counts,
cluster identities and tensor hashes. Too few structures fails explicitly.
The protocol retains the baseline vocabulary and classifier interfaces. It does
not silently fall back to row-wise splitting.

## Candidate records and freeze

`candidate-attempts.jsonl` uses `agg.candidate-attempt/1`. A proposal record is
written before evaluation or mutation. Completion records retain accepted,
rejected, failed and reference-failed attempts. A reference failure still aborts
the experiment. An incomplete run cannot produce a normal frozen report.

Each attempt records family/component, proposal, index, parent state/checkpoint,
parameter count, configured update budget, optimizer provenance, selection
before/after and protected anchor, outcome/reason and wall time. Reuse cycle is
the count of candidate attempts on the same selection set, not the number of
forward passes. Wall time includes trial/telemetry/serialization; it is not an
inference benchmark. FLOPs and unmeasured deployment costs remain null.

Each attempt saves its actual parent model as `candidate-NNNN-parent.pt`.
`original_training_checkpoint` separately identifies the initial optimizer/RNG
artifact; it is not misreported as the model parent after sequential interventions.

`candidate-NNNN.pt` retains loadable candidates, including rejected ones. Fitting
candidates also retain actual fresh AdamW/model/batch-RNG state in
`candidate-NNNN-optimizer.pt`; the parent training checkpoint retains its optimizer
and RNG. A candidate never resumes or overwrites the parent optimizer. Failed
application may have no loadable model and is reported with null measurements.
The original `ledger.jsonl` schema and acceptance semantics remain available.

`selection-frozen.json` uses `agg.frozen-selection/1` and binds all pre-audit files,
including config, candidate history, baseline/selected models and training
checkpoints. Each report validates file containment, existence and hashes and
reconstructs the partition manifest. The selected checkpoint never changes during
audit. Hashes detect accidental alteration relative to a trusted manifest; they
are not signatures or a defense against a host replacing both files and manifest.

## Independent measurements

`audit-report.json` contains baseline, selected and all available candidates.
Each stratum reports independent accuracy (balanced accuracy for hierarchy),
selection score, selection-minus-independent gap and paired difference from the
original baseline. Failed candidates retain their reason instead of disappearing.
An absent observation is null. An accepted candidate is feasible in its trial;
it need not be the final stage winner.

Uncertainty is a seeded 95% percentile bootstrap of whole paired structural groups,
within each ID/OOD stratum. The estimand is row accuracy, or macro class recall
for hierarchy. Resampling repeats all rows of a selected group and keeps candidate
and reference predictions paired. Fewer than two groups, or a resample missing a
class, makes the interval unavailable. Point estimates remain reported. This is a
descriptive interval, not correction for adaptive selection or repeated testing.

Reporting creates a one-use `audit-attempt.json` or `final-attempt.json` before
protected inference. A repeat or incomplete attempt fails rather than overwrites.
Failure records are preserved. A later study informed by an audit must declare
that reuse and designate fresh protected structures/seeds.

## Final release

After fixing the hypotheses, analysis and candidate set, an authorized researcher
computes the SHA256 of `selection-frozen.json`, then invokes:

```bash
python -m agg.experiments.cli independent --run runs/independent --role final \
  --authorization preregistration-or-release-id --manifest-sha256 ACTUAL_SHA256
```

The command refuses missing/blank authorization or a mismatched manifest hash
before final inference. Authorization is a recorded host attestation, not an
identity service. Trusted Python callers can construct task data; deploy the
reporter in a separate process/account if adversarial isolation is required.
Unit tests authorize their synthetic fixtures only. No scientific final release
is authorized by those tests. Final results are unavailable to candidate APIs.

## Selection comparison

`selection_comparison(config)` returns four retrieval configs on the same protected
key groups: existing selection, doubled selection samples, independent audit
reporting, and a conservative selection rule. Execute each with `run_experiment`.
Training data, model seed, steps and candidate grids remain matched. The first
and third configs intentionally make identical selections; generate an audit report
for the third to assess reporting bias. The larger arm adds evaluation samples
only and must report its extra evaluation/search cost. This tests the existing
selector on the new protocol, not numerical equivalence to legacy dataset splits.

`selection_gain_floor` defaults to null. When configured, both ID and OOD
selection gains over the local parent must strictly exceed the floor, in addition
to all existing fixed-anchor constraints. The comparison fixture uses 0.01.
This heuristic is not McNemar, PACE or a Bayesian rule from the paper. It makes
no statistical-validity or superiority claim and never reads independent scores.

## Verification

`test_evaluation_protocol.py` exercises structural overlap, semantic labels,
seeds and legacy defaults. `test_independent_trials.py` exercises proposal intent,
failure accounting and parent/RNG preservation. `test_independent_reporting.py`
trains a real small model and checks frozen audit, optimizer artifacts, paired
statistics, explicit final authorization, stale artifacts and one-use reports.
`test_selection_comparison.py` checks conservative gates and matched training.
Run these with the full existing suite, Ruff and mypy.
