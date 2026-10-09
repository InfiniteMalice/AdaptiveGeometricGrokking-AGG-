# Milestone diagnostics

Milestones describe independently measured properties, not a universal sequence
or verified internal understanding. They remain opt-in reports with no optimizer,
reward or model-mutation interface. This adapts the idea of task-progress evidence
from World Potential Model; it does not reproduce that model or its training.

After freezing a run and producing both independent and causal audit reports:

```sh
python -m agg.experiments.cli causal --run runs/example --seed 29
python -m agg.experiments.cli independent --run runs/example --bootstrap-seed 29
python -m agg.experiments.cli milestones --run runs/example
```

Use fresh output directories; each report is one-use. Milestones validate the
frozen artifacts and require source audit reports bound to the same manifest.
They retain report hashes, measurement provenance, definitions and failures.
Final data are never read. An absent independent audit report leaves compression
observations missing; causal trajectory measurements are required.

## Definition version 1

These point-estimate thresholds are preregistered diagnostic hypotheses in code,
not validated learned progress potentials. Three consecutive sampled observations
confirm each property independently:

| Property | Required observations |
| --- | --- |
| Memorization gap | Training accuracy >=.9 and audit OOD accuracy <=.6 |
| Retrieval/reuse | Correct invariance, required update and relevant-input flip >=.8; irrelevant-input flip <=.1 |
| Structural generalization | OOD accuracy, decisive joint correctness and correct invariance >=.8 |
| Parameter-compression robustness | Fewer actual parameters, baseline audit OOD >=.8, and audit ID/OOD gains >=-.01 |

The source field says whether evidence is measured/verified; it never verifies
the scientific validity of the milestone itself. Training loss cannot qualify a
property. Missing, unverified or conflicting observations break consecutive
support. The history records current/previous measurements, sources and state.
Historical confirmation is preserved alongside subsequent contradictions or
missing evidence. Observed onset brackets and right-censoring use the existing
sustained-crossing implementation. No missing onset is converted into step zero.
The milestone onset lower bound is the last observed unsatisfied checkpoint;
missing, unverified or conflicting observations cannot establish such a bound.
Specific source failure and undefined-metric reasons survive report composition.

Hierarchy uses full-stratum balanced accuracy, with both ancestry classes
required. Causal reports now add `balanced_value`, `balanced_uncertainty`, and
`balanced_missing_reason` inside structural accuracy; all previous fields keep
their meanings. Older reports without these additive fields produce missing
hierarchy accuracy. Non-retrieval tasks have no supported distractor-context
intervention, so their retrieval milestone remains missing.

Compression is observed once, after selection, at the terminal training step.
Earlier checkpoints have no compression evidence. One successful compression
event cannot establish three-observation sustained robustness. Parameter counts
come from frozen inference artifacts; unstructured zeros, logical precision and
physical memory are separate quantities. No deployment savings are inferred.

## Future measurements and limits

For every earlier checkpoint, the report pairs raw training loss and each
milestone's current satisfied/unsatisfied observation with the next checkpoint's
audit OOD accuracy and required-update rate. It preserves actual paired values,
both steps, unavailable observations and descriptive Pearson correlations.
Fewer than three pairs or constant predictors/targets leave correlation null.
Repeated checkpoints share a model and audit structures; they are dependent, and
no independent-run confidence interval is claimed.

This is exploratory temporal association on withheld task structures, not
validated prediction across unseen task families. Use independent seeds/tasks
and frozen thresholds in the later integrated matrix to test predictive value.
The report's `reward_authorized` is always false. A process-feedback adapter and
shaped-training comparison are deferred until evidence and a suitable objective
exist. Explanations, coherence and claimed understanding never become rewards.

Source: Zhao et al. (2026), [reference registry](references.md). Implementation
contract and tests: [specification](specs/milestones.md).

## Executed bounded fixtures

Starting from `configs/causal-smoke.json`, run retrieval with data/model seeds
83/13 and modular with 84/14, leaving the six-update budget, evaluation interval2
and two-candidate search unchanged. Freeze both selections before any audit;
then run causal and independent reports with seed29, followed by milestones.
These are separate task fixtures, not a matched cross-task superiority comparison.

All four milestones are censored at step6 in both runs. Retrieval milestones
remain false; modular retrieval evidence is unavailable because that task has no
distractor-context intervention. Compression has one terminal observation and
does not satisfy the definition. No rewards are authorized.

There are three loss-to-next-checkpoint OOD pairs per run. Retrieval's descriptive
Pearson value is +.879583, based on only three dependent pairs; it does not
establish predictive validity. Modular accuracy and both required-update targets
are constant, leaving their correlations unavailable. Milestone predictors are
constant or missing, so this smoke provides no evidence of predictive advantage.
The absence of a milestone is not proof that no such transition exists later.

After fresh review: 318 tests passed (one upstream geoopt warning), Ruff clean,
mypy clean across 62 source files. Regression fixes prevent unknown evidence from
bounding onset and preserve specific source failure/undefined-metric reasons.
Both smoke pipelines were rerun with these fixes and retain the censored result.
