# Experiment matrix

This document specifies research protocols. Named configuration families are runnable controls; their availability is not evidence that the complete protocols below have been executed. Consult the verification report for actual runs.

| Family | Implemented substrate | Planned substantive experiment |
|---|---|---|
| Modular addition | Seeded pair partitions; random ID holdout and upper-operand-corner OOD holdout | Reproduce delayed generalization across seeds before testing acceleration |
| Hierarchy | Exhaustive strict-ancestry labels; selected deepest descendants withheld | Compare matched adapters and balanced accuracy; control majority-class performance |
| Retrieval | Independently configurable length, relevant distance, distractor density and hard distractors | Estimate a gate × distance × density × context phase surface |
| Representation | Spectra, cheap geometry, normalized-depth derivatives, CKA | Predict accepted interventions on held-out seeds and measure metric redundancy |
| Consolidation | Independent copied candidates, ID/OOD/mechanism constraints, decision ledger | Compare against ordinary continued training using equal extra update budgets |
| Precision/storage | Fake numerical rounding, exact ternary codecs, measured CPU decode-plus-dense operations | Find component-specific acceptance floors; select storage on measured bytes or latency |
| Executable supervision | Integer-register execution trace, source-aware mixing and discounted credit | Validate conflict handling; later train an agent under these evidence regimes |

## Shared protocol

Use paired seeds 0–9 as a proposed initial replication set, then report the exact actual list. Fix dataset construction seeds separately from model seeds where the runner supports that distinction; otherwise record the coupling. Preserve train/ID/OOD metadata and hashes. Match architecture width/depth, examples, baseline updates, candidate updates and evaluation frequency. Count candidate search effort and telemetry cost separately. Include an equal-update baseline whenever a candidate receives extra optimization.

Record every evaluated step. For retrieval, pre-register `R > M + margin`, margin, and the number of consecutive evaluations required. Report first crossing, stable onset, confirmation step, sampled onset bracket, total budget and right-censoring. A sampled bracket is not a confidence interval. The existing seed summary reports only the observed median and retains censor counts; survival estimates and bootstrap intervals require additional analysis. Never replace censored values with zero or silently drop them.

## Retrieval factorial

A proposed starting grid is gates `{0, .25, .5, .75, 1, learned}`, context lengths `{8, 16, 32}`, relevant distances `{1, 3, 6}`, densities `{0, .5, 1}`, and both random/hard distractors. Keep distance strictly below `context_length - 1`, leaving a longer OOD condition. Record excluded combinations rather than changing factors silently. Run the same seeds per valid cell. Distinguish gates active during baseline training from post-training gate interventions using `adapt_during_training`. Inspect relevant-memory removal, distractor removal and counterfactual replacement alongside neighbor agreement. Compare label accuracy and counterfactual target accuracy, not merely changed predictions.

The generator reserves value `(key + seed) % values` for each query key in the ID
partition and excludes that binding from training. This prevents exact-row leakage
even with zero distractors. OOD uses reserved bindings at a strictly longer distance.
This is binding generalization within the same key/value vocabulary; label the
partition rule rather than calling it an unrestricted IID sample.

## Checkpoint compression

Preregister checkpoint classes with independent behavioral/probe thresholds. If no checkpoint meets a class, report it unavailable. At each available class run identical pruning, rank, dimension, geometry and precision grids. The library has `low_rank`; an automated low-rank proposal sweep is not implied by its availability. Preserve both absolute performance and candidate-minus-reference deltas. Compare successful feasible candidates only after displaying rejection rates and reasons.

## Geometry and dimension sufficiency

Sweep dimensions `{8, 16, 32, 64, 128}` only where width permits; the compact default supports a smaller subset. Hold dimension and prototype count fixed across geometry comparisons. Dimensions change parameter cost and must be analyzed separately. Full-telemetry candidate trials record reconstruction and graph topology. Graph radii are independently median-scaled and are not persistent-homology estimates. Other telemetry modes skip those offline summaries.

## Reporting

Retain configs, seeds, environment versions, checkpoints, telemetry, ledger decisions and storage timings. Report effect distributions, compute cost, missing observations and independent test results. `candidate_changes` supplies exploratory adjacent-window shifts, `telemetry_redundancy` supplies Pearson correlations, and `phase_surface` retains seed censoring. These helpers do not establish transitions or causal effects. Multi-seed grokking reproduction and trained-agent credit comparisons remain scientific experiments.
`integrated-evidence.md` records the executed PR0–PR8 fixtures, exact contrasts and protected release procedure; these short runs do not establish the substantive hypotheses above.
