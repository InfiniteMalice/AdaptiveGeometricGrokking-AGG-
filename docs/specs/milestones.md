# PR-3: Evidence-based milestone diagnostics

Add a report-only layer over frozen training, independent audit and causal
measurements. Four independently defined properties (memorization gap, relevant
retrieval, structural generalization, parameter-compression robustness) may occur
in any order, reverse, or never occur. Thresholds are versioned research hypotheses,
not established universal stages. World Potential Model motivates measuring
task progress; AGG does not implement its pretrained potential model or rewards.

Every observation retains metric/source, verification state, current/previous
measurements, unavailable reasons and conflicts. Use existing `retrieval_crossing`
for sustained property confirmation and onset brackets. Missing/unverified or
conflicting observations interrupt consecutive support. Preserve historical
confirmation and later contradictory measurements separately. Loss alone cannot
establish a milestone. A report has no optimizer/model mutation interface.

Default definitions: train >=.9 and audit OOD <=.6 for a memorization gap;
correct invariance >=.8, required update >=.8, relevant flip >=.8 and irrelevant
flip <=.1 for retrieval; OOD accuracy, decisive joint correctness and correct
invariance >=.8 for structural generalization. Compression requires actual
parameter reduction, competent baseline OOD >=.8 and independent ID/OOD gains
>=-.01. Three consecutive sampled observations confirm a property. Compression
is measured only after candidate selection, so a single intervention cannot
receive sustained confirmation. This is parameter robustness, not physical savings.

Evaluate loss and earlier milestone observations against the next saved
checkpoint's held-out accuracy and required-update rate. Report actual pairs,
missingness and descriptive correlations; constant features/targets yield no
correlation. Repeated checkpoints are dependent and are not independent samples.
These diagnostics alone cannot establish predictive validity or authorize reward
shaping. Future multi-seed/family validation belongs in the integrated matrix.

Plan: write deterministic false/missing/conflict/reversal/censoring tests first;
implement typed pure diagnostics; test a real frozen-run adapter and source hashes;
add a one-use CLI report and documentation; execute real retrieval and modular
smokes on distinct withheld audit structures; full checks, fresh review and PR.

Risks: thresholds can be gamed by post-report tuning, class imbalance inflates
raw accuracy, absent original-correct cases leave conditional metrics missing,
and a single compression event cannot estimate sustained robustness. Keep all
sources explicit, use balanced hierarchy audit accuracy, and do not fill nulls.
