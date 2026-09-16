# Supervision and evidence hierarchy

`agg.supervision.Signal` records source, value in `[-1,1]`, confidence in `[0,1]`, temporal position and nonempty provenance. Confidence is supplied by the observer; recording it does not calibrate it. Preserve the original signals alongside any combined value.

The default precedence is:

1. `external_outcome`
2. `environment_state`
3. `execution_log`
4. `task_verifier`
5. `ground_truth`
6. `teacher`
7. `process`
8. `self_report`

`EvidencePolicy` allows ordinary precedence to be configured. High-confidence external outcomes and environment states retain explicit priority over auxiliary sources even if ordinary precedence changes. Among eligible signals, source order and then confidence choose a primary signal. Its value is confidence-weighted. Total auxiliary influence is capped relative to the absolute primary value; adding many teacher signals does not multiply the allowed cap. Opposing source names are retained as conflicts. Signals with zero confidence contribute no usable evidence.

The protected high-confidence override currently applies specifically to `external_outcome` and `environment_state`. Do not describe every signal called a verifier as automatically protected. A verifier can be wrong or supplied with the wrong target, so record its method and calibration separately.

`ToyEnvironment` actually executes integer add, multiply, set and exact divide operations. `ExecutionEvidence` records intent, action, argument, before/after states and hashes, result/error, expected state, verifier outcome/confidence/provenance, and logical order. Failed operations retain their prior state. Intent text has no authority over the observed result. Hashes identify state content; they are not external attestations.

`assign_credit` first combines evidence at each position, then computes discounted returns. Gamma zero yields local evidence; gamma one includes all future evidence without discount. Future evidence is retained in each credit record. This is a transparent credit-assignment demonstration. No claim is made that discounted correlation proves causal contribution, that the demonstration trains an agent, or that it reproduces a particular OPD algorithm.

Ground-truth-anchored distillation is a separate training API. Teacher logits and representations are detached; ground-truth cross-entropy remains present. Generated process explanations are diagnostic or bounded auxiliary data, never privileged ground truth. Test conflicts using deliberately incorrect teacher/process fixtures and independently observed failed actions.
