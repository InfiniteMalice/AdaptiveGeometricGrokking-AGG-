# Adaptive Geometry-Guided controller specification

The September 21 user brief requests a telemetry and control subsystem. The
existing package's research program remains intact. The controller does not
compute rewards or solve training tasks.

## Repository inventory

* `agg/training/loop.py`: deterministic supervised cross-entropy/AdamW loop,
  evaluation callback, model/optimizer and RNG checkpoints. No RL objective,
  PPO, GRPO, DAPO, GEPA or GRN implementation exists.
* `agg/telemetry`: version 0.1 category snapshots and JSONL history. Existing
  depth derivatives measure position across layers, not change over time.
* `agg/experiments/diagnostics.py`, `agg/attribution`, `agg/dimension`,
  `agg/gating`, `agg/horizon`: observational probes, spectra, CKA, gates and
  retrieval proxies. The geometry adapter is not a LoRA implementation.
* `agg/consolidation/trials.py`: copy-based candidate evaluation, RNG restoration,
  independent constraints and rejection records. Reuse for model interventions.
* `agg/evaluation`: ID/OOD/mechanism constraints. `agg/ledger`: append-only JSONL.
* `agg/experiments/checkpoints.py`: versioned inference artifacts. Training
  checkpoints and inference artifacts have different resume semantics.
* `agg/experiments/config.py`: dataclass configuration; `tests`: pytest, Ruff,
  mypy. No replay trainer, procedural memory registry, curriculum runner,
  learned diagnostic classifier or continual-learning repair backend exists.

## Design and behavior

Add an opt-in `agg.controller` package and typed category records under
`agg.telemetry.controller`. Preserve version 0.1 telemetry and provide explicit
conversion; do not silently reinterpret proxies as measured causal evidence.
Temporal features use rolling least-squares trends on actual step coordinates.
Quadratic fits require multiple observations. Missing signals never become zero.
Rule diagnosis can return multiple findings, insufficient evidence or an unknown
anomaly. Replaceable providers permit later statistical or learned diagnosis.

The controller owns the acceptance policy. A proposal cannot weaken its protected
metrics, evidence thresholds, action limits or evaluation window. Only one trial
can be pending. Cooldown starts after execution, including rejected trials.
Execution providers stage private candidate state, evaluate through a trusted
evaluator, and commit only after target and protected constraints pass. The
existing model trial additionally enforces its original behavior/mechanism gates.
Log proposals before validation, and retain rejections, failures and rollback.
Evaluation failure fails closed. Rollback failure blocks subsequent mutations.

Real model execution initially supports bounded learning-rate and regularization
changes in a fresh, seeded continuation trial using the existing `train` and
`trial` functions. This does not resume optimizer state or mutate an in-flight
training run. Other enum actions require an explicit provider. A synthetic budget
provider exists only in the demo and tests; its replay actions are simulations.

Registry entries have independently scoped conditions and exclusions; overlapping
entries and no match are valid. Retrieval evidence separates missing selection,
incorrect content and composition failure. Curriculum evaluation selects the
shallowest failing stage and holds when required measurements are missing.
Repair and compute allocation expose typed protocols without implementing JANUS,
JEV, System-1 routing or a new agent runtime.

## Verification and risks

Tests cover analytic derivatives with irregular spacing, noise, missing/nonfinite
data, simultaneous diagnoses, cooldown, stale/forged proposals, missing protected
evidence, target direction, rollback and provider failures, cumulative tolerance,
scoped retrieval and composition. The four-phase synthetic integration fixture
must reject retention gains that harm transfer and accept a safe alternative.
Run the existing full suite, Ruff, mypy, a synthetic demo and a real short training
trial. Rule thresholds and confidence are configurable heuristics, not calibrated
probabilities. Proxy validity and independent confirmatory evaluation remain
experimental questions. Host, runtime, inference engine, evaluator/harness and
tool environment remain distinct trust boundaries; Python protocols are not an
OS sandbox for malicious provider code.
