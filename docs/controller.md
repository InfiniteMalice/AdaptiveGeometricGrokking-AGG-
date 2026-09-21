# Adaptive Geometry-Guided control

The opt-in controller observes training or agent telemetry, diagnoses patterns,
and recommends one bounded intervention. It does not compute rewards or solve
the underlying task. Existing consolidation experiments keep their original
behavior. The controller reuses their ledger, evaluator and copy-based trials.

## Run the synthetic demonstration

From the repository root after the README installation steps:

```bash
python -m agg.controller.demo --output runs/controller-demo
```

The output directory must not exist. The command writes `config.json`,
`events.jsonl`, `summary.json`, and `timeline.txt`. The timeline shows healthy
progress, falling retention, rollback of a retention gain that harms transfer,
and acceptance of an alternate replay-diversity intervention. All performance
values and replay effects in this demonstration are synthetic. Its provider
changes a scalar budget; it does not implement a replay learner.

## Architecture

```text
Observation -> TemporalTelemetry -> DiagnosticProvider -> InterventionPolicy
                                                           |
                                                           v
EventLog <- Controller validation -> ExecutionProvider.stage (private candidate)
                    ^                         |
                    |                         v
               commit/rollback <- protected gates <- evaluate
```

`agg.telemetry.controller.Observation` is the canonical typed controller schema,
version `agg.controller/1`. The existing `agg.telemetry.TelemetrySnapshot` remains
version `0.1`. `Observation.from_legacy(snapshot)` maps task/ID/OOD accuracy and
loss, effective rank and memorization observations. It retains the complete
legacy record in provenance. Unmapped observations remain available there;
conversion does not invent retention measurements or reinterpret retrieval
ablation effects as retrieval-miss rates.

`agg.controller.temporal` summarizes each scalar signal on the training-step
axis. `agg.depth` continues to describe the independent layer-depth axis.
`DiagnosticProvider` returns multiple structured findings. `RuleDiagnosis` is
the initial implementation. Statistical, learned or JEV implementations can
replace it without receiving model mutation authority.

`InterventionPolicy` consumes diagnoses. `Controller.observe` records raw and
derived telemetry, diagnoses, and the proposal; it makes no training changes.
The caller explicitly chooses whether to call `start(proposal, provider)` and
`finish()`. An unsupported action returns `unsupported` before execution.

## Observe existing training

```python
from pathlib import Path
from agg.controller.config import ControllerConfig
from agg.controller.core import Controller
from agg.controller.events import EventLog
from agg.controller.training import TrainingObserver, TrainingPolicy
from agg.ledger import Ledger
from agg.tasks import modular_addition
from agg.training import TrainConfig, train

config = ControllerConfig()
controller = Controller(
    config, EventLog(Ledger(Path("runs/control/events.jsonl")), "example"),
    policy=TrainingPolicy(config),
)
observer = TrainingObserver(controller, score="accuracy")
data = modular_addition(17)
training = TrainConfig()
result = train(data, training, Path("runs/control/training"), callback=observer)
proposal = observer.latest
```

Removing the callback ablates the controller completely. A controller owns one
component and one increasing step axis. Use separate controllers for separate
layers, tasks or agents; the implementation rejects mixed component histories.
`TrainingObserver` uses the existing callback and does not change optimizer code.
For hierarchy, select `score="balanced_accuracy"` explicitly for both observer
and executor. Accuracy and balanced accuracy cannot be mixed across a trial.

## Execute a real model trial

```python
from agg.controller.training import TrainingExecutor
from agg.evaluation import Constraints
from agg.evaluation.model import make_evaluator

provider = TrainingExecutor(
    result.model, data, training,
    evaluator=make_evaluator(result.model, data),
    constraints=Constraints(),
    ledger=Ledger(Path("runs/control/model-trials.jsonl")),
    output=Path("runs/control/candidates"),
    score="accuracy",
)
if proposal is not None:
    outcome = controller.start(proposal, provider)
    if outcome.status == "pending":
        outcome = controller.finish()
    # The host adopts provider.model only after outcome.status == "committed".
```

`TrainingPolicy` proposes a bounded learning-rate increase when supported task
accuracy is flat and task loss is available. The general policy instead considers
search compute. A custom policy can propose regularization trials, but cannot
fabricate baseline metrics, redefine their configured direction, replace the
diagnosis, weaken protected metrics, or reduce evaluation requirements.

`TrainingExecutor` supports `adjust_learning_rate` and `adjust_regularization`.
It invokes the existing `trial` function, which deep-copies the accepted model,
restores RNG state and enforces `Constraints`. It invokes `train` on that copy
with a fresh seeded AdamW optimizer. This is a continuation experiment, **not an
optimizer-resume implementation**. Candidate checkpoints retain the new trial's
optimizer and RNG states. Rejection leaves the accepted model object unchanged.
Provider-level fraction bounds apply in addition to controller bounds.

The provider restores process-level deterministic settings and thread count
after the trial. Successful staging still needs the controller's target and
protected-metric checks. `make_evaluator` is shared with the existing experiment
runner; it measures ID/OOD capability, CKA, finiteness and reproducibility.

## Telemetry definitions and scope

All scalar measurements default to `None`. Absent data never becomes zero.
Ratios/frequencies use fractions, rather than percentages. Loss, reward, entropy,
norms and custom geometry measures use the producer's declared units. Record
their observation protocol and scope in `provenance`; incompatible protocols
require a separate controller or a reset of its history.

| Category | Fields and measurement expectations |
| --- | --- |
| `performance` | Task loss/accuracy/reward, validation and acquisition scores, measured improvement rate, OOD and memorization scores. Report the selected evaluation score in provenance. |
| `geometry` | Representation drift, gradient/update norms, update cosine, effective rank, singular values, named layer statistics and extensible named metrics. Name metric-specific units in provenance. |
| `attribution` | Concentration, entropy, layer distribution, drift, importance changes and causal contributions supplied by an attribution producer. Concentration alone does not establish a causal shortcut. |
| `gating` | Activation and saturation fractions, context/attention/memory horizons, THINK frequency and named model/tool routing frequencies. Horizon units come from the producer. |
| `resources` | Sparsity, zero density, active parameter fraction, adapter/LoRA utilization, precision mode, quantization state and physical storage encoding. Precision and encoding are distinct labels. |
| `continual` | Acquisition, retained performance, forgetting, retention slope, memory half-life, replay effectiveness/diversity, run variance, capability regression, exact retention, near transfer and far transfer. Only actual reevaluation supplies retention. |
| `abstraction` | Coverage/confidence/predictive utility, false applicability, retrieval misses/false positives, composition failures, contradictions, regression and per-record lifecycle status. |
| `search` | Branch diversity, depth, exploration budget, exploit/explore ratio, marginal gain per compute, dead ends, rollbacks, intervention effectiveness, difficulty, uncertainty and risk. |

`Observation.metrics()` includes scalar leaves of numeric maps, including named
layer statistics. Singular-value tuples and categorical labels remain raw
observations. Annotate proxy scalar names with `proxy_metrics`. Quality flags
`contradictory` and `out_of_distribution` yield `unknown_anomaly` and break
temporal support. Changing a metric between proxy and measured also resets its
support; one measured observation cannot legitimize a proxy history.

Temporal summaries include current value, exponential moving average, rolling
mean/variance, adjacent first difference, fitted first derivative, fitted second
derivative, residual noise, sample count, heuristic confidence and numeric trend.
The quadratic least-squares fit uses at least five observations by default and
centers/scales the actual step coordinates. Derivative units are metric units per
step and per step squared. Trend labels describe numeric increases/decreases;
the acceptance gate uses metric direction separately.

Missing signals reset their series. A gap exceeding `max_gap` resets a series.
The confidence score combines sample support and fit residuals; it is **not a
calibrated probability**. Repeated deterministic evaluations are not independent
scientific replications. Interventions and statistical claims need separate
evaluation protocols and new seeds.

## Configuration and acceptance

Load JSON using `ControllerConfig.from_dict`. `configs/controller.json` contains
the default values. Temporal thresholds, sample/window sizes, noise threshold,
confidence thresholds, cooldown, target gain, protected tolerances, rollback
threshold and diagnosis thresholds are centralized in `config.py`.

`minimize_metrics` declares lower-is-better target metrics; other proposed targets
are higher-is-better. Extend that configuration for new error metrics. Merely
observing a signal does not select it as an optimization objective.

Each `MetricGuard` has its own direction and absolute regression tolerance.
The default protects OOD accuracy with tolerance `0.02`. Hosts must configure
their actual general-capability, transfer, stability, safety and prior-stage
metrics; the package cannot manufacture missing evaluations. A negative-control
metric such as false applicability uses `higher_is_better=False`.

Acceptance requires all of the following:

1. The proposal is the latest unmodified issuance, with adequate confidence.
2. No other intervention is pending, cooldown has elapsed, and the action and
   target are allow-listed and supported by the provider.
3. The provider completes the configured step window and sample count.
4. Direction-adjusted target gain strictly exceeds `minimum_gain`.
5. Every protected metric is finite and within its tolerance against both the
   local baseline and the fixed capability anchor captured at first execution.

Checks 1–2 precede staging; checks 3–5 precede commit. Missing protected evidence
at the end of the window causes rollback. Measured protected regression can
cause earlier rollback. Target deterioration beyond `rollback_threshold` also
causes early rollback. Missing evidence during an unfinished window yields
`pending`. The training provider schedules enough checkpoints for the requested
sample count; it rejects a count exceeding available training steps.

The controller consumes each attempted issuance once. Cooldown starts at the end
of either commit or rollback. A failed replay proposal causes the policy to try
replay diversity next; after both fail for that diagnosis, it observes more.
Failure memory is per controller instance. A failed rollback sets `blocked` and
requires host/human recovery; creating a replacement controller does not repair
provider state automatically.

## Memory, retrieval and curriculum extension points

`AbstractionRegistry` stores multiple global, cluster and local records in the
same block. Conditions are conjunctions of exact field matches. Any matching
exclusion vetoes application. Active records above the configured confidence
threshold can overlap; an empty result is valid. Experimental, retired and
superseded records are retained but excluded from retrieval. Registry APIs return
copies, and lifecycle changes produce new version numbers.

`record_applicability` records negative controls. False applicability is the
fraction of labeled should-not-apply cases where a record was selected, not the
fraction of all queries. `RetrievalEvidence` requires externally established
correct/invalid IDs; the controller does not infer ground truth from retrieval.
Its findings separate misses, false positives, bad knowledge, composition failure,
low separation, poor specificity, poor faithfulness, ambiguity and uncovered queries.
`RetrievalIndexProvider.revise_keys` explicitly separates retrieval keys from
knowledge content. There is no live indexing backend in this repository.

`CompositionEvaluator` compares individual and ordered composed success.
`CompositionResult` records interface, order and nonlocal dependency failures.
`prioritize_compositions` uses explicit dependencies first, then observed
coactivation frequency, with a bounded budget. It never enumerates all subsets.

`CurriculumGate` evaluates every configured stage up to the current stage. Missing
required scores, nonfinite scores, or scores outside `[0, 1]` produce `hold` and
appear in `missing_stages`. Otherwise, any failing stage yields
`backtrack` to the shallowest failure. All passing stages permit `advance`, or
`complete` at the last stage. This is a decision interface; a host curriculum
runner is responsible for executing it and reevaluating before advancement.

`RetentionTracker` maintains named curves so exact, near and far transfer remain
separate. It reports peak-to-current forgetting and linear retention slope. With
sufficient positive declining observations, it estimates half-life from a
log-linear exponential fit. The estimate has no causal or calibrated predictive
guarantee. `ContinualLearningRepairProvider` reserves detect/propose/evaluate
hooks for future repair methods, including JANUS-style work.

`ComputeAllocationPolicy` receives difficulty, uncertainty, progress derivatives,
diversity, marginal gain and risk through observations. `SlopeComputePolicy`
recommends bounded budget directions; high risk can terminate in human review.
It is an interface for future System-1/System-2 routing, not an inference engine.

## Supported actions and authority

| Capability | Initial support |
| --- | --- |
| No-op, observe-more, human review | Recorded controller outcomes; no mutation |
| Learning rate, regularization | Real isolated training provider |
| Replay/diversity and reasoning/exploration budgets | Numeric allow-list plus provider protocol; replay demo is explicitly simulated |
| Curriculum and abstraction changes | Decision/registry interfaces, not an automatic host executor |
| Evaluation frequency, retrieval/index mutation, composition testing, parallelism, adapter allocation/merge, GRN, clipping, batch mix, checkpoint/explicit rollback, regression-suite launch and repair | Enum/interface only; `start` reports unsupported |

To add execution, implement a trusted `ExecutionProvider` with `stage`, `evaluate`,
`commit`, `rollback`, `finalize` and an action set. The current controller permits only its
explicit numeric configuration allow-list. Structural actions require a reviewed
validator and allow-list extension, not merely adding a provider action name.

Providers stage private state, snapshot the current accepted state before a
fallible attempt, and retain rollback ability through commit and outcome logging.
After recording a successful commit, the controller calls `finalize` to release
the transaction snapshot and candidate metadata. A later `rollback` must leave
the accepted state intact. If outcome logging fails, the controller rolls back
before finalization. If finalization itself fails, the audited commit remains
accepted, the error propagates, and the controller blocks further interventions
until the host reviews the provider's state.
They must report actual execution and evidence; an unsupported action is not a
successful no-op. The host owns evaluator, protected suites, reward computation,
ground truth, security/sandbox policy and audit storage. These objects are absent
from the controller's mutation allow-list.

Runtime/orchestration budgets, inference engines, the evaluation harness, tool
environments and the host are separate trust boundaries. The current provider
is trusted in-process Python, single-owner and synchronous. Use external process
isolation for untrusted adapters; this API is not an OS sandbox. JSONL is
append-only through the API, not tamper-proof against host filesystem access.
Audit errors propagate; the controller never reports a successfully audited
transaction if writing the final audit failed.

## Explicit invariants

Telemetry is not automatically reward. Diagnosis and execution are separate.
Multiple diagnoses and abstractions can coexist. Unknown and no applicable
abstraction are valid outcomes. Intervention strength scales with evidence;
expensive actions require the higher confidence threshold. Candidates remain
bounded and reversible. Target gains cannot excuse protected regression.
Individual learning does not establish composition. Retrieval failure does not
establish knowledge failure. Curriculum advancement checks prior capabilities.
Human escalation is a normal terminal outcome.

## Tests and remaining research

Run `python -m pytest -q`, `python -m ruff check .`, and `python -m mypy agg`.
`tests/test_controller_*.py` exercises temporal mathematics, data quality,
diagnosis, proposals, serialization, mutation boundaries, cumulative regression,
rollback failures, memory, curriculum, negative controls and real model isolation.
The synthetic integration test verifies the four-phase controller sequence.
It does not show that replay improves a real model.

Next experiments, with fixed budgets and untouched confirmation seeds:

1. Compare absolute observations with first/second-derivative control, measuring
   intervention precision, unnecessary interventions and protected regressions.
2. Compare monotonic curricula with retention-gated backtracking, measuring
   acquisition time, earlier-stage retention and compute cost.
3. Compare one block summary with multiple scoped abstractions, using positive
   applicability, negative controls, transfer and composition evaluations.

Remaining engineering work includes live replay/curriculum/index/adapter providers,
optimizer-state resume, calibrated diagnosis, crash-resumable controller state,
durable provider transactions, and host process isolation. No empirical learning
improvement is claimed by this implementation.
