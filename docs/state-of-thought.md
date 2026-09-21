# State-of-Thought-inspired AGG control

Gong et al. propose that a compact internal reasoning state can condition
historical evidence activation and continuation in frozen autoregressive models.
Their four coordinates describe dispersion, displacement, direction and predictive
uncertainty. Their paper uses a learned 582-parameter controller. AGG borrows the
state-conditioned control idea, not that controller or its reported results.
See the [paper](https://arxiv.org/abs/2609.16055) and [references](references.md).

This extension is **SoT-inspired**. AGG's current `TinyTransformer` classifies
token sequences; it is not an autoregressive LLM reasoning engine. The utilities
can compute readouts from supplied representations/logits. A future LLM/agent
host can instead provide `Reasoning` telemetry with its own units, sampling
protocol and proxy labels. This implementation does not automatically instrument
the toy model, generate reasoning traces, learn evidence profiles or train a router.

## Three distinct loops

```text
INNER LOOP
model/depth geometry -> AGG telemetry -> bounded gating/precision/sparsity proposals

REASONING LOOP
Reasoning_t -> valid evidence activation -> host's next state / compute recommendation

HARNESS / AUDIT LOOP
observations + routing + proposals + outcomes
    -> anomaly evaluation / protected metrics / private trials / commit or rollback
```

The complete AGG context also includes geometry, temporal first and second
derivatives, attribution, horizons, resources, abstractions and search. A router
names the coordinates it consumes. The implementation does not concatenate all
telemetry into a learned state. Depth derivatives use layer coordinates; the
existing `TemporalTelemetry` uses `Observation.step`. A host using reasoning
steps documents those units and uses a separate controller per component.

## Schema and readout definitions

New `Observation` instances serialize as `agg.controller/2`. `from_dict` explicitly
migrates `agg.controller/1` (also the historical versionless form), filling the
new category with `None`. A v1 payload containing `reasoning` is rejected. The
explicit legacy constructor `Observation(..., schema_version="agg.controller/1")`
can still write the original shape when reasoning is empty. Unknown versions fail.
The separate legacy `TelemetrySnapshot` schema remains `0.1`.

`Reasoning` is frozen and contains the four optional coordinates plus
`active_evidence_fraction`, `evidence_turnover` and signed `reasoning_progress`.
Dispersion, displacement and entropy are nonnegative; cosine is in [-1,1].
Only the two evidence fractions are constrained to [0,1]. Missing values remain
`None`; NaN, infinity and booleans are rejected. Entropy/progress are not
implicitly normalized. Selected IDs belong in routing events, not scalar telemetry.

`agg.telemetry.reasoning` supplies pure, deterministic reference readouts:

| Function | Input shape | Definition and units |
|---|---|---|
| `representation_dispersion` | `[items, features]` | Mean squared Euclidean distance to centroid; squared representation units; no dimension normalization |
| `state_displacement` | Two matching `[features]` states | Euclidean distance; representation units |
| `trajectory_cosine` | Two matching `[features]` transitions | Cosine of successive differences, not absolute states; zero transition returns `None` |
| `predictive_entropy` | `[..., classes]` logits | Mean categorical softmax entropy over leading axes, in nats; no temperature or division by log(class count) |

Extraction detaches tensors, computes on CPU in float64, rejects empty or
nonfinite inputs and rejects unrepresentable results. It changes neither inputs
nor gradients. Hosts choose comparable representations across steps and document
that choice. These Euclidean readouts are reference definitions, not a claim of
matching the paper's information-transfer representation pipeline.

## Activation, relevance and validity

`AbstractionRegistry` retains its current retrieval and negative-control APIs.
Its `snapshot()` returns private copies of every record, including inactive ones.
`AuditedEvidenceRouter` combines that snapshot with registry applicability and an
external `EvidenceValidity` attestation. Only IDs both applicable and explicitly
valid are eligible. Invalid, unknown, unavailable and ineligible evidence stays
visible in the audit. A valid but irrelevant record can remain unselected.

The validity source is a host trust boundary, not a score produced by the router.
Hosts continue using `RetrievalEvidence`, independently labeled positive and
negative controls, and `diagnose_retrieval` to evaluate relevance and validity.
Selection neither establishes truth nor changes confidence, status, proposition,
conditions or registry negative-control counts. **Retrieval failure is not
knowledge failure.** A selector can return no evidence.

`EvidenceRouter` implementations receive `ControlContext` and immutable candidate
IDs/profiles, never a model or registry. Rules, learned small controllers, JEV
classifiers and neural routers can implement this protocol in future work.
The reference implementation uses explicit feature scales and the score
`1 / (1 + scaled Euclidean distance)` to host-supplied relevance profiles.
Every required coordinate must be present in both the control and profile;
otherwise that candidate has score `None` and the reference router abstains.
It applies `minimum_score` and `max_active`; ties use sorted IDs.
The default feature is predictive entropy at scale 1 (nats). Change the feature
scales when producer units differ. No scale calibration is learned.

## Minimal host integration

```python
from dataclasses import asdict, replace
from pathlib import Path
from agg.controller.config import ControllerConfig, ReasoningPolicyConfig
from agg.controller.core import Controller
from agg.controller.events import EventLog
from agg.controller.memory import Abstraction, AbstractionRegistry
from agg.controller.routing import AuditedEvidenceRouter, EvidenceValidity, ReferenceRouter
from agg.controller.routing_diagnostics import diagnose_routing
from agg.ledger import Ledger
from agg.telemetry.controller import Observation, Performance, Reasoning

events = EventLog(Ledger(Path("runs/reasoning/events.jsonl")), "host-example")
registry = AbstractionRegistry(events=events)
registry.add(Abstraction("example", "host-validated rule", "task", confidence=0.9))
router = AuditedEvidenceRouter(registry, ReferenceRouter(), events)
controller = Controller(ControllerConfig(reasoning=ReasoningPolicyConfig(enabled=True)), events)
observation = Observation(
    0, performance=Performance(task_accuracy=0.7, ood_score=0.8),
    reasoning=Reasoning(predictive_entropy=0.4, state_displacement=0.2, trajectory_cosine=0.9),
    provenance={"source": "host measurement", "step_units": "reasoning iteration"},
)
decision = router.route(
    observation, block="task", context={},
    validity=EvidenceValidity(("example",), source="external validation protocol/version"),
    profiles={"example": {"reasoning.predictive_entropy": 0.5}},
)
diagnostic = diagnose_routing(decision)  # supply previous and earlier for turnover/oscillation
proposal = controller.observe(replace(
    observation, reasoning=diagnostic.telemetry(observation.reasoning),
    provenance={**observation.provenance, "routing_id": decision.id},
))
router.record_downstream(decision, {"proposal": proposal.to_dict(),
                                   "routing_diagnostics": asdict(diagnostic)},
                         proposal_id=proposal.id)
# The first observation has insufficient temporal support: observe more.
# The host decides whether to use decision.selected_ids in its next context.
# Execution, if supported, still requires controller.start(...) and finish().
```

Provenance/IDs remain event data. The host owns evidence application; this example
does not alter the stored abstraction or execute a proposed model intervention.

## Motivated-forgetting risk and audit guarantees

State-conditioned selection can drop conflicting evidence, reducing constraints
on later states and reinforcing the same selection pattern. This implementation
makes that sequence inspectable; it does not prevent motivated forgetting or
detect deception. The audit system retains evidence identity even when a host
stops attending to the evidence.

Before returning a selection, the harness writes an `EvidenceRoutingEvent` using
the existing `EventLog`/`Ledger`. It records the complete eligible set, selected and
dropped IDs, invalid/unknown/unavailable/ineligible sets, original observation,
effective control, profiles, scores/rationales, temporal summaries, provenance,
retrieval configuration, router configuration and all record ID/version/SHA-256
references. The full original payloads are not duplicated. Hosts retain the
referenced record versions to reconstruct evidence content; hashes alone cannot
recover deleted payloads. `record_downstream` appends an outcome linked by routing
ID and optional proposal ID. An unavailable audit sink raises before returning
active IDs. Invalid plugin selections are rejected and logged.

The decision stores an immutable canonical JSON snapshot; decoded dictionaries
are copies. The JSONL transport is append-only through this API, **not tamper-proof,
concurrent, crash-resumable or write-once storage**. An untrusted Python plugin
requires external isolation; protocols do not sandbox arbitrary code.

## Evidence dynamics and bounded computation

`diagnose_routing` reports active/eligible fraction and Jaccard turnover between
adjacent active sets. First-sample turnover and empty-eligible fraction are `None`;
two measured empty active sets have turnover zero. Three decisions reveal
drop/re-add oscillation. Other findings cover low entropy with narrow support,
rising entropy while support shrinks, direction reversal with removal, possible
premature stopping, and disagreement with externally supplied semantic continuity.
Thresholds are explicit, heuristic, and uncalibrated. Pass results to evaluation
or record them in a linked outcome. They are not deception labels or rewards.

`ControllerConfig.reasoning.enabled` defaults to false. Enabling it selects
`ReasoningComputePolicy` for supported progress/compute diagnoses. Other diagnostic
families keep their existing priority and proposal paths. The policy requires
supported task progress plus measured entropy, displacement, direction and support.
Missing, contradictory, out-of-distribution, unstable or confidence-capped proxy
evidence causes observation; high supplied risk requests human review. Narrow or
unstable evidence support requests alternate retrieval. High entropy can recommend
more budget; low entropy, small displacement and nonpositive measured progress
can recommend less budget. `allow_stop=True` permits a stop recommendation.

Entropy is telemetry, not a target metric. Proposals still target task accuracy,
retain protected metrics and use existing fraction bounds, cooldown, allow-lists,
provider authority, private trials and acceptance. No new execution provider is
installed. Current training providers support learning rate/regularization only;
reasoning-budget changes, stop/continue and alternate retrieval report
`unsupported` when no authorized provider exists. Recommendations are not successes.
Budget savings alone do not meet the existing task-gain acceptance rule; this
extension preserves that rule instead of introducing a new efficiency objective.

## Causal ablation hooks

`reasoning_ablations` changes only immutable control copies. Audits retain the
original observation and a separate labeled control intervention.

| Control | Hook |
|---|---|
| Coordinate zeroing | `zero_coordinates`; missing coordinates stay `None` |
| Coordinate permutation | `permute_coordinates`; seed-shuffle each coordinate across samples, preserving its units/domain |
| Temporal scrambling | `scramble_temporal`; seed-shuffle complete controls against fixed observed steps |
| Dynamics disabled | `dynamics_disabled`; remove displacement/direction and their derivatives |
| Absolute state only | `absolute_state_only`; additionally remove all temporal derivatives |
| Random evidence | `RoutingConfig(mode="random", seed=...)`; fresh local RNG per call holds draw fixed in paired replay |
| State independent | `mode="state_independent"`; sorted eligible IDs up to budget |
| Fixed active evidence | `mode="fixed", fixed_ids=(...)`; intersect fixed IDs with currently valid/eligible evidence |
| Full history | `mode="full_history"`; all valid eligible IDs, explicitly ignores `max_active` |
| Counterfactual replay | `counterfactual_replay`; fixed observation/store/profiles, labeled perturbed control |

Changed coordinates lose their old derivative features. Temporal scrambling
deliberately breaks chronology and keeps moved control features; it does not
pretend to recompute a coherent trajectory. Hosts can recompute derivatives for
different interventions. Permutation across samples avoids swapping cosine into
entropy or mixing incompatible units. This is a deliberate domain-preserving
interpretation of coordinate permutation.

Hold registry versions, validity evidence, profiles and random draws fixed in
paired comparisons. A changed selection demonstrates a causal dependency inside
this reference router, not that the telemetry improves real reasoning.

## Reproducible synthetic comparison

```bash
python -m agg.controller.reasoning_demo --output runs/sot-demo
```

Choose a new directory. The command writes four routing event logs and
`summary.json` with full traces/configuration. The oracle labels next-step entropy
as high/low on a deterministic trigonometric trajectory with deliberate shifts
at steps 20 and 40. A uses current entropy; B adds AGG's fitted slope; C also adds
half its fitted acceleration for a one-step forecast; D selects independently of
state. The explicit forecast is a proxy, separate from observed entropy.

Metrics: precision is the proportion of oracle-matching selections; stability is
one minus mean Jaccard turnover; unnecessary changes are evidence switches with
unchanged oracle labels. Compute usage is a declared proxy counting coordinate
reads plus active evidence, not tokens, FLOPs or latency. Recovery is steps until
two consecutive correct selections after each shift (or `None`). No model or
provider runs here, so protected regressions and unnecessary interventions remain
`None`. Existing lifecycle tests independently exercise protected rollback.

The fixture is designed around a forecastable signal and is not a realistic
reasoning benchmark. Results neither establish endogenous reasoning in LLMs nor
prove that dynamics cause generalization. Larger causal experiments need calibrated
readouts, independently retained evidence, negative controls, matched budgets,
held-out tasks/models and semantic/outcome evaluation.

## Verification map

`test_controller_reasoning.py` covers schemas, missingness, ranges and numerical
readouts. `test_controller_routing.py` covers selection response, registry
immutability, complete audit, invalid output, anomalies and causal replay.
`test_controller_reasoning_ablations.py` covers perturbation domains/missingness.
`test_controller_reasoning_control.py` covers opt-in policy and protected lifecycle.
`test_controller_reasoning_demo.py` checks reproducible artifacts and real feature
consumption. Run these alongside all pre-existing lifecycle/rollback tests, Ruff
and mypy. Passing tests establishes engineering behavior only.
