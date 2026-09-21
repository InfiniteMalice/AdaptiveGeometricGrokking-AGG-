# Architecture

The optional adaptive telemetry/control subsystem is described in
[controller architecture and extension points](controller.md). It preserves the
research pipeline below and shares its classifier evaluator and copy-based trials.

The [SoT-inspired reasoning loop](state-of-thought.md) is a separate control
timescale: compact state conditions ephemeral activation of existing valid
evidence and bounded compute proposals. It does not merge the model/depth inner
loop with the harness/audit loop. The registry retains knowledge; EventLog retains
routing history; providers alone stage candidates under protected acceptance.

The baseline `TinyTransformer` has no AGG dependency. It consumes token sequences,
uses pre-normalization causal self-attention, and classifies the last position.
An optional `AdaptedModel` applies a prototype-distance adapter to final hidden
states. Hyperbolic and product distances use Geoopt; no exp/log cancellation is
used as a substitute for actual geometric computation.

`TaskData` contains train, ID validation and structural OOD validation splits.
`train` records the zero-step checkpoint and every configured evaluation checkpoint.
Each checkpoint includes model, optimizer, batch generator and Python/NumPy/Torch
random states. `save_model` separately writes the final inference architecture and
weights without pickling a model object.

The `TelemetryCollector` stores versioned snapshots. Spectra are covariance-energy
descriptors, not an estimate of manifold dimension established by theory. Depth
finite differences use sampled positions in [0,1]. The adapter adds an observation
after the last block; interpret that extra depth sample separately from a block.
Offline graph descriptors and optional ripser persistence do not run implicitly.

The policy receives observations and proposes an explicit periodic event. The v0.1
runner executes candidates at the final training checkpoint. Intermediate periodic
events can be collected through the policy API; they are not automatically applied
inside optimizer updates. Every candidate executes on a deep copy.

Stages are ordered: accepted self-distillation; geometry/dimension trials; pruning;
component precision ladders; physical storage. Geometry alternatives share one input
checkpoint. The smallest feasible bottleneck wins, with OOD and ID behavior breaking
ties. Each component precision ladder shares its current accepted input. The lowest
feasible bit width advances to the next component. Every stage checks capability
against the **original** checkpoint, preventing tolerance accumulation.

The evaluator checks ID, OOD, linear centered-kernel alignment (CKA), finite
parameters/logits and repeated deterministic inference. CKA is a mechanism proxy,
not proof of circuit identity. The generic constraints can additionally require
execution scores; the classification runner does not pretend its unrelated toy
environment verifies classifier execution. The latter is a separate demonstration.

The ledger records measured before/after observations, candidate state, constraints,
acceptance, hashes and unavailable fields. Null means unmeasured. Logical precision
simulation and exact physical encoding are separate interfaces. Packing cannot
authorize an unaccepted precision change.

ID/OOD validation guides selection, so held-out confirmatory experiments are still
necessary. The matched continuation artifact controls the additional update budget
for one fitting intervention; a multi-stage pipeline needs a comparator matched to
its total accepted fitting updates before an acceleration claim is warranted.
