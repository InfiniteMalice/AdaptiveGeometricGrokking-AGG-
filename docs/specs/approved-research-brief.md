# AdaptiveGeometricGrokking (AGG)

Create a new open-source research repository named **AdaptiveGeometricGrokking**, abbreviated **AGG**.

AGG investigates whether deliberate consolidation can accelerate or induce grokking, discover simpler internal representations, adapt representation geometry and intrinsic dimensionality to learned structure, and then compress the resulting computation as far as evidence allows without materially damaging generalization or the learned mechanism.

The guiding principle is:

> **Understand → Restructure → Compress → Verify**

Compression is subordinate to capability preservation. Nothing is automatically pruned, dimension-reduced, quantized, sparsified, or ternarized merely because doing so is possible.

AGG should distinguish:

* what the model predicts;
* what internal telemetry suggests;
* what an intervention causally changes;
* what actually happened during execution.

Telemetry informs interventions. **Do not optimize every telemetry metric directly.**

Treat this prompt as the approved high-level scientific design. Use the Superpowers development workflow where available: specification, implementation planning, TDD, verification before completion, and small understandable commits.

Do not stop for cosmetic decisions. Stop only for a genuine architectural contradiction or blocker.

---

# 1. Primary hypothesis

The primary hypothesis is:

**Explicit consolidation can accelerate the transition from memorized/local solutions toward compact, reusable, generalizing invariant representations.**

Secondary hypotheses:

* Learning may pass through distinguishishable memorization, retrieval/reuse, and invariant-generalization regimes.
* Grokked representations may be more compressible than equally capable memorizing representations.
* Geometry/topology changes may precede or accompany behavioral grokking.
* Different structures may require different intrinsic dimensionality and geometry.
* Gating may determine whether a useful representation actually participates in computation.
* Epiplexity or compute-bounded description-length measures may help identify useful consolidation opportunities.
* Numerical precision requirements vary by component.
* Physical storage layout should be chosen separately from logical numerical precision.
* Sparsity does not automatically imply sparse physical encoding is efficient.
* Real execution outcomes and environment/tool states can provide stronger causal supervision than generated reasoning or teacher explanations.
* Discounted temporal credit may help connect delayed verified outcomes to earlier causal decisions without treating every intermediate generated explanation as equally trustworthy.
* Geometric change across network depth may have useful first- and second-order differential signatures.

Do not claim any hypothesis is established until experimentally supported.

---

# 2. Three AGG phases

## Phase A — Understand

Goal:

`memorization → retrieval/reuse → reusable invariant/generalizing computation`

Mechanisms:

* ordinary training;
* grokking measurement;
* controlled memorization/retrieval benchmarks;
* experience/checkpoint collection;
* AGG telemetry;
* attribution;
* geometry/topology telemetry;
* consolidation triggers;
* self-distillation;
* executable causal supervision;
* temporal credit assignment;
* epiplexity / description-length analysis;
* ID and OOD testing.

## Phase B — Restructure

Goal:

`generalizing computation → efficient internal representation`

Mechanisms:

* intrinsic-dimension estimation;
* adaptive bottleneck dimension;
* Euclidean representation;
* hyperbolic representation;
* Euclidean × hyperbolic product representation;
* curvature adaptation;
* gating;
* effective-horizon analysis;
* low-rank analysis;
* depthwise geometric derivatives;
* representation reconstruction/distortion measurement.

## Phase C — Pack

Goal:

`efficient computation → minimal physical implementation`

Mechanisms:

* parameter pruning;
* structured/unstructured sparsity;
* rank reduction;
* adaptive weight precision;
* adaptive activation precision;
* accumulator/computation precision;
* ternarization where empirically justified;
* physical storage selection;
* bit packing;
* BITCOS-like ternary sparse encoding;
* hardware benchmarking.

---

# 3. Core optimization constraint

Formulate compression as constrained optimization.

Conceptually:

`minimize representation/deployment cost`

subject to:

* ID capability degradation ≤ configured tolerance;
* OOD degradation ≤ configured tolerance;
* verified execution behavior preserved where relevant;
* numerical stability preserved;
* configured mechanistic-preservation threshold satisfied;
* reproducibility tests pass.

A candidate violating constraints must be rejected and the previous accepted checkpoint restored.

**Compression failure is a valid scientific result.**

Avoid collapsing all AGG signals into one scalar loss.

---

# 4. AGG telemetry vector

Make the **AGG telemetry vector** a central, versioned data structure.

At checkpoint/layer/component granularity, telemetry should be capable of containing:

`T_AGG = [`

* task loss and task metrics;
* ID and OOD metrics;
* attribution information;
* UTS-style geometric/topological descriptors;
* intrinsic-dimension estimates;
* first depth derivatives;
* second depth derivatives;
* gate statistics;
* effective-horizon statistics;
* memorization score;
* retrieval/reuse score;
* generalization score;
* epiplexity / description-length measurements;
* sparsity state;
* precision state;
* physical-storage state;
* execution/verifier outcomes where applicable;

`]`

Store both:

* raw values;
* normalized/derived values.

Do not obscure raw measurements behind a single learned score.

### Critical architectural rule

**Intervention policies consume telemetry. They do not automatically optimize every telemetry component.**

For example:

* a curvature shift may trigger investigation;
* an attribution shift may indicate a possible phase transition;
* falling effective dimension may suggest a compression candidate;
* gate saturation may motivate an ablation;
* rising zero density may motivate testing sparse storage.

None of those observations are themselves mandatory optimization targets.

This separation is important to prevent AGG from Goodharting its own diagnostics.

Create interfaces conceptually similar to:

`TelemetryCollector`

`TelemetrySnapshot`

`TelemetryHistory`

`InterventionPolicy`

An `InterventionPolicy` may read telemetry and propose an experiment.

It must not silently mutate the model.

---

# 5. Baseline model

Use a small, readable PyTorch Transformer suitable for grokking research.

Do not begin with an LLM.

The initial model must be small enough that:

* full trajectories can be recorded;
* many seeds are affordable;
* checkpoints are cheap;
* representations can be analyzed exhaustively;
* attribution experiments are practical;
* compression sweeps are practical;
* geometry/topology telemetry is practical.

Keep the baseline model independent from AGG-specific mechanisms wherever possible.

---

# 6. Initial task suite

Implement at least three experimental families.

## A. Modular arithmetic

Begin with modular addition.

Design task interfaces for later:

* subtraction;
* multiplication;
* composition;
* related algorithmic tasks.

This is a conventional grokking environment and should not be assumed to favor hyperbolic geometry.

## B. Synthetic hierarchy

Generate controlled trees/taxonomies.

Possible tasks:

* parent/child;
* ancestry;
* hierarchical distance;
* lowest common ancestor;
* taxonomy completion.

This gives hyperbolic representations an appropriate test rather than assuming every task should benefit from negative curvature.

## C. Controlled memorization→retrieval benchmark

Create a benchmark expressly designed to separate memorization from retrieval/reuse.

Sweep at least four independent variables:

`gate strength × relevant distance × distractor density × context length`

### Gate strength

Control how strongly the retrieval/adapted path contributes to the final computation.

### Relevant distance

Vary the distance between the query and the information actually required for the correct answer.

### Distractor density

Control the number and similarity of irrelevant candidate memories/context elements.

Include both:

* random distractors;
* hard/semantically similar distractors.

### Context length

Increase total context while independently controlling relevant distance and distractor density.

Do not conflate context length with retrieval distance.

Use a factorial or carefully sampled design that permits effects and interactions to be measured.

---

# 7. Memorization→retrieval phase-transition measurement

Do not measure only final accuracy.

The key quantity is:

> **At what training step does retrieval/reuse become more explanatory of model behavior than memorization?**

Create separate evidence scores:

`M(step) = memorization evidence`

`R(step) = retrieval/reuse evidence`

`G(step) = invariant/generalization evidence`

These need not sum to one.

Candidate measurements include:

* nearest-neighbor agreement;
* training-example representation similarity;
* individual-example influence;
* kNN prediction agreement;
* relevant-context ablation;
* distractor ablation;
* relevant-memory replacement;
* counterfactual memory replacement;
* attribution to relevant versus memorized features;
* OOD structural generalization;
* interpolation/extrapolation gaps;
* performance when surface similarity conflicts with the true rule.

Define a **retrieval-over-memorization crossing** operationally.

For example, a candidate phase-transition point may require:

`R(step) > M(step) + margin`

for a configured number of consecutive evaluations.

Do not rely on a single noisy crossing.

Record:

* first crossing;
* stable crossing;
* uncertainty interval;
* seed variability.

For the full benchmark estimate:

`step_retrieval(gate, distance, distractors, context)`

This produces a phase surface rather than a single scalar.

Analyze interactions such as:

* stronger gate reducing retrieval onset time;
* relevant distance delaying retrieval;
* distractor density delaying retrieval;
* context length harming retrieval independently of distance;
* gate strength compensating for distance only up to a limit.

Negative or non-monotonic effects are scientifically important.

---

# 8. Effective-horizon telemetry

Create an empirical **effective horizon** measure.

The effective horizon should estimate how far through context, depth, or temporal sequence a signal remains causally useful.

Possible measurements:

* maximum relevant distance at which retrieval remains above threshold;
* decay of attribution to relevant information as distance increases;
* accuracy loss as relevant information is moved farther away;
* counterfactual sensitivity versus distance;
* gate-weighted contribution versus distance;
* temporal-credit decay;
* context ablation sensitivity.

Do not equate nominal context length with effective horizon.

Log:

* nominal horizon;
* retrieval horizon;
* attribution horizon;
* causal/intervention horizon.

Investigate whether grokking increases effective horizon or merely changes how efficiently the horizon is used.

---

# 9. UTS-inspired geometry/topology telemetry

Add geometry/topology telemetry inspired by the principle behind **Unified Topological Signatures**:

> characterize representation spaces using multiple complementary descriptors rather than trusting one geometric statistic.

This is initially telemetry, not a direct training objective.

Candidate geometric descriptors:

* effective dimension;
* participation ratio;
* singular-value spectrum;
* anisotropy;
* pairwise-distance statistics;
* neighborhood-distance statistics;
* cosine distributions;
* hubness;
* local density;
* curvature estimates/proxies;
* geodesic distortion;
* projection neighborhood preservation.

Candidate topological descriptors:

* connectivity;
* neighborhood graph statistics;
* persistent-homology summaries;
* Betti summaries;
* persistence lifetimes;
* cluster connectivity;
* loop/cycle structure.

Separate:

`cheap online telemetry`

from:

`expensive offline telemetry`.

Individual descriptors may be correlated or redundant.

Measure redundancy rather than assuming every metric contributes unique information.

Preserve component-level values even if later dimensionality reduction is used to visualize the telemetry.

Questions include:

* Do geometry/topology changes precede grokking?
* Do they precede retrieval?
* Does topology reorganize before or after dimensional collapse?
* Which signals predict successful consolidation?
* Does compression preserve the learned signature?
* Are some telemetry components consistently redundant?

---

# 10. Continuous-depth geometric derivatives

Treat model depth as a sampled trajectory:

`t ∈ [0,1]`.

Do not convert the Transformer into a neural ODE in v0.1.

Estimate layerwise derivatives with:

* finite differences;
* central differences where possible;
* optional smoothing/spline analysis.

For geometric quantity `q(t)`, estimate:

`dq/dt`

and where meaningful:

`d²q/dt²`.

Candidate quantities:

* intrinsic dimension;
* effective rank;
* curvature;
* anisotropy;
* geodesic distortion;
* topology-signature coordinates;
* gate strength;
* effective horizon;
* attribution concentration;
* retrieval score;
* representation norm.

Representation-trajectory measurements may include:

* `||dh/dt||`;
* angle between consecutive representation changes;
* trajectory curvature;
* Euclidean path length;
* manifold/geodesic path length;
* Jacobian-norm proxies.

Questions:

* Where does abstraction begin?
* Are there identifiable retrieval layers?
* Are grokked trajectories shorter or smoother?
* Do second derivatives identify transition boundaries better than raw metrics?
* Does consolidation move important computation across depth?
* Can output-equivalent compression damage the internal trajectory?

---

# 11. Representation geometry

Create a clean geometry interface.

Initially support:

* Euclidean;
* hyperbolic;
* Euclidean × hyperbolic product spaces.

Candidate dimensions:

`8, 16, 32, 64, 128`

subject to model width.

Allow curvature to be:

* fixed;
* learned globally;
* learned per adapter/component where justified.

Use a trustworthy manifold package such as Geoopt unless a documented reason favors another implementation.

Keep geometry-specific mathematics isolated behind interfaces.

---

# 12. Intrinsic dimensionality

At minimum implement:

* effective rank;
* participation ratio;
* singular-value spectra;
* explained-variance curves.

Produce dimensional-sufficiency curves.

For each candidate dimension measure changes in:

* ID performance;
* OOD performance;
* reconstruction;
* geometry preservation;
* topology preservation;
* execution behavior where applicable.

Do not equate ordinary rank with intrinsic dimension.

---

# 13. Epiplexity / description length

Create an `EpiplexityEstimator` abstraction.

If practical, implement a prequential/online coding based estimator.

If only a proxy is implemented, label it a proxy everywhere.

Do not rename ordinary MDL as epiplexity.

The intended question is approximately:

> How much previously unexplained structure becomes compactly learnable under this representation and compute budget?

Where possible distinguish:

* structural description;
* residual unpredictability;
* computational budget.

Use these measurements to inform consolidation decisions.

Do not maximize epiplexity blindly.

---

# 14. Gating architecture

Adapted representations should expose explicit measurable gates where practical.

A simple initial form may resemble:

`h_out = (1-g) * h_base + g * h_adapted`

Support:

* fixed scalar gate;
* learned scalar;
* per-layer gate;
* later, grouped/contextual gates.

Record:

* gate value;
* entropy if appropriate;
* saturation;
* layer distribution;
* temporal evolution;
* relation to retrieval onset;
* relation to geometry/topology changes;
* relation to effective horizon.

---

# 15. Gating-strength ablations

Support interventions such as:

`g = 0`

`g = 0.25`

`g = 0.5`

`g = 0.75`

`g = 1`

plus learned values.

Required comparisons:

* gate off;
* weak gate;
* intermediate gate;
* learned gate;
* forced full gate;
* parameter-matched Euclidean control.

The causal question is:

> Did the adapted representation actually contribute to the result?

Do not interpret correlation between representation formation and performance as causation.

Look explicitly for non-monotonic behavior.

---

# 16. Self-distillation

Self-distillation must not mean blindly copying teacher outputs.

Candidate objectives may include:

* ground-truth task signal;
* teacher logits;
* relational structure;
* intermediate representation preservation;
* verified execution outcomes.

Teacher errors are not ground truth.

Generated explanations are not ground truth.

All candidates require independent ID/OOD verification.

---

# 17. Executable causal supervision

Add an **Executable Causal Supervision** subsystem.

The core principle is:

> **Prefer independently observable execution evidence over generated descriptions of what supposedly happened.**

Inspired by MIMIC-style instrumentation, record the environment at the layer where actions actually execute wherever possible.

For agent/tool tasks capture:

* pre-action state;
* action/tool call;
* arguments;
* execution result;
* errors;
* post-action state;
* verifier result;
* timestamp/order;
* state hash where practical;
* whether the intended state transition actually occurred.

Separate:

`model intent/process output`

from:

`executed action`

from:

`observed resulting state`.

Do not allow generated reasoning to overwrite execution evidence.

### Evidence hierarchy

When signals conflict, default to approximately:

1. independently verifiable external outcome;
2. deterministic environment/tool state;
3. execution logs;
4. trusted task verifier;
5. ground-truth labels;
6. teacher/process supervision;
7. generated reasoning/self-report.

This hierarchy should be configurable because not every environment has every source.

Do not assume execution logs are infallible. Record verifier confidence/provenance where possible.

---

# 18. Temporal causal credit inspired by γOPD

For sequential tasks, implement a temporal-credit interface inspired by discounted on-policy distillation.

The goal is not to reproduce γOPD blindly.

The goal is to test whether **discounted causal credit** can connect verified downstream outcomes to earlier actions while controlling long-horizon variance.

Conceptually support:

`credit_t = discounted future verified evidence + bounded auxiliary supervision`

where earlier decisions receive less credit as causal distance grows unless evidence indicates otherwise.

Expose configurable discount:

`gamma ∈ [0,1]`.

Compare:

* immediate/local credit;
* undiscounted return-to-go;
* discounted return-to-go;
* verifier-dominant bounded mixing;
* teacher/process-only control.

### Conflict rule

When a high-confidence independently verifiable outcome conflicts with teacher/process supervision:

**the executable/verifiable outcome should dominate.**

Do not average strong contradictory evidence into ambiguity by default.

Auxiliary teacher/process signals may help assign local credit when outcomes are sparse, but their influence must be bounded.

Record:

* source of each supervision signal;
* confidence;
* temporal position;
* discount;
* final combined credit;
* conflicts among sources.

Never use generated chain-of-thought as privileged ground truth.

Generated process text may be retained for diagnostics if appropriate, but execution and outcome evidence are the preferred causal signal.

---

# 19. Attribution telemetry

Make attribution a first-class member of the AGG telemetry vector.

Possible v0.1 measures:

* gradients;
* integrated gradients where practical;
* activation patching on small models;
* feature/layer ablation;
* context-element ablation;
* relevant-memory removal;
* distractor removal.

Prefer **interventional attribution** when affordable.

Track attribution across:

* training time;
* depth;
* memorization/retrieval phases;
* consolidation;
* geometry changes;
* compression.

Questions include:

* Does the model stop depending on individual memorized examples?
* Does relevant-context attribution rise at the retrieval transition?
* Does a grokked solution concentrate attribution around reusable circuit structure?
* Does compression preserve causal pathways?

---

# 20. Parameter pruning

Create modular pruning interfaces.

Magnitude may be used as a proposal heuristic.

It must not determine acceptance.

Use:

`propose → evaluate → accept/rollback`.

Track:

* total parameters;
* active parameters;
* sparsity;
* rank;
* bytes;
* ID delta;
* OOD delta;
* telemetry delta;
* mechanism delta.

---

# 21. Adaptive precision controller

Precision selection is its own stage.

Do not conflate numerical representation with physical storage.

Track separately:

* weight precision;
* activation precision;
* compute precision;
* accumulator precision.

Candidate weight precision ladder:

`FP32 → BF16/FP16 → INT8 → INT4 → ternary`

but candidates may be skipped or rejected.

Nothing automatically becomes ternary.

After structural changes, remeasure precision sensitivity.

A component may validly remain FP32 if evidence shows that lower precision violates constraints.

Another may become ternary.

This heterogeneity is expected.

---

# 22. Physical storage controller

Physical storage selection occurs **after logical precision selection**.

Example:

1. AGG determines that a tensor tolerates ternary weights.
2. Measure actual ternary symbol distribution and zero density.
3. Evaluate candidate physical encodings.
4. Benchmark on target hardware.
5. Select the representation that actually wins under configured goals.

Precision and encoding must remain conceptually distinct.

Candidate storage schemes may include:

* ordinary dense;
* fixed low-bit packing;
* fixed ternary packing;
* BITCOS-like ternary sparse encoding;
* bitmap + compact values;
* CSR;
* CSC;
* block sparse;
* N:M structured sparse.

---

# 23. Ternary BITCOS-like benchmark

For every ternary tensor record:

`z = zero density`.

Benchmark at minimum:

**A. Fixed ternary packing**

versus

**B. BITCOS-like distribution-adaptive representation**

where a presence/zero bitmap is combined with compact storage for signs/nonzero values.

Do not claim an implementation is BITCOS-compatible unless it actually follows the specification sufficiently to justify that statement.

Otherwise call it:

`BITCOS-like`.

Evaluate on the **actual target hardware**.

Record:

* CPU/GPU model;
* architecture;
* software/library versions;
* kernel used;
* tensor shape;
* zero density;
* theoretical bits/weight;
* serialized bytes;
* unpack/decode time;
* matrix operation latency;
* throughput;
* end-to-end effect where feasible.

If actual hardware execution is unavailable, clearly distinguish:

`estimated`

from:

`measured`.

Do not pick sparse encoding merely because zero density is high.

Metadata, unpacking and kernel behavior matter.

---

# 24. Compression ledger

Every intervention should produce a machine-readable ledger record.

Include:

* run ID;
* checkpoint;
* component;
* intervention type;
* telemetry before;
* proposed state;
* telemetry after;
* accepted/rejected;
* gate strength;
* ID delta;
* OOD delta;
* memorization delta;
* retrieval delta;
* generalization delta;
* attribution delta;
* geometry/topology delta;
* epiplexity/description-length delta;
* parameter delta;
* sparsity delta;
* precision delta;
* zero density;
* storage format;
* serialized-byte delta;
* hardware details;
* latency delta;
* execution/verifier evidence;
* reason for decision.

Use an inspectable format such as JSONL or Parquet.

The ledger is a core research artifact.

---

# 25. Mechanistic preservation

Behavior alone is insufficient.

Implement a mechanism-analysis interface.

Initial candidates:

* linear CKA;
* activation similarity;
* effective-rank comparison;
* representation spectra;
* attribution similarity;
* UTS-style signature similarity;
* depth-trajectory similarity;
* causal patching/ablation on toy models.

Behavior and mechanism thresholds must be separately configurable.

---

# 26. Consolidation trigger

Consolidation should initially be periodic and explicit.

Possible trigger inputs:

* task-loss plateau;
* OOD behavior;
* M/R/G phase scores;
* retrieval-over-memorization crossing;
* geometry/topology change;
* first/second derivative spikes;
* intrinsic-dimensional saturation;
* gate evolution;
* effective-horizon expansion;
* attribution redistribution;
* epiplexity/description-length change.

Trigger policies consume telemetry.

Do not optimize telemetry directly simply to trigger consolidation.

Every trigger event must record which signals caused it.

---

# 27. Critical checkpoint experiment

Identify checkpoints representing, where possible:

1. predominantly memorizing;
2. predominantly retrieving/reusing;
3. predominantly grokked/generalizing.

Apply identical compression sweeps to all.

Compare tolerance to:

* pruning;
* rank reduction;
* dimensional reduction;
* geometry substitution;
* weight quantization;
* activation quantization;
* ternarization;
* sparse storage.

Test the hypothesis:

`compressibility(memorization) < compressibility(retrieval) < compressibility(grokking)`

but do not assume it is true.

---

# 28. Phase-transition analysis

Align on a common training-step axis:

* task loss;
* ID performance;
* OOD performance;
* memorization score;
* retrieval score;
* generalization score;
* attribution;
* effective dimension;
* UTS-style telemetry;
* curvature;
* first depth derivatives;
* second depth derivatives;
* gate strength;
* effective horizon;
* epiplexity;
* sparsity;
* precision.

Support candidate change-point detection.

Do not call a change a phase transition because one noisy curve moved.

Compare ordering.

For example:

`retrieval crossover`

→ `attribution redistribution`

→ `dimensional collapse`

→ `OOD jump`

is different from:

`OOD jump`

→ `geometric cleanup`.

Preserve temporal ordering.

---

# 29. Required ablations

All major mechanisms must be disableable declaratively without code forks.

Required experiments include:

* baseline only;
* telemetry only;
* self-distillation only;
* geometry only;
* dimension adaptation only;
* gating only;
* epiplexity only;
* pruning only;
* quantization only;
* storage optimization only;
* executable supervision only;
* teacher/process supervision only;
* verified outcome supervision only;
* immediate temporal credit;
* discounted temporal credit;
* full Phase A;
* A + B;
* A + C;
* A + B + C.

### Gate ablations

* 0;
* 0.25;
* 0.5;
* 0.75;
* 1;
* learned.

### Telemetry ablations

Compare:

* task metrics only;
* geometry only;
* geometry + derivatives;
* attribution only;
* gates/horizon only;
* full telemetry vector.

This tests whether expensive telemetry actually contributes useful intervention information.

### Precision/storage ablations

Compare:

* fixed FP32;
* fixed FP16/BF16;
* fixed INT8;
* adaptive precision;
* adaptive precision + fixed packing;
* adaptive precision + adaptive physical storage.

### Supervision ablations

Compare:

* generated process supervision;
* teacher supervision;
* execution logs;
* verified outcomes;
* execution + discounted temporal credit;
* full bounded mixture.

---

# 30. Metrics

## Behavioral

* training loss;
* validation loss;
* ID accuracy;
* OOD accuracy;
* steps-to-generalization.

## Phase

* memorization score;
* retrieval score;
* generalization score;
* first retrieval-over-memorization crossing;
* stable crossing;
* change points.

## Retrieval benchmark

* gate strength;
* relevant distance;
* distractor density;
* context length;
* retrieval onset step;
* effective horizon.

## Representation

* parameter count;
* active parameters;
* effective rank;
* intrinsic dimension;
* singular-value spectra;
* geometry;
* curvature.

## Telemetry

* attribution;
* UTS-style descriptor vector;
* first depth derivatives;
* second depth derivatives;
* gate statistics;
* effective-horizon statistics.

## Compression

* sparsity;
* weight precision;
* activation precision;
* compute precision;
* accumulator precision;
* zero density;
* storage scheme;
* theoretical bit cost;
* serialized bytes.

## Hardware

* target device;
* kernel;
* memory traffic where measurable;
* latency;
* throughput.

## Executable supervision

* action;
* pre-state;
* post-state;
* execution status;
* verifier outcome;
* credit source;
* temporal discount;
* signal conflicts.

---

# 31. Repository structure

Use focused modules.

`agg/tasks/`
Synthetic tasks and benchmark generation.

`agg/models/`
Baseline models.

`agg/training/`
Training loops and checkpointing.

`agg/telemetry/`
AGG telemetry vector, history and schemas.

`agg/attribution/`
Attribution and causal interventions.

`agg/geometry/`
Euclidean/hyperbolic/product geometry.

`agg/topology/`
Topological descriptors and persistent-homology tools.

`agg/depth/`
Depth trajectories and derivatives.

`agg/dimension/`
Intrinsic-dimensionality estimators.

`agg/probes/`
Memorization/retrieval/generalization probes.

`agg/horizon/`
Effective-horizon metrics.

`agg/gating/`
Gates and gate interventions.

`agg/complexity/`
Epiplexity/description-length estimators.

`agg/distillation/`
Self-distillation.

`agg/supervision/`
Execution evidence, verifiers and signal provenance.

`agg/credit/`
Temporal-credit assignment.

`agg/consolidation/`
Trigger and intervention policies.

`agg/compression/`
Pruning, rank reduction and logical precision selection.

`agg/storage/`
Physical encoding and sparse/dense storage selection.

`agg/evaluation/`
ID/OOD/mechanistic evaluation.

`agg/ledger/`
Experiment/intervention records.

`agg/experiments/`
Declarative experiment orchestration.

`tests/`
Unit, integration, regression and scientific sanity tests.

---

# 32. Engineering standards

Use modern Python and PyTorch.

Prefer maintainable, readable code over clever abstractions.

Use:

* type hints;
* simple typed configs;
* pytest;
* Ruff or equivalent;
* static checking where practical;
* explicit random seeds.

Avoid:

* giant manager classes;
* hidden state;
* unnecessary metaprogramming;
* premature distributed infrastructure;
* premature billion-parameter optimization.

Version telemetry schemas.

Separate optional dependencies.

Keep experiments reproducible.

---

# 33. Documentation

Create:

* `README.md`;
* architecture specification;
* scientific hypotheses document;
* experiment matrix;
* ablation matrix;
* telemetry glossary;
* metric glossary;
* supervision/evidence hierarchy document;
* storage/precision design document;
* roadmap;
* contribution guide;
* references.

Document inspiration from relevant external work including:

* Unified Topological Signatures;
* epiplexity;
* grokking/dimensional-collapse literature;
* hyperbolic/product geometry;
* BITCOS;
* γOPD;
* execution-layer agent instrumentation.

Distinguish:

`adopted technique`

from:

`adapted idea`

from:

`AGG hypothesis`.

Do not make unsupported novelty claims.

---

# 34. v0.1 scope

v0.1 should demonstrate:

`train`

→ `record AGG telemetry`

→ `measure M/R/G transition`

→ `detect consolidation opportunity`

→ `compare geometry/dimension candidates`

→ `gate adapted representation`

→ `self-distill`

→ `validate`

→ `prune/quantize if justified`

→ `choose physical storage`

→ `verify behavior/mechanism`

→ `record ledger decision`.

v0.1 does NOT need:

* arbitrary manifolds;
* per-token general geometry routing;
* LLM-scale training;
* production ternary kernels;
* distributed training;
* complete neural architecture search;
* neural-ODE Transformers.

---

# 35. Scientific failure conditions

AGG has not succeeded merely because:

* the model is smaller;
* training accuracy remains high;
* OOD performance falls but compression improves;
* telemetry becomes smoother;
* epiplexity changes;
* topology changes;
* dimensionality falls;
* gates saturate;
* ternary weights appear;
* sparse storage looks better theoretically;
* teacher/process supervision agrees with itself;
* generated reasoning appears coherent.

Also treat these as failures or important negative findings:

* retrieval probes cannot distinguish retrieval from memorization;
* retrieval crossover does not replicate across seeds;
* geometric telemetry does not improve intervention decisions;
* derivative metrics add no information over raw telemetry;
* adaptive geometry never beats matched controls;
* adaptive precision collapses to one global precision;
* BITCOS-like storage loses to fixed packing on target hardware;
* verified outcomes conflict with teacher signals and AGG follows the teacher;
* compressed models preserve outputs while destroying the configured causal mechanism.

---

# 36. First milestones

## Milestone 1 — Grokking reproduction

Reproduce modular-arithmetic grokking.

## Milestone 2 — Telemetry

Record the full AGG telemetry vector across training and depth.

## Milestone 3 — Retrieval crossover benchmark

Run:

`gate × distance × distractors × context length`

and estimate retrieval-over-memorization training step.

## Milestone 4 — Geometry

Compare Euclidean and hyperbolic representations on hierarchy tasks under matched budgets.

## Milestone 5 — Gate causality

Perform gate-strength intervention.

## Milestone 6 — Continuous-depth derivatives

Test whether first/second geometric derivatives predict important transitions.

## Milestone 7 — Consolidation

Test deliberate consolidation against ordinary continued training.

## Milestone 8 — Compression

Compare memorizing, retrieving and grokked checkpoints under matched compression sweeps.

## Milestone 9 — Adaptive precision

Discover component-level precision floors.

## Milestone 10 — Storage

For ternary tensors, benchmark fixed packing against BITCOS-like encoding based on measured zero density on actual hardware.

## Milestone 11 — Executable causal supervision

Create a small tool/environment task with objectively inspectable state transitions.

Compare:

* teacher/process supervision;
* executable outcome supervision;
* discounted temporal credit;
* bounded mixtures.

---

# 37. Verification requirements

Before claiming v0.1 complete:

1. run full tests;
2. run lint/type checks;
3. reproduce a modular-arithmetic smoke run;
4. run hierarchy smoke experiment;
5. record complete telemetry snapshots;
6. verify first/second depth derivative calculations against synthetic known curves;
7. show distinct memorization and retrieval probe outputs;
8. verify retrieval crossover detection;
9. run at least one gate-strength intervention;
10. verify rejected consolidation candidates roll back;
11. verify precision candidates can be rejected;
12. verify logical precision and physical storage are separate interfaces;
13. demonstrate a case where dense storage wins;
14. demonstrate a case where sparse encoding wins;
15. record ternary zero density;
16. benchmark fixed and BITCOS-like ternary encodings where hardware permits;
17. verify execution logs distinguish intended from actual state transitions;
18. verify high-confidence executable outcomes override conflicting low-confidence teacher/process signals;
19. test at least two temporal-credit settings;
20. verify ablation flags genuinely remove their associated mechanism;
21. produce a complete compression/intervention ledger;
22. clearly identify every proxy metric.

Do not claim success unless checks actually run.

Priority order:

**verified outcomes > causal evidence > scientific interpretability > ablations > reproducibility > maintainability > compression ratio > raw performance.**
