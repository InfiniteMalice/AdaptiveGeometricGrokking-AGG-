# Measured resource-constrained controller

PR-5 adds an opt-in `ControllerConfig.resources` profile. Minimum ID/OOD
performance and maximum latency, model storage, parameter count, process CPU
inference time, training updates/tokens and peak memory are separate requirements.
All configured requirements must pass. Missing/invalid required measurements fail
closed. Per-requirement limits, measurements and margins accompany joint feasibility.
The original fixed capability anchor, target-gain check, mechanism constraints and
transaction rollback still apply. A resource surplus never offsets a regression.

Training limits are checked before execution using the provider's declared upper
budget and again against completed training. Providers without a budget preflight
cannot execute under a training budget requirement. An interrupted run records the
last attested completed step and an explicit unknown final count, never a fabricated
zero. Batch token counts use actual available rows if smaller than the requested batch.

## Measurement scope

The CPU probe warms up twice and records nine synchronous forward-batch wall-time
and process-CPU-time samples by default. Reports retain every sample, their median,
batch shape, thread counts, machine architecture, operating system and Torch version.
Process CPU time includes other process threads and may quantize to zero for tiny
models. Timing excludes input creation and is not end-to-end service latency.
Repeated samples are timing variability, not independent scientific task replicates.

Physical **model tensor storage** counts unique parameter/buffer storages. It does
not include activations, allocator overhead, optimizer state or peak process memory.
Peak runtime memory, energy and FLOPs are explicitly unmeasured. A peak-memory
requirement therefore fails. Fake activation quantization retains the same physical
tensor storage and does not establish deployment savings. No dense zero-count proxy
is substituted for actual storage.

## Reproducible comparison

```sh
python -m agg.experiments.cli resources --config configs/resource-smoke.json --output runs/resources
python -m agg.experiments.cli resource-audit --run runs/resources --seed 41
```

The four arms begin from the same trained checkpoint for each task/model seed:
fixed continuation, existing training controller, joint-resource controller, and
joint-resource controller with earlier selection-only experience. All have the same
intervention update cap. The fixed arm uses unchanged optimizer settings; the
controller proposes a bounded learning-rate change after ten observation ticks repeating the same measured scores
of an unchanged checkpoint. Observation ticks are not optimizer steps. This is a
controlled plateau fixture, not autonomous discovery of a training plateau.

Only the existing learning-rate and regularization provider executes. Depth,
geometry routing, precision and reasoning-budget proposals still lack providers;
they are unsupported, never counted as savings. Accepted resource-feasible arms
are ranked by measured median latency, without overriding capability gates.
Latency rankings on tiny CPU workloads are descriptive and sensitive to noise.

Experience records successful and failed joint candidates with their task, seed,
resource profile, hardware and selection provenance. A prior successful conditional
learning-rate fraction can suggest a later same-family proposal; target execution,
measurements and every gate run again, including on changed hardware. Prior timings
cannot satisfy current constraints. Reuse cannot import audit data or bypass rollback.

All real copy trials retain parent/candidate artifacts, original optimizer checkpoint
paths, planned/executed budgets, outcomes and conditions. A copy may pass model
constraints and then fail controller resources: those are distinct decisions. Rejected
copies remain reportable. Original training checkpoints include optimizer states;
the inference snapshots alone do not claim to resume optimization.

The entire task/seed matrix freezes before the one-use independent audit. Audit
reports every available candidate and retained arm, paired structural-cluster
comparisons against the fixed arm, fresh resource measurements and joint constraints.
Tiny strata and dependent timing samples limit inference. Final data remain untouched.
This adapts requirement-conditioned policy selection; it does not reproduce the
paper's amortized agentic discovery system. See [references](references.md).


## CPU smoke outcome

Data seed 137, model seeds 17/19, retrieval and modular tasks, two initial updates
and three intervention updates per arm, batch 16, width 8, audit seed 41. Sixteen
candidate copies executed 48 intervention updates, plus eight shared initial
updates: 4,928 actual training tokens overall. All retrieval candidates committed;
both joint variants rolled back the modular candidates that missed the configured
absolute performance floor. Fixed/existing arms do not enforce those added floors.
The second retrieval seed reused earlier successful selection experience and
remeasured it on the target model. No modular rule was eligible for reuse.

All four arms had identical audit OOD accuracy within each cell: retrieval seeds
17/19 = 0.1875/0.0; modular seeds 17/19 = 0.25/0.0. These tiny strata do not support
a benefit claim. Model tensor storage was 5,456 bytes (retrieval) and 3,748 bytes
(modular). The 16 retained-arm median selection latencies ranged 0.335-0.773 ms
while other verification work ran on the same CPU; variation does not establish
resource savings. Same architectures and precision give no storage reduction.
Audit can disagree with selection feasibility; independently measured conditions
are reported, not used to revise choices. Final data were untouched.


Fresh review found one important accounting defect: truncated/unreadable logs could
erase incurred work. Prefix-preserving accounting and three failing-then-passing
regressions now keep unknown totals unavailable, including a real interrupted run.
One minor is deferred: CPU model identifier is absent; available hardware provenance
is architecture/OS/Torch/thread counts only. Do not compare these timings across
unidentified CPUs. No critical issues were found; CodeRabbit was not run.
`python -m pytest -q`: 343 passed (one upstream warning); Ruff clean; mypy clean across 67 files.
