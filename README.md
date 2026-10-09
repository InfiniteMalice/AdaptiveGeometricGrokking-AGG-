# AdaptiveGeometricGrokking (AGG)

**Understand → Restructure → Compress → Verify**

AGG is a small PyTorch research framework for testing whether deliberate
consolidation helps models discover reusable representations and whether those
representations tolerate geometry changes and compression. Capability preservation
is a constraint, not a term traded against model size.

The code provides reproducible toy tasks, trajectory recording, causal probes,
geometry adapters, explicit candidate trials, and exact reference storage codecs.
**The scientific hypotheses are not established.** Smoke runs demonstrate software
execution; they do not demonstrate grokking, compression benefits, or faster kernels.

## Quick start

Python 3.12 or newer is required. Use a fresh virtual environment.

```bash
python -m pip install -e ".[dev,geometry]"
python -m agg.experiments.cli run --config configs/smoke.json --output runs/modular
python -m agg.experiments.cli run --config configs/smoke.json --task hierarchy --output runs/hierarchy
python -m agg.experiments.cli run --config configs/retrieval-smoke.json --output runs/retrieval
```

For a CPU-only PyTorch installation, first install PyTorch from its CPU wheel index:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The runner refuses to overwrite an existing output directory. Choose a new name
for every run. Optional persistent homology requires `pip install -e ".[topology]"`.
Geoopt is imported only for curved adapters. Euclidean-only baseline runs do not need it.

## Experiments

```bash
python -m agg.experiments.cli run --config configs/smoke.json --ablation baseline --output runs/baseline
python -m agg.experiments.cli run --config configs/grokking.json --output runs/grokking-seed0
python -m agg.experiments.cli matrix --output runs/retrieval-plan
# Executes all cells; use a suitable training budget for scientific work:
python -m agg.experiments.cli matrix --steps 100 --execute --output runs/retrieval-factorial
```

The default factorial has 120 cells: five gate values × two distances × two
distractor densities × two context lengths × three seeds. Context padding varies
length without moving the relevant memory. Hard distractors and learned gates are
additional configurable controls. The phase detector records first and sustained
retrieval-over-memorization crossings, censoring, and sampled onset intervals.

Modular addition holds out a high-operand corner for OOD validation. Hierarchy uses
strict ancestry and holds out selected deepest descendants; balanced accuracy is
the acceptance metric because exhaustive ancestry is imbalanced. Retrieval holds
out one query-value binding per key and uses a strictly longer OOD distance, so
zero-distractor cases cannot reuse exact training rows as held-out observations.

These ID/OOD partitions are **development validation sets** used by candidate
selection. Final scientific claims require untouched test sets and new seeds.
The opt-in [independent evaluation protocol](docs/independent-evaluation.md)
adds structural training/selection/audit/final roles, separate task/model seeds,
complete attempt accounting and frozen checkpoint reports. Legacy runs retain
the development partitions described above.

## What a run records

- Training configs, optimizer/RNG checkpoints, and ID/OOD loss and accuracy.
- Versioned telemetry with raw/derived values, availability, and proxy labels.
- Before/after intervention ledger, capability deltas, rejection reasons and state hashes.
- Geometry/dimension trials, five-value gate interventions, and continued-training controls.
- Component precision choices with weight/activation/compute/accumulator separation.
- Metadata-inclusive codec sizes and measured CPU decode-plus-dense-matmul timings.
- Actual toy environment transitions, verifier conflicts, and temporal credit.

Candidates execute on copies. Rejection retains the accepted model. Sequential
stages compare to a fixed original capability anchor to prevent accumulated drift.
Telemetry policies never mutate models. `selected-state.pt` contains a versioned,
reloadable architecture and weights; use `agg.experiments.checkpoints.load_model`.

Logical INT8/INT4/ternary operations are **fake quantization** stored in floating
tensors. Compute and accumulators remain FP32. The codecs are CPU reference
implementations, not accelerated production kernels. BITCOS-like means a presence
bitmap plus packed signs, with no external compatibility claim.

## Verification and documentation

The opt-in [Adaptive Geometry-Guided controller](docs/controller.md) adds typed
temporal telemetry, diagnosis, bounded proposals, protected-metric acceptance,
and rollback. It reuses the existing training callbacks and model trials.
Run its synthetic demonstration without training a model:

```bash
python -m agg.controller.demo --output runs/controller-demo
```

The replay outcomes in this demo are synthetic. Real controller execution currently
supports isolated learning-rate and regularization continuation trials; other
mechanisms have explicit interfaces and unsupported-action reporting.

The [SoT-inspired extension](docs/state-of-thought.md) adds compact reasoning
telemetry, audited activation of externally valid evidence, causal ablations and
opt-in bounded compute recommendations. It does not reproduce SoT or add an LLM
reasoning engine. Compare absolute state and temporal dynamics on a synthetic fixture:

```bash
python -m agg.controller.reasoning_demo --output runs/sot-demo
```

```bash
python -m pytest -q
python -m ruff check agg tests
python -m mypy agg
```

See [verification](docs/verification.md) for executed checks and their limits,
[architecture](docs/architecture.md), [hypotheses](docs/hypotheses.md),
[experiment matrix](docs/experiment-matrix.md), [ablations](docs/ablation-matrix.md),
[telemetry glossary](docs/telemetry-glossary.md), [metric glossary](docs/metric-glossary.md),
[evidence hierarchy](docs/evidence-hierarchy.md), [precision and storage](docs/storage-precision.md),
[references](docs/references.md), and [roadmap](docs/roadmap.md).

The [approved research brief](docs/specs/approved-research-brief.md) preserves the
full scientific program. Its aspirations are not a list of established results.
The [research integration baseline](docs/research-integration-audit.md) and
[staged integration contract](docs/specs/research-integration.md) define the
independent-evaluation prerequisites for the next research stages.
Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.

Independent oracle-validated interventions: [causal evaluation](docs/causal-evaluation.md).

Versioned progress hypotheses: [milestone diagnostics](docs/milestones.md).

Lagged task-rule curation and matched learning: [block-summary benchmark](docs/block-learning.md).
`docs/resource-control.md` describes the opt-in measured four-arm resource controller benchmark.
`docs/scaling-analysis.md` documents controlled geometry scaling, held-out larger-model extrapolation and explicit compute limits.
`docs/multistep-credit.md` describes the isolated finite-MDP k-step objective and its inconclusive coupling results.
`docs/integrated-evidence.md` connects the research matrix, reproducible audit runs, portable evidence bundles and explicit final-release gate.
