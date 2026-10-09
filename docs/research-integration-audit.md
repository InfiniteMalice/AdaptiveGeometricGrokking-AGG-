# Research integration baseline audit

Audited 2026-10-08 at `c50fbe4aacf302d2c146456a55d0decdfb12c951` (main).
No `AGENTS.md` or `CLAUDE.md` exists in this checkout. Development instructions
are `CONTRIBUTING.md` and the approved brief in `docs/specs/`.

## Inventory

| Area | Implemented baseline | Limitation |
|---|---|---|
| Architecture | Small pre-norm causal `TinyTransformer`, last-position classifier; optional `AdaptedModel` and activation-precision wrapper | No autoregressive LLM or learned world model |
| Modular task | Addition modulo p; randomized ID split; upper-operand-corner OOD split | Three roles only; commuted pairs can cross boundaries |
| Hierarchy task | Exhaustive strict ancestry in balanced trees; even deepest descendants held out; balanced accuracy for acceptance | Related subtrees remain shared across partitions |
| Retrieval task | Seeded key/value memories, reserved value per key for validation; OOD increases relevant distance | Selection ID/OOD reuse bindings; model/data seed coupled in runner |
| Phase detection | Operational M/R/G proxies; first/sustained R-over-M crossings, confirmation, onset brackets and right-censoring | Proxy validity and causal phase mechanism not established |
| Geometry/gates | Euclidean, Geoopt Poincare, product prototype-distance adapters; dimensions, curvature and fixed/learned gates; exact zero bypass | Geometry and dimensionality change budgets; no demonstrated geometry advantage |
| Trials | Deep copies; fixed original capability anchor; finite/reproducible behavior and ID/OOD/CKA constraints; rejected candidate retained only in ledger | Reference evaluator failure propagates before outcome ledger; all proposed attempts not yet durably accounted |
| Rollback | Original model and optimizer untouched by trial; Python/NumPy/Torch/CUDA RNG restored; provider stage/commit/finalize and rollback transaction | Python provider is trusted code; ledger not tamper-proof or crash-resumable |
| Controller | Opt-in observations, diagnoses, bounded proposals, event log and protected gates | Real training provider supports learning-rate/regularization continuation only; fresh optimizer; other action labels do not execute |
| Storage | Fake precision rounding, dense pruning/rank changes; exact ternary symbol codecs; complete envelope bytes and CPU decode-plus-dense-matmul timings | No integer/sparse kernels, whole-model deployment memory or general runtime savings |
| Evidence/memory | Source precedence, externally valid evidence routing, `AbstractionRegistry`, `CurriculumGate`, `RetentionTracker`, complete routing audits | No trained curator or proven future-block transfer |
| Reasoning telemetry | `agg.controller/2`, migration from v1; dispersion/displacement/cosine/entropy and temporal diagnostics | Synthetic reference routing, not paper reproduction or reasoning reward |
| Executable supervision | Integer add/multiply/set/exact divide, actual before/after trace, errors and verifier conflicts | Deterministic demonstration; no learned agent |
| Credit | Discounted return-to-go over source-aware evidence | Not CCRL, PPO, GRPO or causal-credit proof |

Inspected README, approved brief, architecture, hypotheses, evidence hierarchy,
experiment matrix, State-of-Thought documentation, storage precision, references,
roadmap, task generators, runner/config/candidates/analysis, trials, controller
providers/core/evaluation/memory/curriculum, telemetry, supervision, credit,
storage, and test coverage. Existing end-to-end tests exercise real tiny training,
candidate rejection, reloadable checkpoints, provider rollback and routing audits.

## Conflicts resolved by design

1. Keep audit/final sets outside `TaskData`, because the trainer passes `TaskData`
   into candidate fitting and metrics callbacks.
2. Group structures before splitting. Separate row hashes do not protect retrieval
   variants or related hierarchy examples.
3. Audit after freezing selection. A conservative acceptance gate that consumes
   audit scores would turn those scores into selection validation.
4. Extend the existing trial ledger and registry. Separate schemas describe new
   evidence; they do not create competing controllers or memory systems.
5. Treat controller continuation and consolidation fitting as fresh optimizers.
   Existing training checkpoints preserve optimizer state, but candidates do not
   resume that optimizer.
6. Reward shaping, a learned latent predictor and an SCP solver remain conditional
   on their stated evidence gates. Module availability cannot substitute for those gates.

## Reproducibility

Fresh editable installation with `[dev,geometry]`, CPython 3.12.13 on Windows
AMD64, PyTorch 2.14.1+cpu, NumPy 2.5.3, SciPy 1.18.1, Geoopt 0.5.1,
pytest 9.1.1, Ruff 0.16.10, mypy 2.4.0. Optional ripser was not installed.
Commands used the isolated environment's Python executable from the repository root.

| Command | Observed result |
|---|---|
| `python -m pytest -q` | Exit 0; 283 passed, 1 warning, 114.06 seconds |
| `python -m ruff check agg tests` | Exit 0; all checks passed |
| `python -m mypy agg` | Exit 0; no issues in 54 source files |

Pytest emitted Windows WMI `0x8007000e` diagnostic traces during PyTorch startup
and hardware metadata collection, then continued and exited successfully. A
separate `platform.machine()`/PyTorch import completed successfully. The warning
is Geoopt's use of deprecated `torch.jit.script` under PyTorch 2.14. These are
recorded environmental limitations; no source workaround was applied. The suite
includes end-to-end small training/trials fixtures. No scientific training sweep
or independent final test was performed in PR-0.

See [integration contract](specs/research-integration.md) for the staged scope.
