# Ablation matrix

Ablations remove mechanisms declaratively through `agg.experiments.config.Features`. Its ordinary constructor enables all features; use `Features.baseline()` when constructing a control manually. Changes to observation flags must not accidentally change training randomness or optimization budgets.

| Named family | Intended comparison |
|---|---|
| `baseline` | Ordinary task training with optional mechanisms disabled |
| `telemetry_only` | Add observation without candidate intervention |
| `self_distillation_only` | Add ground-truth-anchored teacher fitting candidates |
| `geometry_only` | Compare geometry candidates at fixed dimension |
| `dimension_only` | Compare Euclidean bottleneck dimensions |
| `gating_only` | Isolate adapter contribution through a scalar gate |
| `epiplexity_only` | Add the explicitly labeled online code-length proxy |
| `pruning_only` | Generate magnitude-pruning candidates |
| `quantization_only` | Generate logical weight/activation rounding candidates |
| `storage_only` | Run physical-codec comparison independently of training precision |
| `executable_only` | Collect executed status signals |
| `teacher_only` | Teacher fixture evidence without verified signals |
| `process_only` | Process fixture evidence without verified signals |
| `verified_only` | Independently checked environment-state evidence |
| `immediate_credit` | Local credit, gamma zero |
| `discounted_credit` | Future evidence with configured gamma |
| `full_a` | Phase A observation, fitting and evidence mechanisms |
| `a_b` | Phase A plus geometry/dimension/gates |
| `a_c` | Phase A plus compression/storage |
| `a_b_c` | Combined mechanisms |

These names describe comparisons, not independent scientific results. Inspect each emitted configuration for its exact flags. Candidate-generating controls also require the consolidation execution path. `geometry_only` or `gating_only` still needs an adapter: disabling a gate uses the configured ungated adapter path, not removal of all adapted computation. A forced zero gate bypasses the adapted path; test this separately.

Additional controls are specified by configuration fields: gate values `0, .25, .5, .75, 1` and learned; telemetry modes `task`, `geometry`, `geometry_derivatives`, `attribution`, `gates_horizon`, `full`; gamma `0`, `.9`, `1`; fixed FP32, FP16, BF16, INT8 and adaptive candidate ladders. Compare fixed packing against adaptive physical storage only after numerical acceptance.

The supervision controls currently operate a deterministic execution/credit demonstration. They do not constitute separate trained-agent learning algorithms. `teacher` evidence is also distinct from the `distillation` training objective. Distillation optional representation/relation terms exist in the loss API but are not silently enabled by every named experiment.

For a valid ablation, check artifacts and actual computation: no proposals from disabled mechanisms, no offline topology work merely because a placeholder exists, no coding probe with epiplexity disabled, no adapter contribution under a forced zero gate, and identical baseline data and budget. Do not treat an empty telemetry field alone as proof that associated computation was skipped.
