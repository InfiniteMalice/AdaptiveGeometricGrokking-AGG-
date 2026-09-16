# Telemetry glossary

`TelemetrySnapshot` schema `0.1` identifies `step`, `checkpoint`, optional `layer`, and `component`. `raw` and `derived` retain category-level values. `availability` distinguishes observed from missing values and `proxy` marks interpretive limitations. Missing categories are `null`, not zero. JSON serialization rejects NaN and infinity. Availability means that a category has data, not that every possible submetric has been observed.

| Category | Current meaning |
|---|---|
| `task`, `id`, `ood` | Caller-supplied loss and held-out metrics |
| `attribution` | Executed ablation/replacement score changes where applicable |
| `geometry` | Norm, pairwise distance statistics and spectral anisotropy |
| `topology` | Explicit offline radius-graph or optional persistence results; not automatically collected |
| `dimension` | Centered singular spectrum, covariance entropy rank, participation ratio and cumulative explained variance; linear proxies |
| `depth_first`, `depth_second` | Normalized-depth finite differences; collector computes effective-rank derivatives |
| `gates` | Scalar-gate observations supplied by diagnostics |
| `horizon` | Nominal and empirically qualifying sampled distances |
| `memorization` | Training-representation nearest-neighbor agreement proxy |
| `retrieval` | Relevant-memory ablation accuracy loss proxy |
| `generalization` | Structural OOD correctness, interpreted as invariant-generalization evidence with caution |
| `complexity` | Prequential predictive code-length proxy with protocol/budget metadata |
| `sparsity` | Parameter and nonzero counts |
| `precision` | Logical weight/activation/compute/accumulator state |
| `storage` | Physical codec and measured serialized/latency properties |
| `execution` | Observed actions, states, errors and verifier outcomes |

`TelemetryCollector.collect(step, checkpoint, metrics, hidden=None, component='model')` accepts category-keyed JSON-compatible metrics and an aligned sequence of layer tensors. It returns one snapshot per hidden layer, or one snapshot without hidden inputs. Each tensor's last axis is the feature axis; other axes are flattened into observations. The runner chooses last-token representations, so these diagnostics do not describe every token by default.

Cheap geometry uses at most the first 256 observations deterministically. That cap is reproducible but order-biased; retain sampling policy in analysis. Spectral computation uses all supplied observations. Fewer than three hidden layers leave depth derivatives unavailable. Curvature, manifold path length and Jacobian norms are not inferred from Euclidean distance summaries.

`TelemetryHistory(path).append(snapshot)` writes inspectable JSONL; `.read()` reconstructs snapshots. It is a simple local append-only writer, not a concurrent database. Policies read telemetry and propose experiments; collecting a number does not make it an optimization objective. Redundancy, scale sensitivity and sampling effects must be measured before treating descriptors as independent evidence.
