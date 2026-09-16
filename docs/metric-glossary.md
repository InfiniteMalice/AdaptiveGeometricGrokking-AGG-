# Metric glossary

| Metric/API | Definition and interpretation |
|---|---|
| Accuracy | Fraction of exactly correct labels on the named split |
| Balanced accuracy | Mean class recall; necessary context for naturally imbalanced hierarchy labels |
| `spectral_statistics` | Center columns, compute singular values `s`, normalize energy `p_i = s_i² / sum(s²)` |
| Effective rank | `exp(-sum(p_i log p_i))`; this implementation uses covariance energy, which differs from entropy of normalized singular values |
| Participation ratio | `1 / sum(p_i²)` |
| Explained variance | Cumulative sum of ordered covariance-energy fractions |
| Anisotropy | Largest centered spectral energy divided by total energy |
| Degenerate spectra | Constant centered representations return zero rank, ratio and explained variance; they provide no evidence of dimension |
| `depth_derivatives` | Second-order finite differences along strictly increasing depth coordinates in `[0,1]`; three samples minimum |
| `trajectory_statistics` | Euclidean path length, speed, acceleration norm and consecutive-change angles on aligned representations |
| `linear_cka` | Squared Frobenius norm of centered cross-covariance divided by the product of within-representation covariance norms |
| Degenerate CKA | Returns zero when either centered representation is constant; high constant similarity is not preservation evidence |
| `intervention_attribution` | Baseline-minus-intervened metric, sign-adjusted when lower is better |
| M | Agreement of predictions with labels of nearest training representations; associative proxy |
| R | Original accuracy minus accuracy after relevant-memory ablation; signed causal sensitivity proxy, possibly negative |
| G | Mean correctness on a separate OOD set; independent of M and R |
| First crossing | First evaluated step where finite `R > M + margin` |
| Stable crossing | Start of the first run of `sustain` consecutive qualifying evaluations |
| Confirmation step | Evaluation where that sustained run becomes observable |
| Crossing interval | Previous evaluation and stable onset; lower endpoint unavailable if the run begins at the first sample |
| Censored run | No confirmed crossing within the observed budget; retain last observed step |
| Effective horizon | Maximum sampled distance with response at or above a configured threshold; missing if no qualifying measurement |
| Graph cycle rank | Edges minus vertices plus connected components in a radius graph; not manifold homology |
| Persistence | Optional Ripser Vietoris–Rips finite lifetimes and essential-class counts |
| Predictive coding proxy | `-sum(log2(p_target_before_update))` under a recorded learning order and budget |
| Zero density | Exactly zero entries divided by total entries |
| Serialized bytes | Complete codec envelope including metadata, padding and checksum |
| Decode-plus-matmul latency | Measured CPU decode, conversion to FP32 and dense matrix multiplication |
| Discounted return | `credit_t = sum_{k>=t} gamma^(k-t) combined_evidence_k` |

`PrequentialCodeLengthProxy` is an implementation of the `EpiplexityEstimator` interface, but its output is explicitly a **proxy**. It measures residual predictive coding cost. It does not separate structural description length from time-bounded entropy and does not include model or compute cost. The online linear-probe diagnostic scores each training example before updating the probe; order, seed and update budget materially affect results.

In `Constraints`, ID/OOD tolerances are absolute accuracy-point losses, the mechanism minimum is a separate threshold, and execution evidence can be independently required. Nonfinite metrics and declared numerical/reproducibility failures reject candidates. A low-cost candidate must still satisfy all configured constraints. CKA and accuracy are limited observations; causal preservation requires additional interventions.

No single metric establishes grokking. A steps-to-generalization report needs a preregistered sustained held-out threshold, a prior overfitting period, full training curves and replicated seeds; those are analysis criteria rather than claims derived from a short smoke run.
