# Research hypotheses

AGG is a small research harness for **Understand → Restructure → Compress → Verify**. Its current implementation makes experiments possible; passing smoke tests does not establish these hypotheses.

| Hypothesis | Discriminating experiment | Evidence against it |
|---|---|---|
| Consolidation accelerates generalization | Compare sustained held-out accuracy onset against equal-update continued training, with paired seeds | No benefit after matching updates, parameter budgets, and selection effort |
| Retrieval becomes more explanatory than memorization | Compare representation-neighbor agreement with relevant-memory ablation and replacement, controlling surface similarity | Probes cannot distinguish known retrieval and memorization controls |
| Generalizing checkpoints tolerate more compression | Apply identical candidate grids and independent acceptance constraints at preregistered checkpoint classes | No ordering, or apparent gains disappear after matching initial capability |
| Geometry suits task structure | Compare Euclidean, hyperbolic and product adapters with the same dimension, prototype count, fitting steps and gate | Curved adapters do not beat matched Euclidean controls |
| Depth derivatives add information | Predict held-out consolidation outcomes with raw telemetry versus raw telemetry plus derivatives | No out-of-seed predictive improvement |
| Gates mediate useful adapted computation | Sweep forced gate values while holding learned weights fixed | Gate changes have no causal performance effect |
| Compact learnable structure predicts opportunity | Test online coding proxy against independent candidate acceptance outcomes | Proxy correlates only with accuracy, ordering or probe compute |
| Verified outcomes improve credit assignment | Compare verified, teacher, process and mixed signals under deliberate conflicts | Strong independent evidence loses to incorrect auxiliary signals |

The evidence scores M, R and G are operational proxies, not mutually exclusive psychological states. Spectral concentration is a linear representation diagnostic, not a verified intrinsic manifold dimension. Lower code length is not proof of greater epiplexity. CKA similarity alone does not establish causal mechanism preservation.

Pre-register splits, selection criteria, tolerances, evaluation cadence, training budget and seed list before examining outcomes. Treat absent crossings as censored observations. Report harmful and rejected candidates alongside successful ones. Repeated candidate selection on the same OOD set makes that set a validation set; obtain a separate untouched final test set for scientific claims.
