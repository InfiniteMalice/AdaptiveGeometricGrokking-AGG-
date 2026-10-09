# References and provenance

## October research integration

Checked against primary arXiv pages on 2026-10-08. These are preprints, not
independently reproduced AGG results. Metadata and the narrow findings below were
verified against the primary abstracts; Hu and Tang's acceptance-rule discussion,
WPM's author block, and CCRL's author block were also checked in full-text HTML.
Algorithm-level reproduction requires further full-paper and code review in the
relevant stage. No empirical paper result below is claimed for AGG.

1. Hu, L., & Tang, Y. (2026). *The winner's curse in LLM self-improvement loops:
   Selection noise, lock-in, and acceptance rules* (arXiv:2610.09239v1). arXiv.
   https://arxiv.org/abs/2610.09239v1

   Finding: reused selection scores can overstate held-out improvement; tested
   acceptance rules did not consistently beat greedy loops. AGG adaptation:
   independent reporting and complete candidate accounting. AGG hypothesis:
   structural audit gaps reveal optimistic intervention selection. AGG does not
   assume the paper's Gaussian model or calibrated priors apply to its candidates.

2. Zhao, J., Tang, J., Shu, Y., Wu, J., Lu, Y., Tong, J., Xu, H., Ge, W., &
   Zhang, Q. (2026). *World Potential Model: Pretrained world knowledge as progress
   potentials* (arXiv:2610.09560v1). arXiv. https://arxiv.org/abs/2610.09560v1

   Finding: task-relative milestone potentials supported agent optimization in
   ALFWorld and ScienceWorld. AGG adaptation: independently verified milestones
   as diagnostics first. AGG hypothesis: milestones predict later generalization.
   No pretrained evaluator, GRPO reproduction or unverified text reward is implied.
   The primary author block lists nine authors despite inconsistent abstract-page
   generated author-count prose; the entry follows that block.

3. Zhao, Y., Liao, L., Shen, L., Zhao, X., Zhang, Y., Feng, F., & He, X. (2026).
   *Learning to accumulate knowledge with mutual information*
   (arXiv:2610.10042v1). arXiv. https://arxiv.org/abs/2610.10042v1

   Finding: Knowledge Weaver combines standalone/marginal success and MI-inspired
   curation signals in agent benchmarks. AGG adaptation: provenance and future-block
   standalone/marginal utility in the existing registry. AGG hypothesis: lagged
   curation improves later learning. MI-inspired distinctiveness is not truth.
   The title here follows arXiv, correcting the longer title supplied in the request.

4. Wang, X., Liu, Z., Zheng, T., Feng, S., Yuan, P., Li, Y., Shi, J., Zhang, Y.,
   Tan, C., Zhang, J., Pan, B., & Li, K. (2026). *From Pareto to preference:
   Personalized test-time scaling via amortized agentic policy discovery*
   (arXiv:2610.09684v1). arXiv. https://arxiv.org/abs/2610.09684v1

   Finding: PersonTTS evaluates joint accuracy, latency and cost requirements with
   experience reuse and target-profile evaluation. AGG adaptation: conjunctive
   resource constraints on verified providers. AGG hypothesis: resource-conditioned
   control reduces measured cost while preserving protected capabilities. No
   PersonTTS controller or benchmark reproduction is claimed.

5. Zholus, A., Beltran-Velez, N., Yuan, J., Chandar, S., Nagarajan, T., Severo, D.,
   Sinha, K., Drozdzal, M., Romero Soriano, A., Bohg, J., Ballas, N., & Assran, M.
   (2026). *RoboJEPA: Scaling robotic latent world models* (arXiv:2610.10515v1).
   arXiv. https://arxiv.org/abs/2610.10515v1

   Finding: latent rollout error and planning performance scale with compute in
   the studied robotic setting. AGG adaptation: fit/extrapolate under controlled
   representation and resource budgets. AGG hypothesis: geometry changes scaling
   or compression tolerance. RoboJEPA's fitted law and robotics results are not
   transferred to AGG; a toy predictor requires separate justification.

6. Beucler, T., Neelin, J. D., Su, H., Asthana, S., Bretherton, C., Chapman, W.,
   Christopoulos, C., Clark, S. K., Grover, A., Lopez-Gomez, I., Schneider, T.,
   Subel, A., & Watt-Meyer, O. (2026). *Artificial intelligence pathways from
   weather to climate* (arXiv:2610.09770v1). arXiv.
   https://arxiv.org/abs/2610.09770v1

   Methodological argument: forecast skill alone does not validate response to
   changed forcings; interventions and OOD stress tests are needed. AGG adaptation:
   oracle-validated invariant/decisive computational interventions. AGG hypothesis:
   causal robustness accompanies reusable rule learning. No climate model or
   ethical-invariance claim is involved.

7. Deshmukh, S. V., Chittepu, Y., Gupta, D., Thomas, P., & Niekum, S. (2026).
   *Convex-concave reinforcement learning* (arXiv:2610.09108v1). arXiv.
   https://arxiv.org/abs/2610.09108v1

   Finding: the paper develops a difference-of-convex formulation and sequential
   convex programming, with multi-step policy comparisons. AGG adaptation:
   optional numerically verified discrete k-step experiments. AGG hypothesis:
   coupling delayed decisions can improve sample efficiency. Existing discounted
   credit is not CCRL; no neural-policy convergence claim follows. The entry uses
   all five named authors, confirmed in HTML despite inconsistent count prose.

8. Chen, P. B., Yu, G. X., Liu, X., Madden, S., Roth, D., Andreas, J., Downey, D.,
   & Cafarella, M. (2026). *ExperienceIndex: Artifact-grounded memory*
   (arXiv:2610.10091v1). arXiv. https://arxiv.org/abs/2610.10091v1

   Motivation: persistent artifact-specific experience. AGG adaptation is limited
   to artifact/version provenance in its existing evidence registry. Retrieval
   does not establish validity; no new memory architecture is implied.

9. Du, W., & Chung, S. (2026). *Can AI agents make open-ended scientific discovery?
   Evidence from Station* (arXiv:2610.08927v1). arXiv.
   https://arxiv.org/abs/2610.08927v1

   Motivation: persistent scientific investigation and independent review. AGG
   adaptation is limited to research records and evaluation protocols. The author
   initials and title correct those supplied in the request. No multi-agent
   operating system or autonomous-discovery result is implemented here.

10. Ramadan, O., Siavoshian, S., Saeed, A. K., Johnson, B. A., Diab, A. M. E.-A.,
    & Rodriguez, B. M. (2026). *Constrained diffusion for data-scarce orbital Monte
    Carlo in constellation tasking* (arXiv:2610.09323v1). arXiv.
    https://arxiv.org/abs/2610.09323v1

    Motivation: evaluate distributional fidelity, physical constraints and shifted
    support separately. AGG adaptation: distinguish statistical similarity,
    constraint satisfaction and OOD validity. No orbital simulator or diffusion
    model is added, and orbital findings are not AGG evidence.

11. Gong, Z., Hou, Y., Zeng, Z., Xiao, M., Yuen, C., & Lim, W. Y. B. (2026).
    *State of thought enables endogenous reasoning* (arXiv:2609.16055v2). arXiv.
    https://arxiv.org/abs/2609.16055v2

    Finding: compact internal-state control is evaluated on frozen language/vision
    backbones. AGG adaptation: compatibility with its existing versioned telemetry
    and audited routing. No duplicate controller, learned readout reproduction,
    or transfer of the reported efficiency gains is implied. Existing AGG work
    below was based on v1; the new registry records v2 without retroactively
    claiming that implementation reproduced v2.

## Earlier implementation provenance

Gong, Z., Hou, Y., Zeng, Z., Xiao, M., Yuen, C., & Lim, W. Y. B. (2026).
*State of thought enables endogenous reasoning* (arXiv:2609.16055). arXiv.
https://arxiv.org/abs/2609.16055

Checked 2026-09-21 against arXiv v1. Relationship: adapted motivation for compact
state-conditioned evidence activation and continuation. AGG uses deterministic
reference routing, richer named telemetry, protected trial acceptance and explicit
audit boundaries. This is not a reproduction of the paper's learned controller,
backbones, readout protocol, training method or empirical results.

Primary source pages checked 2026-09-16. Categories distinguish implemented techniques, adapted motivations and AGG hypotheses. This list does not assert that AGG reproduces the papers' results.

| Source | Relationship to AGG |
|---|---|
| Power et al., [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) (2022) | Adopted experimental motivation: small algorithmic tasks and delayed generalization. A smoke run is not a reproduction. |
| Rottach et al., [From Topology to Retrieval: Decoding Embedding Spaces with Unified Signatures](https://arxiv.org/abs/2511.22150) (2025) | Adapted idea: complementary geometric/topological descriptors and redundancy analysis. AGG does not claim an exact UTS implementation. |
| Finzi et al., [From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence](https://arxiv.org/abs/2601.03220) (2026) | Research motivation for bounded learnable structure. AGG's predictive coding proxy is not the paper's structural epiplexity quantity. |
| [Geoopt manifold documentation](https://geoopt.readthedocs.io/en/latest/manifolds.html) | Adopted library mathematics: Poincare ball maps and distances. Product routing and gating are local AGG design choices. |
| Nickel and Kiela, [Poincare Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) (2017) | Motivation for testing negative curvature on hierarchy tasks; no claim of reproducing the embedding experiments. |
| Kazanskii, [How to Tame Grokking: Representation Geometry as a Control Signal](https://arxiv.org/abs/2607.11666) (2026) | Related dimensional-collapse and intervention work. AGG does not adopt its spectral regularizer as a mandatory objective. |
| Fu et al., [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562v1) (2026, version-specific link) | Related context on temporal coupling and estimator bias/variance. The currently served version emphasizes teacher local support matching. AGG does not reproduce that algorithm. |

The approved brief names **gamma OPD / γOPD**, **BITCOS**, and execution-layer instrumentation associated with **MIMIC** as inspirations. Exact provenance and version correspondence for those names remain unresolved here. The supplied OPD page does not justify renaming generic discounted return-to-go as a verified γOPD reproduction. Resolve original source/version/algorithm correspondence before making such a claim.

Likewise, the local bitmap-plus-packed-signs codec is called **BITCOS-like**; neither external-format compatibility nor a reproduction of an optimized kernel has been established. The deterministic execution recorder is AGG's local implementation of intent-versus-observation separation, not a verified MIMIC reproduction.

The brief also proposes links between grokking, dimensional collapse, geometry and compression. Those are AGG hypotheses until specific source claims and independent experiments are supplied. Ordinary linear rank diagnostics cannot establish intrinsic dimension or the direction of a causal phase transition.
