# Empowerment geometry and exploratory skill learning

Status: staged research design; PR 1 reference implementation first. The six
stages extend the existing research-integration stack without merging it or
authorizing its independent final-test release.

## Scientific contract

Test whether controllable outcomes, independent verification and consolidation
improve transferable, compressible capabilities. This is a hypothesis, not a
result. Latent geometry, information geometry and reward geometry are distinct.
External memory changes are not parameter learning. Missing measurements are
unavailable, never zero. Effective information and task reward are reported
separately. All new mechanisms are opt-in; old configurations retain semantics.

## Audit and integration map

The audit covered README, architecture, controller, state-of-thought, experiment
and ablation matrices, hypotheses, roadmap, references, CLI, checkpoint and
configuration conventions. Current integration points are:

| Concern | Existing owner | Extension boundary |
| --- | --- | --- |
| Environment | `agg.supervision.ToyEnvironment`, `agg.credit.multistep` | Both specialize different state/action semantics; add a finite navigation interface, reuse evidence and ledger infrastructure later. |
| Records | `agg.supervision.ExecutionEvidence`, `agg.ledger.Ledger` | Record actual executed transitions with environment hashes. |
| Memory | `agg.controller.memory.AbstractionRegistry` | Add episode provenance and externally tested skills; no parallel registry. |
| Validity | `agg.controller.routing.AuditedEvidenceRouter` | Retrieval eligibility cannot establish factual validity. |
| Geometry | `agg.geometry.GeometryAdapter`, `AdaptedModel` | Reuse Euclidean, hyperbolic and product adapters with matched budgets. |
| Observation | `agg.telemetry.controller.Observation` | Namespaced versioned opt-in diagnostics; preserve v1/v2 loading. |
| Intervention | `agg.consolidation.trial`, `agg.evaluation.Constraints` | Candidate copies, original capability anchor, RNG rollback, additional protected skill scores. |
| Checkpoints | `agg.experiments.checkpoints` | Retain model architecture and physical-storage distinctions. |
| Experiment | `agg.experiments.cli`, runner/config/independent | Lazy new commands and fixed manifests; no automatic factorial or final release. |

## PR boundaries

1. Finite deterministic environments, discounted occupancy, effective MI,
   exhaustive or restricted capacity, topology and reward controls, tests/docs.
2. Functional affordance probes, matched geometry, telemetry and negative controls.
3. Bounded seeded exploration with distinct directed/random/novelty/empowerment
   policies, inspectable skill records and held-out execution evaluations.
4. Existing-registry integration, independent verification, correction history,
   paired memory removal/injection/reordering/unavailability/restoration.
5. Block summaries with delayed utility, skill-aware copy-based compression and
   matched memory/no-memory baselines; rollback evidence.
6. Small preregistered smoke suite, machine summaries/plots, broader execution
   plan and hypotheses/reference/provenance updates. Large grids remain opt-in.

Each stage gets its own branch/PR, tests, Ruff, mypy, CPU smoke and actual results.
No automatic merge. Later-stage requirements are not asserted as PR 1 features.

## PR 1 mathematical contract

Use known finite deterministic transitions and stationary skill policies.
Discounted occupancy includes time zero:
`d_z = (1-gamma) sum_{t>=0} gamma^t P_z^t e_start`, in row-vector convention.
Solve `(I-gamma P_z.T)d_z=(1-gamma)e_start` in float64. Effective empowerment is
`sum_z p_z sum_s d_z(s) log(d_z(s)/sum_j p_j d_j(s))`, in nats. The default
effective prior is uniform; capacity optimizes the prior explicitly.

Enumerating all deterministic stationary policies covers occupancy-polytope
vertices for this finite discounted MDP. Blahut-Arimoto returns a lower/upper
capacity interval and a convergence flag, not an exact-arithmetic certificate.
The full-space claim is conditional on exhaustive enumeration. Goal-policy
subsets yield a restricted-policy capacity estimate and a lower bound on full
potential empowerment; their upper bounds apply only to the restricted channel.
An enumeration cap is enforced before allocating policies. Never silently fall
back from exhaustive to restricted mode.

Graph shortest paths count environment actions; reciprocal occupancy counts
independent geometrically terminated rollouts. They are different measurements.
Directed betweenness is a graph diagnostic, not assumed equivalent to empowerment.
Downstream reward uses normalized discounted occupancy, with task reward vectors
recorded. Gaussian-width estimation is deferred: a reward diagnostic is optional,
and direct finite reward comparisons suffice for the first deterministic stage.

Fixtures include a small grid, a one-cell gateway, central branches, absorbing
forks with redundant actions, and controllable distractors versus useful outcomes.
Use reward anisotropy to expose high information with zero adaptation advantage.
No learned policy, gradient update, memory experiment or compression claim in PR 1.

## Source status and limits

Ji et al. (2026), §§3–5 and relevant appendices B–F, H–I supply the discounted
occupancy/channel formulation and distinguish reward from information geometry.
The implementation below is a small mathematical reference, not a replication
of their optimizer or grid experiments. Graph probes and AGG fixtures are adapted
experimental ideas; their relationship to representation quality remains open.

Cloos et al. (2026), §§2–7, appendices A.4–A.5, C.4–C.5, C.8 and F motivate
bounded exploration, persistent artifacts and causal memory interventions.
Their agents' external memory changed while weights remained fixed. Selected
memory case studies motivate controls, not a general guarantee of improvement.
AGG will not claim to reproduce Eko, Clawblox, or unconstrained play.

Primary sources: https://arxiv.org/abs/2610.07796v2 and
https://arxiv.org/abs/2610.07130v1. See `docs/references.md` for APA entries.
