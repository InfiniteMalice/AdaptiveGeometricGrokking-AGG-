# Finite empowerment reference

This opt-in reference calculates information from known deterministic MDPs.
It does not train an agent or establish improved generalization.

```bash
python -m agg.experiments.cli empowerment --config configs/empowerment-smoke.json --output runs/empowerment-smoke
```

The output directory must be new. `config.json` records settings; `summary.json`
uses schema `agg.empowerment/1`. Legacy commands and controller telemetry retain
their defaults and schemas. The new module is imported only for this command.

## Mathematics

For stationary skill z, start s and 0 <= gamma < 1, solve
`(I - gamma P_z.T) d_z = (1-gamma) e_s` in float64. This includes t=0, following
Ji et al. (2026), §3.1; it is not finite-horizon truncation or rollout sampling.
Residual and probability checks reject invalid solves rather than renormalize.

Effective empowerment is `I(Z;S+)` in **nats**, with uniform prior unless supplied.
Duplicating labels does not create outcomes. Duplicating only some labels changes
their uniform-prior weights and can change effective MI, but not channel capacity.
The report counts distinct occupancy rows rounded to 12 decimals separately.

Blahut-Arimoto optimizes the skill prior. Achievable MI is a capacity lower bound;
maximum row-to-marginal KL is an upper bound. Numerical gap, iteration count and
convergence are explicit. These are floating-point bounds, not exact arithmetic.
For `exhaustive_fixtures`, all deterministic stationary policies are enumerated
up to `max_policies`, covering vertices of the finite discounted occupancy
polytope. A converged exhaustive estimate covers the full finite MDP. Other
fixtures use one shortest-path policy per goal: **restricted-policy capacity**.
Its lower bound also bounds full potential, but its upper bound applies only to
the restricted channel. Exceeding the cap raises, without silently changing mode.

## Environments and controls

| Fixture | Purpose |
| --- | --- |
| fork | Three states, two actions, absorbing leaves: capacity = gamma ln(2). |
| grid | Six cells, stay/up/right/down/left; blocked actions self-loop. |
| gateway | Two rooms connected through one cell; centrality comparison. |
| central | Four two-step branches around a central cell. |
| distractors | Six controllable absorbing outcomes; only one rewards the task. |

The distractor control removes the only rewarding goal policy while preserving
many distinguishable outcomes. A constant reward vector separately gives zero
adaptation advantage despite nonzero MI. These are constructed controls, not
learned-policy failures. Reward is `d_z @ reward`, a normalized discounted state
reward; unnormalized return divides it by `1-gamma`. Best-skill reward assumes
oracle skill selection, not adaptation learning.

Directed normalized shortest-path betweenness counts unique edges, not action
aliases. Shortest distances count environment actions (`null` means unreachable).
Reciprocal occupancy counts expected independent geometrically terminated rollout
attempts, not contiguous step hitting time. The gateway itself need not have the
largest graph betweenness. No empowerment/centrality equivalence is assumed.

## Protocol and limitations

Config: `fixtures`, `exhaustive_fixtures`, `gamma`, `all_starts`, `max_policies`,
`capacity_tolerance`, `capacity_iterations`. Unknown keys fail. Reports include
environment tables/version/hash, occupancy/prior, intervals, reachability, reward
vectors and cost counts. Executed actions and training updates are zero; linear
solves and capacity iterations are counted separately. Wall time is observational.

Seeds, model/partition hashes and sampling uncertainty are unavailable with
reasons: no model, split or sampling exists here. Deterministic repetitions do not
provide independent scientific uncertainty. The estimators never access accepted
models or global RNG. No checkpoint migration or acceptance change is needed.

Direct mathematics: discounted occupancy and finite channel MI/capacity.
Adapted experiments: topology and anisotropy controls. Graph centrality and rounded
occupancy diversity are diagnostics, not task-success measurements. Gaussian width,
stochastic transitions and the paper's full optimizer are omitted. Representation,
exploration, memory and compression hypotheses remain untested here.

Next: held-out affordance probes using existing geometry adapters and versioned
telemetry, with behavior/similarity dissociation controls. See the
[six-stage design](specs/empowerment-play.md) and [references](references.md).
