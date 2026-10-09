# Discrete multi-step credit experiment

Run `python -m agg.experiments.cli multistep --config configs/multistep-smoke.json
--output runs/multistep` (one line). Outputs refuse overwrite. Schema version 1
records all sampled and exhaustive episodes, verified execution traces, behavior
policies, accepted and rejected line-search proposals, exact and sampled values,
importance ranges, terminal episode ESS, variance, learning curves and wall costs.

The task executes H binary additions in the existing integer-register environment.
Only the final verified all-ones outcome earns reward. Time and register count
identify the tabular state. The critic is exact backward dynamic programming under
the current behavior policy; critic learning is deliberately absent. Training
starts with probability 0.5 at every state, gamma 0.9. Every round samples a fresh
batch under its recorded policy. Streams share seeds across arms, while policies
and hence trajectories may diverge. H is bounded to 8 for exhaustive validation.

The objective adapts Eq. 2 of Deshmukh et al. (2026), with the latest min(k,t+1)
action ratios multiplying the exact behavior advantage at time t and discount
gamma^t. k=1 is the single-step policy surrogate; k=H equals exact improvement in
expectation. Logit derivatives include every action in that window. Exhaustive
behavior probabilities verify the expectation and variance; sampled variance uses
ddof=1 and is missing for one episode. ESS describes weighted episode records,
not independent time steps or calibrated uncertainty. No importance clipping is
used; nonfinite values fail. Full-support policies and immutable behavior hashes
reject stale/mixed batches. These checks protect trusted-host consistency, not
adversarially fabricated traces or SHA256 signatures.

The optimizer takes at most 8 normalized-gradient steps, each with at most 20
halving proposals. Every accepted proposal improves the sampled surrogate and
obeys max reachable-state KL(target||original behavior)<=0.02 and a 1e-4
probability floor. Exact target return is report-only. This is bounded gradient
ascent, not PPO or the paper's SCP/CCRL solver. Certified convex subproblems and
their assumptions have not been implemented, so no associated convergence claim
is made. The original discounted credit module supplies a non-learning diagnostic
arm and is never described as an RL baseline that trains a policy.

## Observed CPU fixture

H=3/5, seeds 3/5/7, 64 episodes x 8 rounds, k=1/2/H plus the diagnostic arm:
24 runs completed. 12,288 sampled episodes executed 49,152 actions; separately,
2,880 exhaustive-reference episodes executed 13,248 actions. Each learning run
used 512 episodes. The reference computation and exact oracle critic are extra
costs, not free environmental sample savings. Total wall time 8.05s included
logging and diagnostics while tests also ran; optimizer wall times below are
descriptive, not controlled performance benchmarks.

| H | Arm | Exact final discounted return range across 3 seeds | Optimizer seconds/run |
|---|---|---|---|
| 3 | non-learning diagnostic | 0.101250 | 0 |
| 3 | k1 | 0.779726–0.789874 | 0.213–0.239 |
| 3 | k2 | 0.778798–0.800642 | 0.226–0.490 |
| 3 | kH | 0.778337–0.799131 | 0.230–0.492 |
| 5 | non-learning diagnostic | 0.020503 | 0 |
| 5 | k1 | 0.569479–0.587297 | 0.308–0.389 |
| 5 | k2 | 0.567822–0.585935 | 0.327–0.431 |
| 5 | kH | 0.561384–0.598055 | 0.386–0.396 |

Maximum accepted KL was 0.019999994. Full-horizon reference bias relative to exact
improvement was at most 1.67e-16. Intermediate/full coupling did not consistently
outperform k1. Three seed-level values do not establish calibrated confidence
intervals or broad sample-efficiency gains. No ID/OOD task split applies to this
known MDP. No extrapolation, neural-policy convergence, AGG acceleration or causal
credit discovery follows. Keep this module independent of supervised AGG training.
See [source registry](references.md) for paper metadata and adaptation boundaries.
