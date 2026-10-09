# Discrete multi-step policy objective plan (PR-7)

Spec: exploratory PR7 in the research-integration contract. Use existing verified
ToyEnvironment transitions and discounted-return diagnostics; remain independent
from supervised AGG training. Source: Deshmukh et al. arXiv:2610.09108v1, Eq2,
verified against primary HTML 2026-10-09. This is a k-step PDIS investigation,
not a CCRL/SCP/PPO implementation or neural convergence claim.

## Design and controls

Finite delayed-outcome task: execute H binary add actions in ToyEnvironment;
reward1 only when the verified final register equals H, else0. State includes
(time,register); a triangular tabular Bernoulli policy covers all reachable states.
Gamma.9; exact behavior-policy advantages from finite backward dynamic programming.
This oracle-critic fixture isolates policy objective behavior rather than testing
critic learning. Environment episodes are actually executed and verified.

Jk = E_mu sum_t gamma^t product_{j=max(0,t-k+1)}^t pi(a_j|s_j)/mu(a_j|s_j) A_mu(s_t,a_t).
Use k1, intermediate2, fullH. Exhaustive known finite-MDP reference verifies k1
single-step surrogate and kH exact improvement; analytical logit gradients checked
against finite differences. Behavior support/probability floors prevent undefined
ratios. Reuse a batch only inside its bounded update; never mix behavior policies.

Bounded backtracking gradient ascent must improve the sampled surrogate and obey
max reachable-state KL(pi||mu)<=delta. No certified convex subproblem is present,
so SCP and associated convergence claims remain unavailable. Exact target return
is used for reporting/validation only, not candidate acceptance during learning.

## Tasks and interfaces

1. agg/credit/multistep.py: executed finite episodes, exact value/advantage reference,
   sliding-window PDIS value/gradient and weight/variance diagnostics. Test executed
   state consistency, kH identity, k1 reference, finite-difference gradients, invalid
   support, nonfinite values; RED->GREEN and full checks.
2. Add trust-region bounded optimizer and independent experiment/CLI. Test every
   accepted update satisfies support/KL/objective gates; actual execution budgets,
   deterministic seeded replay, existing diagnostic baseline remains non-learning.
3. CPU matrix H3/5, seeds3/5/7, 64episodes*8rounds, max8innersteps. Record actions,
   terminal rewards, true return, surrogate error/variance, weights/ESS, backtracking
   and measured optimizer wall cost; equal environment episode budgets across k.
   Negative/inconclusive results valid. Full checks, one fresh branch review, one
   regression fix pass, publish stacked PR. Do not merge.

Review focus: wrong sliding-window index/discount, stale behavior ratios, leaked
exact target returns in optimization, unsupported convergence/SCP claims, unverified
reward versus actual executed outcome, numerical overflow/support failure, concealed
rollout/optimizer costs, training objective mistaken for independent evidence.
