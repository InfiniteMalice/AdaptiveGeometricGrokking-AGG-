# ADR 0009: Scaling methodology before a new world model

Status: accepted for opt-in research; latent predictor deferred.

Reuse the Transformer and matched geometry adapters in a controlled parameter,
training-volume and update grid. Fit only predeclared smaller configurations and
report separately on larger held-out configurations. Expose failures, sparse seed
uncertainty, residuals and unsupported extrapolation rather than assuming a law.
Equal update/token budgets do not imply equal geometry FLOPs; retain unavailable
operator-complete FLOPs instead of relabeling a proxy. Diagnostic compression uses
simulated activation rounding and does not establish physical savings.

ToyEnvironment already supplies verified executed state transitions. A separate
latent predictor needs independent scientific justification that the current smoke
cannot provide. Defer PR-6B instead of adding redundant unvalidated world-model
scaffolding. This limits claims to controlled supervised scaling methodology.
