# ADR 0010: isolate a finite multi-step policy experiment

Accepted for exploratory PR-7. Reuse verified ToyEnvironment actions and original
discounted diagnostics. Use a finite tabular policy and exact behavior critic to
test the k-step importance objective against exhaustive references. Keep sampled
surrogate optimization separate from exact target-return reporting. Enforce a
fixed-behavior KL trust region and count reference execution separately.

The SCP solver is deferred because no certified convex subproblem implementation
or assumption validation is present. Bounded gradient ascent does not reproduce
CCRL. The exact critic narrows the scientific question and cannot establish
general RL sample efficiency. Results do not feed the supervised controller.
