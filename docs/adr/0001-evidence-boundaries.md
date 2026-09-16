# ADR 0001: Separate measurement, intervention, acceptance, and storage

Status: accepted from the user-approved research design.

A typed telemetry snapshot records observations, not targets. Proposal policies
cannot mutate a model. Candidates are deep-copied and evaluated against a frozen
reference; only feasible candidates may replace it. Logical precision produces
rounded tensors and metadata; codecs choose lossless representations afterward.
Execution observations remain distinct from intent and teacher claims.

This costs extra evaluation and copies but makes rejected candidates reversible,
negative results inspectable, and capability loss visible. Scalar weighted
"AGG losses" and automatic ternarization are excluded.
