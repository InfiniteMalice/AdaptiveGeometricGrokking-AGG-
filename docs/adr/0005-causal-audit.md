# ADR 0005: Causal probes consume frozen audit checkpoints

Status: accepted for opt-in research integration.

Save inference snapshots in the existing checkpoint callback only when independent
evaluation is enabled. A separate report validates the frozen manifest, constructs
oracle-checked audit pairs, and measures all saved checkpoints. No evaluator or
transformed audit examples are passed to the candidate generator.

Use transformations supported by the current task encoding. Tree-edge edits need
an explicit graph input and are not simulated by relabeling query tokens. Require
non-retrieval replacements to remain in the same protected audit stratum; merge
statistical clusters when replacement endpoints overlap. Preserve missing pairs,
failed checkpoints and censored transitions. Milestones may consume this evidence
later, but this PR creates no reward or optimization authority.
