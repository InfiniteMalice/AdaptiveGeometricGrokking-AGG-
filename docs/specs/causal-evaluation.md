# PR-2: Causal intervention benchmark

The independent evaluator will compare frozen models on oracle-validated pairs.
Development selection and training remain unchanged. This adapts Beucler et al.'s
distinction between prediction and intervention validity, not a climate model.

Use existing modular arithmetic, fixed-tree ancestry and retrieval encodings.
For retrieval, shuffle only irrelevant positions or replace the unique relevant
value. For modular addition, swap operands or replace one operand. For hierarchy,
replace one queried node while preserving or changing ancestry. Pair replacements
must already exist in the same private audit stratum; never cross protected task
groups. Fixed-tree queries do not encode edges: edge changes and arbitrary node
renaming are unsupported without a new task interface, and will not be claimed.
No-op transformations and unavailable same-role replacements remain missing.

An independent decoder recomputes original and transformed labels. Correct
invariance requires both answers correct. Required update is transformed
correctness conditional on original correctness for decisive pairs. Also report
raw flip, joint correctness, original/transformed accuracy, wrong-to-correct
counterfactual recovery, and decisive-minus-irrelevant flip sensitivity. These
are behavioral diagnostics, not proofs of mechanisms or iterative recovery.
Report all denominators and paired cluster uncertainty with missingness.

Save weights-only inference artifacts at existing checkpoints when independent
evaluation is enabled. After selection closes, a separate audit-only command
evaluates every frozen checkpoint, then links available observations to existing
sustained M/R crossing brackets. No observed crossing means censored, not onset
zero. No checkpoint after confirmation means after evidence unavailable. No final
data, rewards, automatic selection, or geometry-causation claims are introduced.

## Execution plan

1. Write oracle, no-op, wrong-consistency, required-update and protected-role
   tests; observe missing APIs, implement transformations and metrics.
2. Write a real-run test for immutable frozen checkpoint trajectories and missing
   crossing/final data; add inference snapshots, report API and CLI. Record failed
   checkpoints without suppressing the rest. Run full pytest, Ruff and mypy.
3. Run a tiny retrieval trajectory and causal audit. Document exact measurements,
   scope and missingness; fresh branch review, fixes, verification, separate PR.

Risks: duplicate rows are clustered, small structural groups limit uncertainty,
replacement support may be sparse, fixed embeddings restrict representation
changes, and selecting epochs from development proxies does not establish a
causal mechanism. Version reports and transform definitions independently.
