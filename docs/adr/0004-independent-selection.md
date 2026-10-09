# ADR 0004: Freeze selection before independent evaluation

Status: accepted for the opt-in research integration.

The development runner consumes only training and candidate-selection tensors.
The task protocol separates structural groups, including retrieval distractor
keys and hierarchy subtrees. A separate reporting command reconstructs private
audit/final data after verifying a closed selection manifest. Final evaluation
also requires explicit authorization and the manifest digest.

Record proposal intent before evaluation, actual parent and optimizer artifacts,
and outcomes for rejected and failed candidates. Preserve the existing fixed
capability anchor and copy-based rollback. Audit results never feed this selector.

The manifest detects accidental artifact changes; it is not a sandbox against
a hostile operator replacing the manifest and artifacts together. Intervals use
paired structural clusters and expose unavailable uncertainty. Hierarchy depth
must be at least four to retain both classes; a single subtree per role cannot
support a cluster interval. See the [protocol guide](../independent-evaluation.md).
