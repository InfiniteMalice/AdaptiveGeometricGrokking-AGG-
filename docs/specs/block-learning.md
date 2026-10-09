# PR-4: Executable block summaries and lagged transfer

## Scope and research contract

The current TinyTransformer cannot consume prose. Represent a bounded summary
as an oracle-checked task transformation, with a readable proposition and source
episode references in the existing `AbstractionRegistry`. Candidate generation
enumerates explicit templates supported by executed source examples. This tests
source-supported data augmentation and curation, not language-summary generation
or extraction of a neural model's internal rule. The user was offered a prose
architecture alternative; proceed with the stated smaller default unless steered.

Reuse the registry, `CurriculumGate`, `RetentionTracker`, evidence validity/router,
EventLog, copy-based `trial`, candidate recorder and fitting optimizer. Do not
create another memory store. Known transformation templates: retrieval irrelevant
permutation/relevant-value cycle, modular operand swap, and fixed-tree query
replacement validated by its oracle. Summaries are experimental on creation.

Each block supplies actual predictions and independently decoded labels for source
episodes. Record template support, redundancies, contradictory controls and exact
scope. Future blocks differ in retrieval distance/density/context, hierarchy depth
or task family. Keep one data seed and stable structural roles within each task;
do not change modulus inside a curriculum. Predeclare per-task token namespaces
and a common output head/left-padded maximum sequence length so family switches do
not silently reset the model or share protected token identities across families.
Report logical context lengths separately from encoded padded token counts.

## Authority, timing and comparisons

Maintain matched no-summary, cumulative-summary and replay model lineages. They
start identically and receive equal optimizer updates and batch/token budgets.
Application is a host-authorized, guarded copy trial, never direct registry model
mutation. Protect current and previously encountered capability anchors. Use
selection data only for training/intervention decisions. Audit/final tensors never
enter fitting, routing, utility-based promotion or curriculum decisions.

On a later block, evaluate each applicable experimental lesson in two matched
copy contrasts from the cumulative lineage's same parent: alone versus no lesson,
and with existing lessons versus existing lessons alone. Record standalone and
marginal future-block selection utility, paired structural-cluster intervals,
headroom, retention, negative transfer, and additional search cost. No utility
update at the source block. Positive source reconstruction is not transfer.

Promote a lesson only after oracle validity, guarded feasibility and positive
future standalone/marginal utility with a positive marginal lower interval bound.
Promotion affects subsequent blocks, never the already measured block. These are
development retention decisions, not independently established scientific truth.
Every future application must recheck scope and oracle validity. Negative transfer
can retire a lesson. Registry snapshots and event records preserve obsolete facts.

Negative controls cover incorrect, irrelevant, contradictory, obsolete, random,
out-of-scope lessons and replay without abstraction. Corrupted-training controls
run only as isolated research copies and can never enter a retained lineage or
validity attestation. Unsupported transformations remain explicit missing results.
Check scope/validity rejection separately from accuracy effects. Retrieval alone
never proves a summary valid.

After all blocks, freeze configuration, partitions, every attempt/checkpoint,
registry, routing and training records. A separate one-use report audits all
available models on private per-block structures. No audit-driven adaptation.
Compare future learning and retention, first/second changes in improvement,
headroom-normalized gains (null at saturation), source semantic preservation and
contextual distinctiveness. The latter are curation proxies, not mutual information
estimates or evidence of truth. Physical compute savings are outside this PR.

## Execution plan and checks

1. Tests first: source provenance and deterministic template semantics; delayed
   utility rejects same/earlier blocks; no premature activation; incorrect and
   out-of-scope controls cannot become valid registry knowledge.
2. Add minimal registry update API and template generation. Preserve existing
   serialization/defaults, version new summary/utility payloads in provenance.
3. Tests first: tiny real multi-block learning with matched updates/tokens, actual
   optimizer artifacts, record failed/rejected candidates, lineage immutability,
   private-role separation, negative controls and audit only after freezing.
4. Implement the explicit benchmark and CLI, run CPU fixtures across related
   blocks and a family switch. Record exact budgets, costs, intervals, censored
   or absent utility, and negative findings. Full suite/Ruff/mypy, fresh review,
   regression fixes, separate unmerged PR.

Scientific gate: template-based summaries may prove useless or harmful. Do not
claim cumulative knowledge accelerates learning from a tiny fixture. Source
inspiration is Knowledge Weaver's lagged curation idea; no mutual-information
objective or paper reproduction is claimed.

Cross-depth hierarchy encoding reserves one fixed query symbol above all configured node IDs; raw task oracles retain their native encoding.
