# Executable block-summary benchmark

This opt-in benchmark tests whether source-supported task transformations improve
later learning beyond ordinary continuation and raw replay. It uses the existing
TinyTransformer, registry, evidence router, curriculum gate, retention tracker,
copy trials and candidate accounting. A summary is an executable transformation
with a readable proposition, not language-model-generated prose or a claim about
the student's internal understanding. Templates are enumerated by the researcher
and checked against actual source episodes and independent task oracles.

```sh
python -m agg.experiments.cli blocks --config configs/block-smoke.json --output runs/blocks
python -m agg.experiments.cli block-audit --run runs/blocks --seed 31
```

The first command trains and freezes the entire curriculum. The second validates
all frozen hashes and performs a one-use independent audit. Final data remain
untouched. Do not feed this audit into the same curriculum's lesson selection.

## Data, authority and controls

Training, selection, audit and final structural roles come from PR-1. A single
task-construction seed keeps retrieval key roles and hierarchy subtree roles fixed
across related blocks. Modulus cannot change inside a curriculum because that
would change the modular grouping allocation. Task-specific token namespaces
prevent token-identity sharing across families. Hierarchy uses one fixed query
symbol above the maximum configured node ID, so increasing depth cannot reuse
an earlier delimiter embedding as a protected node. The same initialized model and
output head persist across a family switch. Every input is left padded to the
predeclared maximum length: logical context variation is not a compute saving.

Three model lineages start identically: no summaries, cumulative summaries, and
replay. Each receives the same planned optimizer updates, batch size and encoded
token budget per block. Replay replaces half the current examples with earlier
training examples; summary transformations also replace rows rather than append
examples. Attempted updates, optimizer states and wall time are recorded even when
a candidate is rejected. All retained changes pass current capability gates and
each earlier block's fixed capability/mechanism anchor separately. An average
score cannot hide a regression in one protected block.

Incorrect, irrelevant, contradictory, obsolete, random and out-of-scope controls
are separate research copies. Corrupted labels are deliberately visible to those
copies only. No control can enter a retained lineage or become learned registry
knowledge. Oracle agreement and eligibility for knowledge retention are separate:
an irrelevant/no-op control can have correct labels but remain ineligible. The
out-of-scope operation fails before fitting and records zero executed updates;
its accuracy is missing, not an equal-budget treatment result.

The existing evidence router requires both applicability and a fresh validity
attestation. An active lesson is checked again on the current block before routing.
The registry records scope false positives and duplicate activations; task-family
switches change applicable lessons. CurriculumGate reports advance/hold/backtrack
recommendations. This benchmark follows its preregistered block schedule to keep
comparison budgets fixed; recommendations do not silently rewrite that schedule.

## Lagged utility and curation

After each block, record the model's actual source predictions, oracle labels,
source IDs/hashes, supported templates and redundancies in `AbstractionRegistry`.
Every new summary is experimental. Source accuracy/coherence cannot activate it.

On a **later** block, fit matched copies from the cumulative model's same parent:

- Standalone: a lesson versus no lesson.
- Marginal: a lesson plus other available lessons versus those other lessons.

For an already-active lesson, remove it from the marginal reference; comparing a
lesson against itself would incorrectly measure zero contribution. For a new
redundant lesson, the incumbent set remains the reference. Candidate counts,
actual updates/tokens and fitting/snapshot time include all these search copies.

Utility uses paired structural-cluster comparisons on future **selection** data.
That role is independent of the source episodes but remains development feedback.
Promotion requires oracle validity, protected feasibility, positive standalone and
marginal effects, and a positive marginal interval lower bound. A missing interval
cannot authorize activation. Changes affect the next block, never the block that
supplied the measurement. Negative marginal utility retires a record; retired
records require a new candidate identity. Source and transfer provenance remain
visible, including negative or unavailable utility. Binary registry confidence is
an activation gate, not a calibrated probability of scientific truth.

Semantic-preservation proxy = oracle agreement of generated labels. Contextual
distinctiveness = fraction of generated input/label rows absent from earlier
training blocks. These are curation measurements, not mutual-information estimates
or proofs of truth. A random incorrect label can be novel without being valid.
Record standalone and marginal utility separately from both proxies.

## Reporting and interpretation

`block-learning.json` records lineages, all shadow attempts, budgets, selection
scores, retained-model gains, changes in gain, headroom-normalized gains, retention
curves, scope switches and curriculum recommendations. Headroom normalization is
missing at saturation. Differences across blocks with different task difficulty
are descriptive, not significance claims. Every accepted/rejected/failed attempt
also uses the existing candidate JSONL and optimizer/checkpoint artifacts.

`block-audit.json` reports current-block ID/OOD and earlier-block retention for
every retained lineage, plus independent scores for every available candidate
checkpoint. Corrupt or unsupported candidates retain explicit missing reasons.
Paired intervals compare each candidate with the current no-summary baseline;
that comparison may involve different parents after histories diverge. The
same-parent standalone/marginal selection contrasts are separately identified.
Few structural clusters limit uncertainty; hierarchy's single subtree cluster
cannot yield a cluster interval. No deployment FLOPs or physical savings are inferred.

This adapts Knowledge Weaver's lagged utility and curation idea. It does not
reproduce its mutual-information objective or train a natural-language summarizer.
See [references](references.md), [design](specs/block-learning.md), and the
[architecture decision](adr/0007-executable-block-summaries.md).


## Reviewed CPU smoke

The six-block config uses data/model seeds 101/17, two updates per block, batch
16, width 8, and a common 17-class output head. Independent audit seed: 31.
The corrected run made 74 candidate attempts: 68 capability-feasible copies and
six out-of-scope failures before training. Controls remain isolated research copies;
trial acceptance does not imply lesson promotion. Total executed work was 136
optimizer updates and 26,112 padded training tokens, including search copies.
Each retained lineage received 12 updates. Registry serialization used 43,725
bytes; the measured curriculum wall time was 18.59 seconds, excluding freeze and
audit, on a CPU also running verification. This is not a speed comparison.

All nine lessons remained experimental. Seven later-block marginal selection
comparisons measured zero effect; none authorized activation. Audit OOD accuracy
was zero for every lineage except retrieval block 2: baseline/summary 0.125,
replay 0.03125. Hierarchy uses balanced accuracy. All curriculum recommendations
were backtrack; the fixed benchmark schedule continued as declared. These weak,
short runs verify accounting and separation, not beneficial knowledge accumulation.
No final scientific data were used.

The fresh review found one important cross-depth delimiter collision, fixed with
a regression test that first failed. It found no critical or minor issues and
declined scientific/statistical validity judgments. CodeRabbit was not run.
`python -m pytest -q`: 328 passed (one upstream geoopt/Torch warning). Ruff clean; mypy clean across 64 source files.
