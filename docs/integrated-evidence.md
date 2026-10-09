# Integrated research matrix and protected release

This is an implementation and small-fixture evidence release. It does not establish
grokking acceleration, geometry superiority, useful knowledge accumulation, or real
resource savings. The experiments are separate, heterogeneous studies; do not pool
their effects into one score. Negative findings and unavailable conditional work
remain part of the matrix. Paper metadata and adaptation boundaries are in
[references](references.md); no entry below claims a paper reproduction.

## Reproduce

Use Python 3.12+ and the repository's documented dependencies. From an installed
checkout, set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for CPU smoke runs, then run:

```text
python scripts/reproduce_research_matrix.py --output runs/research-matrix
```

The script refuses an existing output directory. It runs the integrated comparison,
four selection arms, block learning, resource controls, controlled geometry scaling,
and finite-MDP credit using committed configs. It audits frozen outputs and copies
complete evidence with hashes into `evidence/`. It prepares a final-release manifest
without running final inference. A new execution is a replication, not a byte-identical
archive: wall measurements, serialization hashes and some environment-dependent
floating-point results can differ. Exact configs, seeds and runtime records govern
every result. Earlier historical fixtures use their own saved configs and seeds.

Individual CLI commands remain available. For example:

```text
python -m agg.experiments.cli integration --config configs/integration-smoke.json --output runs/integrated
python -m agg.experiments.cli evidence-bundle --sources sources.json --output evidence
python -m agg.experiments.cli prepare-release --run runs/integrated/retrieval-23 --output release.json
```

The sources file maps safe unique names to existing source directories, relative
to that file. Bundling validates every nested frozen selection, copies all files,
and hashes post-freeze reports as well. No result is silently filtered for success.
Hashes protect accidental mutation on a trusted host; they are not signatures.

## Hypotheses and measurements

| Question / exact working hypothesis | Independent variable and controls | Primary measurements / interpretation limit |
|---|---|---|
| Consolidation improves generalization beyond extra training | Self-distillation vs ordinary continuation; copied same baseline, same batches, seed, optimizer type and eight extra updates; fixed original capability gate | Paired selected-minus-continuation ID/OOD accuracy; report candidate result and gate decision. A rejected treatment leaves the baseline selected, so this is the gated procedure's effect, not a successful-distillation-only effect. No onset acceleration claim without observed onset. |
| Causal robustness accompanies generalization | Oracle-preserving vs oracle-changing interventions at saved checkpoints; within-role paired rows and merged endpoint clusters | Joint correctness, correct invariance, spurious flip, required update conditional on original correctness, recovery conditional on original error. Phase availability uses successful checkpoints; missing crossing gives no before/after causal contrast. |
| Milestones predict subsequent generalization beyond raw loss | Fixed evidence-based definitions vs training loss on the same adjacent checkpoint pairs | Next-audit OOD/update correlations; sustain=3, reversals/conflicts/missing evidence retained. Adjacent pairs are dependent; correlations are descriptive and no reward is authorized. |
| Accumulated summaries improve future learning | Persistent summary/no-summary/raw-replay lineages; matched updates and tokens; independent shadow invalid/random/obsolete/out-of-scope controls | Future selection standalone and marginal utility, all prior capability anchors, learning-curve changes and independent audits. Researcher-enumerated executable templates are not learned prose or discovered internal rules. |
| Geometry improves OOD at controlled scale | Euclidean/hyperbolic/product with matched width and adapter settings; small widths fit, larger width held out | Error vs actual parameter count, small-only power/log-linear fits, held-out residuals, whole-seed bootstrap, horizon and precision tolerance. Updates/tokens match; operator FLOPs are unavailable. Sparse seeds do not validate interval calibration or a scaling law. |
| Requirement conditioning saves measured resources while preserving capability | Fixed/existing/joint/experience controllers from shared initial checkpoints and fixed resource profiles | Separate ID/OOD gates and latency/bytes/update/token constraints; all candidate and retained measurements. Only real LR/WD continuation providers; the plateau trigger is a declared controlled fixture. No hardware/general compute-saving conclusion from equal scores. |
| New latent prediction generalizes beyond executed supervision | Conditional one-step vs autoregressive learned latent predictor | Unavailable: existing exact executed-state supervision already covers the current hypothesis; no distinct justified prediction experiment. No fabricated null result. |
| Coupling decisions improves delayed outcomes | k1/k2/kH, same episode counts and seed streams; non-learning discounted-return diagnostic | Exact finite-MDP return, sample/exact surrogate error and variance, importance/ESS, KL, optimizer cost. Exact behavior critic; no neural-policy claim. SCP solver remains unavailable. |
| Selection gains survive independent evaluation | Frozen development selection vs audit vs explicitly authorized final on structural role partitions | Absolute ID/OOD, selection-minus-independent gaps, all candidates and paired cluster intervals. Audit is descriptive; final stays unavailable until exact release authorization. |

## Exact historical fixture budgets

The evidence bundle's source subdirectories retain every config, training log,
candidate record, checkpoint, raw report and uncertainty calculation. These are
the authoritative row-level sample counts and cost records, including missing data.

| Source | Seeds and matched budget | Search / evaluation accounting |
|---|---|---|
| selection (PR1) | Four arms x paired data/model seeds (71,11)/(72,12), three baseline updates, one-update fitting candidates, batch32 | Two proposals/cell, including rejected attempts; larger-selection arm doubles selection rows only. Independent reports retain all candidate ID/OOD and structural-cluster intervals. Legacy runs have no frozen continued endpoint. |
| causal (PR2) | Retrieval data73/model11, six updates, checkpoints0/2/4/6; audit seed29 | Oracle-validated paired ID/OOD interventions; no observed crossing, right-censored at6. |
| milestones (PR3) | Retrieval83/13 and modular84/14, six updates, same checkpoint schedule, audit seed29 | Three adjacent dependent pairs; explicit missing compression support and censored milestones. |
| blocks (PR4) | Data101/model17, six fixed blocks, two updates/trial, batch16; retrieval lengths8/10/12, hierarchy depths4/5, modular17 | 74 attempts:68 capability-accepted,6 out-of-scope failures;136 updates,26,112 tokens including shadow trials. Each retained lineage12 updates. Registry43,725 bytes. |
| resources (PR5) | Data137/models17/19, retrieval/modular, four arms; initial2 and intervention3 updates, batch16 |16 candidates,48 intervention+8 initial updates,4,928 training tokens. CPU2 warmups+9 measured forwards per resource probe; probe samples and scope retained. |
| scaling (PR6A) | Data149/models17/19, widths8/12/16/20 fit and24 extrapolate, three geometries, volumes16/64, updates4/8, batch16 |120 cells,720 updates,92,160 training tokens;2,520 audit forwards/983,040 input tokens.24 fit groups; failed or missing observations remain explicit. |
| credit (PR7) | H3/5, seeds3/5/7,64 episodes x8 rounds, at most8 inner updates, gamma.9, maxKL.02 |24 runs;12,288 sampled episodes/49,152 actions plus2,880 exhaustive-reference episodes/13,248 actions. All optimizer evaluations/backtracks and wall times recorded. |
| integrated (PR8) | Retrieval data211/modular223; models23/29;32 baseline updates,8 continuation and8 self-distillation, batch16,width16, one layer; nine inference checkpoints0..32 |Four candidates:two accepted/two rejected, zero failed.192 total updates,16,896 training tokens.436 development forwards/105,406 tokens;120 independent-audit forwards/37,296 tokens;400 causal forwards/123,280 tokens. |

Counts of forward input tokens are inference/training call workloads, not unique
examples or FLOPs; teacher calls and repeated diagnostics are included. Earlier
stages lack complete forward instrumentation, so their total evaluation cost is
unavailable rather than inferred as zero. Training log/attempt budgets do not
measure energy, peak memory, or complete operator FLOPs. Hardware reports include
Windows/AMD64, PyTorch2.14.1+cpu, thread counts and batch shape; CPU model identifier
is a deferred metadata limitation, precluding cross-CPU latency comparisons.

## Observed independent audit outcomes

The integrated four-cell result is below. Intervals are descriptive 95% paired
whole-cluster percentile intervals, seed41/1,000 draws. Retrieval OOD has only four
clusters/32 rows; modular OOD has three clusters/six rows. They are not corrections
for adaptive selection or multiple comparisons.

| Task/model seed | Selected ID | Continued ID | Selected OOD | Continued OOD | OOD effect [interval] | Candidate gate |
|---|---:|---:|---:|---:|---|---|
| Retrieval23 |.250000|.260417|.468750|.437500|+.031250 [0,.111111]|rejected; baseline retained|
| Retrieval29 |.135417|.156250|0|0|0 [0,0]|rejected; baseline retained|
| Modular23 |.045455|.045455|0|0|0 [0,0]|accepted|
| Modular29 |.045455|.045455|0|0|0 [0,0]|accepted|

The positive retrieval effect reflects avoiding a harmful continuation in this
cell; it is not evidence of successful self-distillation. All four milestone
definitions remain right-censored at32 in all cells. Raw M/R crossing is unavailable
under task-only telemetry, not a zero onset. Earlier PR2/3 phase-enabled runs are
censored at6. No generalization transition was demonstrated.

Earlier results remain negative or inconclusive:

- Selection: audit OOD existing/larger/audit-reporting arms=.5625/.1875 by paired seed;
  conservative=.3125/.1875. Tiny fixtures do not establish a better acceptance rule.
- Causal retrieval fixture: audit OOD accuracy/correct invariance=.34375, required
  update=0. No observed grokking crossing to support a phase contrast.
- Milestones: no sustained milestone; the retrieval loss/next-accuracy correlation
  of+.879583 uses only three dependent pairs. Predictive validity remains unestablished.
- Blocks: zero active learned lessons after the utility gate. Audit OOD was0 except
  retrieval block2, where baseline/summary=.125 and replay=.03125. No accumulation gain.
- Resources: every arm had the same audit OOD within each cell (.1875/0 for retrieval,
  .25/0 for modular). Model storage was5,456/3,748 bytes; this is not peak RAM.
- Scaling: audit OOD ranged0–.53125. No model qualified for a capability-preserving
  compression baseline or thresholded horizon. Held-out RMSE varied. No geometry,
  data-saturation, efficiency or scaling-law conclusion follows.
- Credit: k2/kH did not consistently beat k1 across three seeds; full-horizon exact
  improvement identity held to numerical precision. See the complete seed table in
  [multi-step credit](multistep-credit.md).

Full ID and OOD results, failures, gates, onset histories, intervals and unavailable
reasons are preserved in the raw reports rather than compressed into a success rate.

## Compression reporting boundary

Logical FP16/INT8 activation rounding in scaling retains FP32 parameter storage.
No physical model reduction is claimed. `resources.model_bytes` measures unique
parameter/buffer bytes; peak RAM, energy and deployment memory remain unavailable.
Fake rounding has no codec decoder, so codec overhead is inapplicable; runtime
rounding overhead is not separately benchmarked in that experiment. Precision
reports retain paired ID/OOD behavior degradation and baseline capability flags.
Protected mechanism changes use the existing behavioral/CKA constraints and causal
diagnostics, not an assertion that internal mechanisms are proven preserved.
Exact ternary codecs remain separately documented in [storage precision](storage-precision.md).
No accepted deployment-compression result was established by this matrix.

## Final release gate and schema migration

Independent report schema2 adds `continued`, `selected_vs_continued`, and explicit
missing-control reasons; baseline/selected/candidate fields keep their meaning.
Old frozen runs remain valid and report an unavailable control. Milestone reporting
accepts schema1 and2. Source selection manifests and accepted checkpoints are unchanged.

The prepared final release binds each of the four integrated runs' complete frozen
selection manifests. Its primary endpoint is selected-minus-continued OOD accuracy;
secondary endpoints are ID accuracy, selection gaps and all candidate outcomes.
All listed seeds are retained, with no choice of a winner from the audit. It does
not authorize final studies of custom block/resource/scaling estimands or a pooled
claim across all mechanisms. These remain audit-only.

Before final inference, require explicit researcher authorization naming this
release and its SHA256. `release-final --manifest ... --authorization ... --sha256 ...`
validates every source first, records a one-use attempt before inference, and
preserves failures without retries. The authorization is a trusted-host attestation,
not an authentication service. Unit tests authorize disposable fixtures only.
The scientific release has not been authorized or evaluated. Prior exploratory
audits informed infrastructure development; neither those audits nor eventual
small final measurements warrant confirmatory general claims.
