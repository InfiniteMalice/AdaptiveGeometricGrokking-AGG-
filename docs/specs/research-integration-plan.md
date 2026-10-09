# Independent evaluation implementation plan

> For agentic workers: use `superpowers:executing-plans` task by task. The approved
> brief authorizes continued implementation. Keep subsequent research stages in
> separate plans and PRs after these prerequisites pass.

**Goal:** Make independent evaluation auditable without exposing protected data
to candidate generation or changing legacy experiments.

**Architecture:** A four-role protocol produces an ordinary development `TaskData`.
The existing runner/trials perform selection. A frozen artifact manifest and a
separate reporting API evaluate private audit/final partitions after selection.

**Tech stack:** Python 3.12+, PyTorch, NumPy, pytest, Ruff, mypy; no new dependency.

**Spec:** `docs/specs/research-integration.md`.

## Global constraints

- Existing configs retain existing behavior; new evaluation is opt-in.
- Preserve fixed capability anchors, copy trials, checkpoint immutability and RNG.
- Never substitute zeros for missing evidence or infer physical cost from precision.
- No audit/final evidence in proposal generation, ranking, acceptance or callbacks.
- No automatic merges, rewards or scientific claims from smoke fixtures.

## Review focus

- Variants and distractors can leak protected structures despite different rows.
- A failed baseline evaluator can erase the record of a proposed candidate.
- Rejected and failed candidates can disappear from winner-only reporting.
- Artifact alteration after selection can make a final score refer to another run.
- Few clusters, correlated pairs and absent classes can give misleading intervals.

### Task 1: Structural roles and development-only interface

Files: create `agg/tasks/protocol.py`, `tests/test_evaluation_protocol.py`;
modify `agg/experiments/config.py`, `runner.py`.

Produces: role-tagged `EvaluationProtocol`, `RoleData`, deterministic factory,
development `TaskData`, serializable partition manifest; separate `data_seed`.
Consumes: existing synthetic generators and tensor types.

- [x] Write tests for reproducibility, disjoint commuted modular groups, disjoint
  hierarchy subtrees, retrieval keys including distractors, inadequate groups,
  and no protected tensors in the development interface.
- [x] Run focused tests and observe missing API failure.
- [x] Implement deterministic grouping and typed opt-in config; leave legacy
  `make_data` unchanged when disabled. Use structural strata for ID/OOD.
- [x] Run focused tests, then full pytest, Ruff and mypy; commit.

### Task 2: Candidate attempt records and frozen checkpoints

Files: modify `agg/consolidation/trials.py`, `agg/experiments/runner.py`;
create `agg/experiments/independent.py`, extend protocol tests.

Produces: append-only attempt/result records, candidate checkpoint artifacts,
frozen run manifest; parent/optimizer/budget provenance and reuse counts.
Consumes: existing `trial`, `state_hash`, `save_model`, `Ledger`.

- [x] Test accepted, rejected, apply-failed and baseline-evaluator-failed attempts;
  verify the original checkpoint/hash and RNG remain unchanged.
- [x] Observe failure; add optional attempt sink and candidate snapshot hook.
  Persist intent before work and outcomes without changing legacy ledger schema.
- [x] Freeze all artifact hashes after the selector completes. Serialize failures
  as null scores with reasons. Validate manifest integrity before reporting.
- [x] Run focused tests and complete repository checks; commit.

### Task 3: Independent paired reporting and explicit final evaluation

Files: create `agg/evaluation/independent.py`; extend independent runner module,
CLI, protocol tests and `docs/independent-evaluation.md`; add small config.

Produces: selection/audit gaps, paired cluster-bootstrap summaries, separate
authorized final report. Consumes: frozen manifests, checkpoint loader, role data.

- [x] Test an oracle, constant-wrong predictions, unequal cluster sizes, one
  cluster, paired ordering, final authorization denial, stale artifacts, and audit
  invariance of selected checkpoint.
- [x] Observe failures; implement reports with explicit estimand, seed, counts,
  uncertainty and missingness. Prevent final access without authorization.
- [x] Train a tiny real model, run bounded candidates and independent audit,
  validate rejected/failed accounting, then explicitly evaluate fixture final data.
- [x] Record commands/results/limits, run full checks and a fresh branch review;
  commit and open PR-1 against PR-0. No automatic merge.

### Later-stage handoff

After PR-1 passes, each stage in the integration contract needs its own bounded
plan and tests. Scientific prerequisites for rewards, latent prediction and SCP
are not satisfied by this implementation plan. PR-8 final scientific evaluation
needs frozen hypotheses and an independently designated final authorization;
fixture authorization does not authorize a scientific final-data release.
