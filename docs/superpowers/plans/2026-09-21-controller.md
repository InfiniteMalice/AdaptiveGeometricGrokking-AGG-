# AGG controller implementation plan

**Goal:** Add observable, bounded, reversible adaptive control to the existing
research package without changing its objectives.

**Architecture:** Extend telemetry with typed category records and explicit legacy
conversion. Separate temporal analysis, diagnosis, proposal policy and execution.
Reuse the ledger and copy-based model trial; implement memory/curriculum as generic
interfaces because the repository has no live engines for them.

**Tech stack:** Python 3.12, existing NumPy/PyTorch, dataclasses, pytest, Ruff, mypy.
**Spec:** `docs/specs/controller.md` and the user's full AGG request.
**Execution:** Implement in this isolated clone, with tests before product code.

## Global constraints

Telemetry is not reward. Preserve legacy snapshots and experimental entrypoints.
Only allow-listed numeric configuration can be proposed for mutation. Unknown and
unsupported actions remain visible. Protected metrics are independent constraints.

## Review focus

1. Out-of-order or sparse observations must not create confident trends.
2. A caller cannot edit a proposal to bypass controller constraints.
3. Missing protected measurements and exceptions must not result in acceptance.
4. Repeated trials must not consume regression tolerance cumulatively.
5. Provider failure must retain the accepted model or block further execution.

## Tasks

- [x] Add failing tests in `tests/test_controller_telemetry.py` for typed schema
  round trips, legacy conversion, finite values, irregular quadratic derivatives,
  noisy plateau and temporal support. Implement `agg/telemetry/controller.py`
  and `agg/controller/temporal.py`; centralize defaults in `config.py`.
- [x] Add `tests/test_controller_policy.py` for multiple diagnoses, uncertainty,
  proportional proposals and expensive-action evidence. Implement `diagnosis.py`
  and `policy.py`, including action enums and replaceable provider protocols.
- [x] Add `tests/test_controller_lifecycle.py` for validation, one pending trial,
  cooldown, target/protected gates, rollback, stale proposals and exceptions.
  Implement `core.py`, `events.py` and `evaluation.py` using the existing ledger.
- [x] Add `tests/test_controller_memory.py` for overlapping scopes, false
  applicability, retrieval versus knowledge failure, prioritized composition,
  retention curves and shallowest-stage backtracking. Implement `memory.py`
  and `curriculum.py`, including future repair interfaces.
- [x] Add `tests/test_controller_integration.py` for the four-phase synthetic
  regression scenario and a real copied-model trial. Implement `training.py`
  using `train`, `trial`, existing `Constraints`, and the training callback.
- [x] Add `agg/controller/demo.py`, JSON config and architecture/extension docs.
  Run `python -m agg.controller.demo --output <new directory>` and inspect events.
- [x] Run `python -m pytest -q`, `python -m ruff check .`, `python -m mypy agg`.
  Review the final diff, document actual results and unsupported actions, and
  package the source changes and report for the user.
