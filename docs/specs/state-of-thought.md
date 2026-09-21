# SoT-inspired control extension

## Intent and constraints

Extend main at `b1da38e` with compact reasoning telemetry, state-conditioned
activation of existing abstractions, auditable anomalies, and bounded compute
recommendations. The user's supplied specification is the requirements source.
No reward, model architecture, acceptance criteria, provider authority, or
rollback behavior changes. Existing configs disable the compute extension.

## Design

1. `Reasoning` contains four optional state coordinates plus optional routing
   diagnostics. `Observation` writes `agg.controller/2`; its loader explicitly
   migrates v1, leaving new measurements absent. TemporalTelemetry already
   traverses scalar categories, so no second derivative implementation is needed.
2. Pure torch extraction utilities define units and tensor shapes, detach inputs,
   reject nonfinite values, and return None for undefined trajectory direction.
3. The registry supplies a read-only eligibility snapshot. An external validity
   assessment is separate from registry applicability and router relevance.
   A router protocol consumes immutable IDs, profiles and selected control
   features, never a model or writable registry. Reference similarity uses
   explicitly configured coordinate scales and can consume temporal derivatives.
4. A routing harness validates selections and appends a complete decision to
   EventLog before returning any active subset. Audits retain state, eligible,
   selected, dropped, invalid, unknown and unavailable IDs, record hashes/versions,
   scores, configurations and provenance. Downstream decisions are linked by a
   subsequent event. JSONL is append-only through this API, not tamper-proof storage.
5. Pure diagnostics compare adjacent decisions; no anomaly is called deception.
   Compute policy is opt-in, uses existing progress confidence and risk handling,
   and leaves mutations to Controller.start/finish. Stop and continue are explicit
   recommendations without execution authority; unsupported actions remain so.
6. Control-only perturbations preserve original observations and provenance.
   A deterministic synthetic experiment compares absolute, first-order,
   second-order and state-independent controls using the actual router and
   TemporalTelemetry. Metrics are routing/compute proxies, not scientific validation.

## Alternatives considered

A separate learned agent framework would duplicate existing authority boundaries.
Putting reasoning metrics into geometry.metrics would obscure semantics and units.
Choose a dedicated category, small modules and an explicit v1 loader instead.

## Verification and risks

Baseline: 207 tests passed before changes (one existing PyTorch warning).
Tests cover schema migration, missingness, domains, numerical extremes, registry
immutability, external validity, routing response, audit failures, oscillation,
ablation/replay, opt-in policy and existing transaction guards. Final commands:
`python -m pytest -q`, `python -m ruff check .`, `python -m mypy agg`, and
`python -m agg.controller.reasoning_demo --output <new-directory>`.

Risks: selection can discard conflicting evidence; audit makes it visible but
cannot prevent it. Relevance profiles and validity attestations are trusted host
inputs. Stable hashes require the host to retain original evidence versions.
The toy transformer is not an LLM reasoning engine. No learned selector, JEV
implementation, LLM integration, or live stop/retrieval provider is implied.
