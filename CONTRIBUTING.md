# Contributing

Contributions should preserve scientific interpretability before optimizing compression or speed. Start with the approved brief and implementation specification in `docs/specs`, then read the relevant glossary and architecture document.

Use Python 3.12 or newer in an isolated environment. Install an editable development checkout with `python -m pip install -e ".[dev,geometry]"`; add `[topology]` for optional persistent homology. Do not silently replace an unavailable manifold or topology method with a different calculation.

Before submitting a substantive change, reproduce the behavior with a focused failing test, implement the smallest understandable change, and run:

```text
python -m pytest
python -m ruff check .
python -m mypy agg
```

Include relevant smoke/config runs from the README when changing experiment behavior. State the exact commands, environment and results. Tests should check analytic or independently observable behavior: known curves, lossless codec round trips, fail-closed constraints, exact rollback, conflict precedence and actual mechanism removal. Avoid tests that merely repeat a formula from the implementation without an independent expectation.

Keep proposals separate from execution and acceptance. Never mutate the accepted model while evaluating a candidate. Behavior and mechanism constraints must remain independently configurable. Preserve failed/rejected results and missing values; do not hide them behind averages or fabricate zeros.

New telemetry needs a definition, units, observation protocol, scope, availability behavior and proxy status. Preserve raw values, version incompatible schema changes, and document sampling limitations. New compression needs numerical acceptance tests and honest physical cost reporting. New evidence sources need provenance, conflict behavior and confidence assumptions.

A pull request should explain the concrete behavior change, scientific assumptions, validation, and limitations. Include deterministic seeds/configurations for experiments. A small test run supports implementation claims only. Claims of grokking, better geometry, accelerated consolidation or mechanistic preservation require the independent experiments described in `docs/experiment-matrix.md`.

Keep commits small and reviewable. Do not bundle generated large checkpoints or machine-specific virtual environments with source changes. Add primary references when borrowing an algorithm and distinguish exact reproduction from adapted ideas. Negative findings and rejected candidates are useful contributions.
