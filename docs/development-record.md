# Development and review record

The user supplied and approved the scientific design. Implementation followed
tests-first subsystem work, integration, independent review, regression tests and
executed experiment checks. The empty remote was cloned into a separate local
checkout; work began on `feat/agg-v0.1`.

Rulings:

- Use Python 3.12 as the floor because the installed NumPy type stubs require it.
- Use disjoint query-value bindings for retrieval ID validation. Randomly sampling
  sparse contexts allowed exact training rows into ID; the original construction
  was rejected after a reviewer reproduced 57/64 leaked observations.
- Require spare distance for an OOD condition. The maximum in-context distance
  cannot also serve as a strictly longer-distance condition.
- Keep negative scientific outcomes. Do not retune or smooth away the unstable
  single-seed arithmetic trajectory or censored retrieval onsets.
- Evaluate accepted stages against the original capability anchor. Per-stage
  reference drift would permit more total degradation than configured.
- Count bytes and latency separately. BITCOS-like payload savings do not imply a
  faster decoder or a faster model.

Independent review fixes included finite subnormal quantization, nonfinite OOD
rejection, behavioral configuration in state hashes, restored teacher module modes,
actual diagnostic bypasses, dimension flag handling, factorial boundary validation,
and baseline CLI removal of training-time adapters. Regression tests exercise each.

The implementation plan's six tasks were completed for the implemented harness;
the full research program remains subject to the limitations in verification.md.
