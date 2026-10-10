# ADR 0011: retain matched controls and release independent evidence explicitly

Accepted for PR8. The ordinary continuation control already ran in consolidation
experiments but its model was discarded. Persist its checkpoint, fresh optimizer,
actual update/example/token budget and wall scope before selection freezes.
Report schema2 adds independent control measurements and paired selected-minus-
control effects. Preserve schema1 source compatibility and explicit missing values.

Integrate small, separately interpretable stage studies in a portable hash-bound
evidence archive rather than constructing a single composite fitness score.
Provide a full runnable matrix recipe. Retain all negative findings, failed and
rejected candidates, and censored/missing onsets. Avoid pooling heterogeneous tasks.

The final-release manifest fixes all standard-run candidates and analysis before
final evaluation, requires explicit named authorization plus its digest, validates
source hashes first, and uses a durable one-use attempt. It is a trusted-host
protocol, not hostile-code isolation. Custom block/resource/scaling final estimands
remain unregistered and audit-only. Researcher approval is still required for the
concrete scientific release; disposable test fixtures do not grant it.
