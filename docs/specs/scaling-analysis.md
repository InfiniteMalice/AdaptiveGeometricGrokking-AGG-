# Scaling analysis plan (PR-6A)

Implement an opt-in controlled CPU retrieval grid with fixed training roles,
shared selection/audit data and train-only volume subsets. Fit smaller widths only;
predeclare a larger held-out width. Euclidean/hyperbolic/product adapters use the
existing matched scalar controls, same adapter dimension/gate and optimizer update
and token budgets. Report actual trainable/total parameters and storage differences;
do not call equal updates equal FLOPs. Training FLOPs remain unmeasured with reason.
Record wall execution, updates/tokens, unique available/observed rows, resource scope.

Fit predeclared power-error (log error versus log parameters) and log-linear-error
forms separately by geometry/data volume/budget. Do not choose forms on larger data.
Bootstrap whole model seeds across widths; report sparse-seed limitations, coefficient
intervals, per-cell residuals, held-out mean predictions and extrapolation errors.
Do not clip out-of-range extrapolations; flag failed/zero/rank-deficient fits explicitly.
Saturation uses observed volume contrast and intervals, not an assumed scaling law.

Freeze training and selection before audit-only scaling analysis. Larger model scores
must be excluded from every fit/resample. Audit also measures oracle-valid distance
moves and sampled effective horizons using existing horizon helper (missing when no
distance qualifies), causal invariance/update diagnostics from PR-2, and FP16/INT8
simulated compression tolerance versus exact paired baseline. Simulations are copies
and not retained; all configurations/results and budgets remain visible. Final data
untouched. No predictor in PR6B without an independent hypothesis beyond ToyEnvironment.

Tasks: 1 statistics module + known-law, withheld larger/outliers, zero errors,
sparse seeds/singular data tests RED->GREEN; 2 real controlled geometry/volume grid,
freeze/audit protection and actual count tests; 3 CLI/docs CPU smoke, full checks,
fresh reviewer, one fix pass, stacked PR. No new dependencies.


Concrete default grid: fit widths8/12/16/20, held-out width24; geometry3; seeds17/19;
training volume16/64 rows from one seed149 protocol; budgets4/8updates; batch16,
context8, adapterdimension4, fixedgate0.5, learnedcurvature true with matched
Euclidean bandwidthscalar. 120 independently initialized cells,720updates,92160
trainingtokens. Default primary response auditOODerror; predeclare forms power
(log(error)=a+b log(parameters)) and log-linear(error=a+b log(parameters)). Positive-
error domain failure explicit; no zero substitution. Whole-seed bootstrap200draws;
with<3seeds intervals descriptive and sparse support explicitlywarned.
