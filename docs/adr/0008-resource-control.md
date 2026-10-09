# ADR 0008: Conjunctive measured resource control

Status: accepted for opt-in research.

Keep existing capability anchors, target improvement and rollback as independent
mandatory gates. Add absolute resource requirements and individual margins; never
combine them into a compensatory score. Compare latency only among verified feasible
candidates. CPU execution and unique tensor-storage measurements have explicit
scope; unavailable peak memory/FLOPs cannot pass requested limits. Existing bounded
LR/regularization execution is reused, leaving unsupported action labels unsupported.

Earlier selection-only experience proposes conditions, never authorizes execution.
Every target task and hardware is remeasured. Matrix audit remains separate and
frozen. The cost is limited evidence about adaptive inference architecture: current
experiments can test feasibility and reject unsafe changes, but cannot establish
new inference-depth or precision savings without additional real providers.
