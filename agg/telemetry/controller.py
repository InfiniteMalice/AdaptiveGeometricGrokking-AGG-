"""Typed controller observations. Missing evidence is None, never an implicit zero.

These signals are observations, not reward terms. Step is the temporal coordinate;
layer statistics retain their own names and are never treated as time samples.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, get_type_hints

from agg.telemetry import TelemetrySnapshot


@dataclass(frozen=True)
class Performance:
    task_loss: float | None = None
    task_accuracy: float | None = None
    reward: float | None = None
    validation_score: float | None = None
    acquisition_score: float | None = None
    improvement_rate: float | None = None
    ood_score: float | None = None
    memorization_score: float | None = None


@dataclass(frozen=True)
class Geometry:
    representation_drift: float | None = None
    gradient_norm: float | None = None
    update_norm: float | None = None
    update_cosine: float | None = None
    effective_rank: float | None = None
    singular_values: tuple[float, ...] = ()
    layer_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Attribution:
    concentration: float | None = None
    entropy: float | None = None
    drift: float | None = None
    layer_distribution: dict[str, float] = field(default_factory=dict)
    importance_change: dict[str, float] = field(default_factory=dict)
    causal_contributions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Gating:
    activation_mean: float | None = None
    saturation: float | None = None
    effective_horizon: float | None = None
    attention_horizon: float | None = None
    memory_horizon: float | None = None
    think_frequency: float | None = None
    routing_frequency: dict[str, float] = field(default_factory=dict)
    tool_routing_frequency: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Resources:
    sparsity: float | None = None
    zero_density: float | None = None
    active_parameter_fraction: float | None = None
    precision_mode: str | None = None
    quantization_state: str | None = None
    storage_encoding: str | None = None
    adapter_utilization: float | None = None
    lora_rank_utilization: float | None = None


@dataclass(frozen=True)
class Continual:
    acquisition_score: float | None = None
    retained_performance: float | None = None
    forgetting: float | None = None
    retention_slope: float | None = None
    memory_half_life: float | None = None
    replay_effectiveness: float | None = None
    replay_diversity: float | None = None
    run_variance: float | None = None
    capability_regression: float | None = None
    exact_retention: float | None = None
    near_transfer: float | None = None
    far_transfer: float | None = None


@dataclass(frozen=True)
class AbstractionSignals:
    coverage: float | None = None
    confidence: float | None = None
    predictive_utility: float | None = None
    false_applicability_rate: float | None = None
    retrieval_miss_rate: float | None = None
    retrieval_false_positive_rate: float | None = None
    composition_failure_rate: float | None = None
    contradiction_rate: float | None = None
    regression: float | None = None
    statuses: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Search:
    branch_diversity: float | None = None
    depth: float | None = None
    exploration_budget: float | None = None
    exploit_explore_ratio: float | None = None
    marginal_gain_per_compute: float | None = None
    dead_end_frequency: float | None = None
    rollback_frequency: float | None = None
    intervention_effectiveness: float | None = None
    difficulty: float | None = None
    uncertainty: float | None = None
    risk: float | None = None


CATEGORY_TYPES = {
    "performance": Performance,
    "geometry": Geometry,
    "attribution": Attribution,
    "gating": Gating,
    "resources": Resources,
    "continual": Continual,
    "abstraction": AbstractionSignals,
    "search": Search,
}


@dataclass(frozen=True)
class Observation:
    step: int
    performance: Performance = field(default_factory=Performance)
    geometry: Geometry = field(default_factory=Geometry)
    attribution: Attribution = field(default_factory=Attribution)
    gating: Gating = field(default_factory=Gating)
    resources: Resources = field(default_factory=Resources)
    continual: Continual = field(default_factory=Continual)
    abstraction: AbstractionSignals = field(default_factory=AbstractionSignals)
    search: Search = field(default_factory=Search)
    component: str = "model"
    timestamp: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    proxy_metrics: tuple[str, ...] = ()
    out_of_distribution: bool = False
    contradictory: bool = False
    schema_version: str = "agg.controller/1"

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0 or not self.component:
            raise ValueError("step must be a nonnegative integer and component must be named")
        if self.schema_version != "agg.controller/1":
            raise ValueError("unsupported controller telemetry schema")
        for name, kind in CATEGORY_TYPES.items():
            category = getattr(self, name)
            if not isinstance(category, kind):
                raise ValueError(f"{name} must be {kind.__name__}")
            for key, hint in get_type_hints(kind).items():
                value = getattr(category, key)
                if hint == float | None and value is not None:
                    if type(value) not in (float, int) or not math.isfinite(value):
                        raise ValueError(f"{name}.{key} must be a finite numeric measurement")
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Observation":
        raw = dict(value)
        for name, kind in CATEGORY_TYPES.items():
            fields = dict(raw.get(name, {}))
            if name == "geometry" and "singular_values" in fields:
                fields["singular_values"] = tuple(fields["singular_values"])
            raw[name] = kind(**fields)
        raw["proxy_metrics"] = tuple(raw.get("proxy_metrics", ()))
        return cls(**raw)

    def metrics(self) -> dict[str, float]:
        """Only finite measured scalars enter temporal analysis; labels stay in raw data."""
        result: dict[str, float] = {}

        def walk(prefix: str, value: Any) -> None:
            if type(value) in (int, float):
                result[prefix] = float(value)
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk(f"{prefix}.{key}", item)

        for name in CATEGORY_TYPES:
            walk(name, asdict(getattr(self, name)))
        return result

    @classmethod
    def from_legacy(cls, snapshot: TelemetrySnapshot) -> "Observation":
        def metric(category: str, key: str) -> float | None:
            value = snapshot.raw.get(category)
            item = value.get(key) if isinstance(value, dict) else None
            if isinstance(item, (float, int)) and not isinstance(item, bool):
                return float(item)
            return None

        return cls(
            snapshot.step,
            performance=Performance(
                task_loss=metric("task", "loss"),
                task_accuracy=metric("task", "accuracy"),
                validation_score=metric("id", "accuracy"),
                ood_score=metric("ood", "accuracy"),
                memorization_score=metric("memorization", "score"),
            ),
            geometry=Geometry(effective_rank=metric("dimension", "effective_rank")),
            component=f"{snapshot.component}/layer-{snapshot.layer}"
            if snapshot.layer is not None
            else snapshot.component,
            provenance={"legacy": snapshot.to_dict()},
            proxy_metrics=tuple(
                name
                for name, category in (
                    ("performance.memorization_score", "memorization"),
                    ("geometry.effective_rank", "dimension"),
                )
                if snapshot.proxy.get(category)
            ),
        )
