"""Central controller thresholds, in metric units per training step unless stated."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .resources import ResourceProfile


@dataclass(frozen=True)
class TemporalConfig:
    window: int = 9
    min_samples: int = 5
    ema_alpha: float = 0.3
    plateau_slope: float = 0.001
    acceleration_threshold: float = 0.0001
    noise_threshold: float = 0.05
    max_gap: int = 1000

    def __post_init__(self) -> None:
        if any(type(n) is not int for n in (self.window, self.min_samples, self.max_gap)):
            raise ValueError("temporal sample counts, window and max_gap must be integers")
        if not 5 <= self.min_samples <= self.window or self.max_gap < 1:
            raise ValueError("window >= min_samples >= 5 and positive max_gap required")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0,1]")
        for value in (self.plateau_slope, self.acceleration_threshold, self.noise_threshold):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("temporal thresholds must be finite and positive")


@dataclass(frozen=True)
class MetricGuard:
    name: str
    tolerance: float = 0.02
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("guard requires a name and finite nonnegative tolerance")


@dataclass(frozen=True)
class ReasoningPolicyConfig:
    """Reference-policy thresholds; entropy uses producer units (default nats)."""

    enabled: bool = False
    entropy_low: float = 0.2
    entropy_high: float = 1.0
    displacement_floor: float = 0.01
    minimum_support: float = 0.25
    maximum_turnover: float = 0.75
    allow_stop: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.allow_stop) is not bool:
            raise ValueError("reasoning enabled and allow_stop must be booleans")
        for name in (
            "entropy_low",
            "entropy_high",
            "displacement_floor",
            "minimum_support",
            "maximum_turnover",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.entropy_high <= self.entropy_low:
            raise ValueError("entropy_high must exceed entropy_low")
        if self.minimum_support > 1 or self.maximum_turnover > 1:
            raise ValueError("support and turnover thresholds must be in [0,1]")


@dataclass(frozen=True)
class ControllerConfig:
    resources: ResourceProfile | None = None
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    confidence_threshold: float = 0.7
    expensive_confidence: float = 0.9
    proxy_confidence_cap: float = 0.6
    cooldown: int = 20
    evaluation_window: int = 20
    evaluation_samples: int = 2
    minimum_gain: float = 0.005
    max_fraction: float = 0.15
    regression_slope: float = 0.002
    rollback_threshold: float = 0.1
    drift_threshold: float = 0.2
    error_rate_threshold: float = 0.2
    saturation_threshold: float = 0.95
    coverage_threshold: float = 0.6
    diversity_threshold: float = 0.2
    marginal_gain_threshold: float = 0.001
    abstraction_activation_threshold: float = 0.7
    minimize_metrics: tuple[str, ...] = (
        "performance.task_loss",
        "geometry.representation_drift",
        "abstraction.retrieval_miss_rate",
        "abstraction.false_applicability_rate",
        "abstraction.composition_failure_rate",
        "abstraction.contradiction_rate",
    )
    protected_metrics: tuple[MetricGuard, ...] = (MetricGuard("performance.ood_score"),)
    reasoning: ReasoningPolicyConfig = field(default_factory=ReasoningPolicyConfig)

    def __post_init__(self) -> None:
        if self.resources is not None and not isinstance(self.resources, ResourceProfile):
            raise ValueError("resources must be ResourceProfile or None")
        if not isinstance(self.reasoning, ReasoningPolicyConfig):
            raise ValueError("reasoning must be ReasoningPolicyConfig")
        if any(
            type(n) is not int
            for n in (self.cooldown, self.evaluation_window, self.evaluation_samples)
        ):
            raise ValueError("controller step and evaluation counts must be integers")
        for name in (
            "confidence_threshold",
            "expensive_confidence",
            "proxy_confidence_cap",
            "drift_threshold",
            "error_rate_threshold",
            "saturation_threshold",
            "coverage_threshold",
            "diversity_threshold",
            "abstraction_activation_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.expensive_confidence < self.confidence_threshold:
            raise ValueError("expensive interventions require stronger evidence")
        for name in (
            "minimum_gain",
            "regression_slope",
            "rollback_threshold",
            "marginal_gain_threshold",
            "max_fraction",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_fraction > 1 or self.cooldown < 0:
            raise ValueError("max_fraction <= 1 and cooldown >= 0 required")
        if self.evaluation_window < 1 or self.evaluation_samples < 1:
            raise ValueError("positive evaluation window and sample count required")
        if len({g.name for g in self.protected_metrics}) != len(self.protected_metrics):
            raise ValueError("protected metric names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ControllerConfig":
        raw = dict(value)
        if raw.get("resources") is not None:
            raw["resources"] = ResourceProfile(**raw["resources"])
        raw["temporal"] = TemporalConfig(**raw.get("temporal", {}))
        raw["reasoning"] = ReasoningPolicyConfig(**raw.get("reasoning", {}))
        if "minimize_metrics" in raw:
            raw["minimize_metrics"] = tuple(raw["minimize_metrics"])
        if "protected_metrics" in raw:
            raw["protected_metrics"] = tuple(MetricGuard(**g) for g in raw["protected_metrics"])
        return cls(**raw)
