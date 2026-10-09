import math
from dataclasses import dataclass, field
from typing import Any

from agg.evaluation import Constraints
from agg.training import TrainConfig


@dataclass
class Features:
    telemetry: bool = True
    consolidation: bool = True
    geometry: bool = True
    dimension: bool = True
    gating: bool = True
    distillation: bool = True
    epiplexity: bool = True
    pruning: bool = True
    quantization: bool = True
    storage: bool = True
    executable: bool = True
    teacher: bool = True
    process: bool = True
    verified: bool = True

    @classmethod
    def baseline(cls) -> "Features":
        return cls(**dict.fromkeys(cls.__dataclass_fields__, False))


@dataclass
class ExperimentConfig:
    task: str = "modular"
    training: TrainConfig = field(default_factory=TrainConfig)
    features: Features = field(default_factory=Features)
    constraints: Constraints = field(default_factory=Constraints)
    modulus: int = 17
    tree_depth: int = 3
    samples: int = 128
    context_length: int = 12
    distance: int = 3
    density: float = 0.5
    hard_distractors: bool = False
    gate: float = 0.5
    learned_gate: bool = False
    curvature: float = 1.0
    learned_curvature: bool = False
    dimensions: tuple[int, ...] = (8, 16, 32)
    geometries: tuple[str, ...] = ("euclidean", "hyperbolic", "product")
    candidate_steps: int = 10
    prune_fraction: float = 0.2
    precisions: tuple[str, ...] = ("FP16", "BF16", "INT8", "INT4", "ternary")
    activation_precisions: tuple[str, ...] = ("INT8", "INT4")
    gamma: float = 0.9
    telemetry_mode: str = "full"
    adapt_during_training: bool = False
    storage_objective: str = "bytes"
    independent_evaluation: bool = False
    data_seed: int | None = None
    selection_samples: int | None = None
    selection_gain_floor: float | None = None

    def __post_init__(self) -> None:
        if type(self.independent_evaluation) is not bool:
            raise ValueError("independent_evaluation must be boolean")
        if self.independent_evaluation and self.data_seed is None:
            raise ValueError("independent evaluation requires an explicit data_seed")
        if self.data_seed is not None and (
            type(self.data_seed) is not int or not 0 <= self.data_seed < 2**32 - 4
        ):
            raise ValueError("data_seed must be an integer in [0, 2**32-4)")
        if self.selection_samples is not None and (
            not self.independent_evaluation
            or self.task != "retrieval"
            or type(self.selection_samples) is not int
            or self.selection_samples < 10
        ):
            raise ValueError("selection_samples requires independent retrieval and count >= 10")
        if self.selection_gain_floor is not None and (
            not self.independent_evaluation
            or type(self.selection_gain_floor) not in (int, float)
            or not math.isfinite(self.selection_gain_floor)
            or self.selection_gain_floor < 0
        ):
            raise ValueError("selection_gain_floor requires independent evaluation and finite >= 0")
        if self.task not in {"modular", "hierarchy", "retrieval"}:
            raise ValueError("task must be modular, hierarchy, or retrieval")
        if not self.dimensions or any(d < 1 or d > self.training.width for d in self.dimensions):
            raise ValueError("candidate dimensions must fit model width")
        if not self.geometries or set(self.geometries) - {"euclidean", "hyperbolic", "product"}:
            raise ValueError("unknown candidate geometry")
        if self.candidate_steps < 1 or not 0 <= self.gate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid candidate budget, gate, or gamma")
        if not math.isfinite(self.curvature) or self.curvature <= 0:
            raise ValueError("curvature must be positive and finite")
        if not 0 <= self.prune_fraction <= 1 or self.storage_objective not in {"bytes", "latency"}:
            raise ValueError("invalid pruning or storage objective")
        if self.telemetry_mode not in {
            "full",
            "task",
            "geometry",
            "geometry_derivatives",
            "attribution",
            "gates_horizon",
        }:
            raise ValueError("unknown telemetry mode")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        raw = dict(values)
        raw["training"] = TrainConfig(**raw.get("training", {}))
        raw["features"] = Features(**raw.get("features", {}))
        raw["constraints"] = Constraints(**raw.get("constraints", {}))
        for key in ("dimensions", "geometries", "precisions", "activation_precisions"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
