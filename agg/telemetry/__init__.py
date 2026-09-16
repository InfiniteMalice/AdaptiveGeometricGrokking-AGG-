"""Versioned, inspectable telemetry with explicit missing observations and proxies."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from agg.depth import depth_derivatives
from agg.dimension import spectral_statistics
from agg.topology import geometry_statistics

CATEGORIES = (
    "task",
    "id",
    "ood",
    "attribution",
    "geometry",
    "topology",
    "dimension",
    "depth_first",
    "depth_second",
    "gates",
    "horizon",
    "memorization",
    "retrieval",
    "generalization",
    "complexity",
    "sparsity",
    "precision",
    "storage",
    "execution",
)
PROXIES = {"dimension", "memorization", "retrieval", "generalization", "complexity"}


@dataclass
class TelemetrySnapshot:
    step: int
    checkpoint: str
    layer: int | None = None
    component: str = "model"
    raw: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    availability: dict[str, bool] = field(default_factory=dict)
    proxy: dict[str, bool] = field(default_factory=dict)
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.step < 0 or self.schema_version != "0.1":
            raise ValueError("Unsupported schema version or negative step")
        unknown = (set(self.raw) | set(self.derived)) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown telemetry categories: {sorted(unknown)}")
        for category in CATEGORIES:
            self.raw.setdefault(category, None)
            self.derived.setdefault(category, None)
            self.availability[category] = (
                self.raw[category] is not None or self.derived[category] is not None
            )
            self.proxy.setdefault(category, category in PROXIES)
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryHistory:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, snapshot: TelemetrySnapshot) -> None:
        encoded = json.dumps(snapshot.to_dict(), allow_nan=False, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def read(self) -> list[TelemetrySnapshot]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            return [TelemetrySnapshot(**json.loads(line)) for line in stream if line.strip()]


class TelemetryCollector:
    """Metrics use category keys. Hidden tensors are aligned layer observations."""

    def collect(
        self,
        step: int,
        checkpoint: str,
        metrics: Mapping[str, Any],
        hidden: Sequence[Tensor] | None = None,
        component: str = "model",
        *,
        derivatives: bool = True,
    ) -> list[TelemetrySnapshot]:
        if hidden is None or len(hidden) == 0:
            return [TelemetrySnapshot(step, checkpoint, component=component, raw=dict(metrics))]
        snapshots = []
        for layer, tensor in enumerate(hidden):
            raw = dict(metrics)
            raw["dimension"] = spectral_statistics(tensor)
            raw["geometry"] = geometry_statistics(tensor)
            snapshots.append(TelemetrySnapshot(step, checkpoint, layer, component, raw))
        if derivatives and len(snapshots) >= 3:
            ranks = torch.tensor([s.raw["dimension"]["effective_rank"] for s in snapshots])
            first, second = depth_derivatives(ranks)
            for i, snapshot in enumerate(snapshots):
                snapshot.derived["depth_first"] = {"effective_rank": float(first[i])}
                snapshot.derived["depth_second"] = {"effective_rank": float(second[i])}
                snapshot.availability["depth_first"] = True
                snapshot.availability["depth_second"] = True
                snapshot.proxy["depth_first"] = True
                snapshot.proxy["depth_second"] = True
        return snapshots
