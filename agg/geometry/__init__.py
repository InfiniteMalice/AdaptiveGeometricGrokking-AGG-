"""Prototype-distance adapters; geometry affects routing before the Euclidean lift.

Projected inputs and learned prototypes are tangent coordinates at the origin.
Hyperbolic distances use Geoopt's Poincare ball of sectional curvature -c.
Product distances sum squared Euclidean and hyperbolic component distances.
Softmax routing interpolates *tangent* prototypes, then lifts to model width.
This is a manifold-distance kernel, not a Riemannian barycenter or a cancelling
expmap/logmap adapter. Equal numbers of prototypes ensure matched budgets.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from agg.gating import ScalarGate
from agg.models import TinyTransformer


class GeometryAdapter(nn.Module):
    def __init__(
        self,
        width: int,
        dimension: int,
        geometry: str = "euclidean",
        gate: float = 0.5,
        learned_gate: bool = False,
        curvature: float = 1.0,
        learned_curvature: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= dimension <= width:
            raise ValueError("dimension must lie between 1 and model width")
        if geometry not in {"euclidean", "hyperbolic", "product"}:
            raise ValueError("unknown geometry")
        if geometry == "product" and dimension < 2:
            raise ValueError("product geometry needs at least two dimensions")
        if not math.isfinite(curvature) or curvature <= 0:
            raise ValueError("curvature magnitude must be positive and finite")
        self.width, self.dimension, self.geometry = width, dimension, geometry
        self.project = nn.Linear(width, dimension)
        self.lift = nn.Linear(dimension, width)
        self.prototypes = nn.Parameter(torch.randn(dimension, dimension) / math.sqrt(dimension))
        self.gate = ScalarGate(gate, learned_gate)
        self.manifold: Any = None
        if geometry != "euclidean":
            try:
                import geoopt
            except ImportError as exc:
                raise ImportError('Curved geometry requires pip install -e ".[geometry]"') from exc
            self.manifold = geoopt.PoincareBall(c=curvature, learnable=learned_curvature)
        # An active kernel bandwidth provides the Euclidean matched scalar control.
        if geometry == "euclidean" and learned_curvature:
            raw = curvature + math.log(-math.expm1(-curvature))
            self.raw_scale = nn.Parameter(torch.tensor(raw))
        else:
            self.register_parameter("raw_scale", None)

    def squared_distances(self, z: Tensor) -> Tensor:
        p = self.prototypes
        if self.geometry == "euclidean":
            scale = 1.0 if self.raw_scale is None else F.softplus(self.raw_scale)
            return ((z.unsqueeze(-2) - p) ** 2).sum(-1) * scale
        split = self.dimension // 2 if self.geometry == "product" else 0
        points = self.manifold.expmap0(z[..., split:])
        prototypes = self.manifold.expmap0(p[..., split:])
        # Poincare metric has factor 2 at origin: /4 matches Euclidean c->0 scale.
        curved = self.manifold.dist(points.unsqueeze(-2), prototypes).square() / 4
        if split:
            curved = curved + ((z[..., :split].unsqueeze(-2) - p[..., :split]) ** 2).sum(-1)
        return curved

    def forward(self, h: Tensor) -> Tensor:
        if h.shape[-1] != self.width:
            raise ValueError("input width does not match adapter")
        # A fixed/forced zero really removes the adapted computation, including NaNs.
        if not self.gate.value().requires_grad and self.gate.value().item() == 0:
            return h
        z = self.project(h)
        weights = torch.softmax(-self.squared_distances(z) / math.sqrt(self.dimension), dim=-1)
        adapted = self.lift(weights @ self.prototypes)
        return self.gate(h, adapted)


class LayerGeometryAdapters(nn.Module):
    """Explicit index-based adapters for insertion after individual model blocks."""

    def __init__(self, adapters: Sequence[GeometryAdapter]) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(adapters)

    def forward(self, hidden: Tensor, layer: int) -> Tensor:
        return self.adapters[layer](hidden)


class AdaptedModel(nn.Module):
    """TinyTransformer wrapper adapting final pre-normalization hidden states."""

    def __init__(self, base: TinyTransformer, adapter: GeometryAdapter) -> None:
        super().__init__()
        self.base, self.adapter = base, adapter

    def forward(self, tokens: Tensor, *, return_hidden: bool = False) -> Any:
        _, hidden = self.base(tokens, return_hidden=True)
        adapted = self.adapter(hidden[-1])
        logits = self.base.readout(self.base.norm(adapted[:, -1]))
        return (logits, [*hidden, adapted]) if return_hidden else logits
