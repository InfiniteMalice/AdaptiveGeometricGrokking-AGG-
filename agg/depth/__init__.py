"""Finite differences on normalized depth and Euclidean trajectory diagnostics."""

from typing import Any

import torch
from torch import Tensor


def depth_derivatives(q: Tensor, t: Tensor | None = None) -> tuple[Tensor, Tensor]:
    q = q.detach().to(dtype=torch.float64, device="cpu")
    if q.ndim == 0 or len(q) < 3 or not torch.isfinite(q).all():
        raise ValueError("Derivatives need at least three finite depth samples")
    t = torch.linspace(0, 1, len(q), dtype=q.dtype) if t is None else t.double().cpu()
    if (
        t.ndim != 1
        or len(t) != len(q)
        or not torch.isfinite(t).all()
        or not torch.all(t[1:] > t[:-1])
        or t[0] < 0
        or t[-1] > 1
    ):
        raise ValueError("Depth coordinates must increase within [0, 1]")
    first = torch.gradient(q, spacing=(t,), dim=(0,), edge_order=2)[0]
    second = torch.gradient(first, spacing=(t,), dim=(0,), edge_order=2)[0]
    return first, second


def trajectory_statistics(hidden: Tensor) -> dict[str, Any]:
    """Aligned samples across layers; Euclidean quantities, no manifold claim."""
    if hidden.ndim < 2 or len(hidden) < 3 or not torch.isfinite(hidden).all():
        raise ValueError("Need at least three finite aligned representations")
    x = hidden.detach().double().cpu().flatten(start_dim=1)
    changes = x[1:] - x[:-1]
    lengths = changes.norm(dim=1)
    first, second = depth_derivatives(x)
    products = lengths[1:] * lengths[:-1]
    valid = products > 0
    angles = torch.full_like(products, float("nan"))
    angles[valid] = torch.acos(
        ((changes[1:] * changes[:-1]).sum(1)[valid] / products[valid]).clamp(-1, 1)
    )
    return {
        "path_length": float(lengths.sum()),
        "speed": first.norm(dim=1).tolist(),
        "acceleration_norm": second.norm(dim=1).tolist(),
        "turn_angles": [float(a) if torch.isfinite(a) else None for a in angles],
        "metric": "euclidean",
        "jacobian_norm": None,
    }
