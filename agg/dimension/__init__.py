"""Linear representation spectra; these are not intrinsic-dimension estimates."""

from typing import Any

import torch
from torch import Tensor


def observation_matrix(x: Tensor) -> Tensor:
    """Flatten observation axes while preserving the final feature axis."""
    if x.ndim < 2 or x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("Expected a finite nonempty observation-feature tensor")
    return x.detach().to(dtype=torch.float64, device="cpu").reshape(-1, x.shape[-1])


def spectral_statistics(x: Tensor) -> dict[str, Any]:
    x = observation_matrix(x)
    singular = torch.linalg.svdvals(x - x.mean(dim=0))
    energy = singular.square()
    total = float(energy.sum())
    p = energy / total if total > 0 else torch.zeros_like(energy)
    positive = p[p > 0]
    return {
        "singular_values": singular.tolist(),
        "effective_rank": float(torch.exp(-(positive * positive.log()).sum())) if total else 0.0,
        "participation_ratio": float(1 / p.square().sum()) if total else 0.0,
        "explained_variance": p.cumsum(0).tolist(),
        "definition": "entropy rank of covariance spectrum; linear dimension proxy",
    }
