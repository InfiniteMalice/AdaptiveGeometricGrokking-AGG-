"""Small-sample descriptors; graph and persistence operations are offline."""

from importlib import import_module
from typing import Any

import torch
from torch import Tensor

from agg.dimension import observation_matrix


def geometry_statistics(x: Tensor, max_samples: int = 256) -> dict[str, Any]:
    if max_samples < 2:
        raise ValueError("max_samples must be at least two")
    x = observation_matrix(x)[:max_samples]
    distances = torch.pdist(x)
    centered = x - x.mean(0)
    energy = torch.linalg.svdvals(centered).square()
    return {
        "sample_count": len(x),
        "sampling": "first observations, deterministic cap",
        "pairwise_distance_mean": float(distances.mean()) if len(distances) else None,
        "pairwise_distance_std": float(distances.std(unbiased=False)) if len(distances) else None,
        "representation_norm": float(x.norm(dim=1).mean()),
        "anisotropy": float(energy.max() / energy.sum()) if energy.sum() > 0 else 0.0,
        "curvature": None,
    }


def connectivity(x: Tensor, radius: float) -> dict[str, Any]:
    if not 0 <= radius < float("inf"):
        raise ValueError("radius must be finite and nonnegative")
    x = observation_matrix(x)
    adjacency = torch.cdist(x, x) <= radius
    adjacency.fill_diagonal_(False)
    unseen = set(range(len(x)))
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = set(torch.where(adjacency[node])[0].tolist()) & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    edges = int(adjacency.sum()) // 2
    return {
        "components": len(sizes),
        "component_sizes": sizes,
        "edges": edges,
        "graph_cycle_rank": edges - len(x) + len(sizes),
        "radius": radius,
        "method": "offline radius graph; cycle rank is not manifold homology",
    }


def persistence_summary(x: Tensor, maxdim: int = 1) -> dict[str, Any]:
    """Optional Vietoris-Rips persistence; never silently substitute another method."""
    if maxdim < 0:
        raise ValueError("maxdim must be nonnegative")
    try:
        ripser = import_module("ripser").ripser
    except ImportError as exc:
        raise ImportError("Install AGG topology support: pip install .[topology]") from exc
    diagrams = ripser(observation_matrix(x).numpy(), maxdim=maxdim)["dgms"]
    summaries = []
    for diagram in diagrams:
        lifetimes = diagram[:, 1] - diagram[:, 0]
        finite = lifetimes[torch.isfinite(torch.as_tensor(lifetimes)).numpy()]
        summaries.append(
            {
                "finite_lifetimes": finite.tolist(),
                "essential_classes": int(len(lifetimes) - len(finite)),
            }
        )
    return {"method": "offline vietoris-rips persistence", "dimensions": summaries}
