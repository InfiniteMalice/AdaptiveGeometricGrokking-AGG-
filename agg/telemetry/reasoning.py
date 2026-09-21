"""Pure SoT-inspired readouts, independent of control and autograd.

These are explicit reference definitions, not a reproduction of the paper's
representation readout. Hosts choose representations, aggregation and step units.
Inputs are finite real tensors; calculations use detached CPU float64 values.
Overflow of a result is rejected instead of emitting an infinite measurement.
"""

import math

import torch


def _tensor(value: torch.Tensor) -> torch.Tensor:
    if value.is_complex() or value.dtype == torch.bool or value.numel() == 0:
        raise ValueError("nonempty real numeric tensor required")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("representations and logits must be finite")
    return result


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("readout exceeds finite float64 range")
    return value


def representation_dispersion(representations: torch.Tensor) -> float:
    """Mean squared Euclidean distance to centroid for [items, features].

    Units are squared representation units; no feature/dimension normalization.
    One item or identical items have zero dispersion. Empty axes are rejected.
    Scaling inside the calculation prevents intermediate square overflow.
    """
    x = _tensor(representations)
    if x.ndim != 2:
        raise ValueError("representations must have shape [items, features]")
    scale = float(x.abs().max())
    if scale == 0:
        return 0.0
    normalized = x / scale
    # Translation before reduction makes identical rows exactly zero, including
    # very large unequal coordinates where centroid rounding would be amplified.
    centered = normalized - normalized[0]
    variance = float(((centered - centered.mean(dim=0)) ** 2).sum(dim=1).mean())
    radius = math.sqrt(variance) * scale
    return _finite(radius * radius)


def _vectors(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = _tensor(left), _tensor(right)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("states/transitions must be matching nonempty [features] vectors")
    return a, b


def state_displacement(previous: torch.Tensor, current: torch.Tensor) -> float:
    """Euclidean distance between [features] states, in representation units."""
    a, b = _vectors(previous, current)
    delta = b - a
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("state difference exceeds finite float64 range")
    return _finite(math.hypot(*delta.tolist()))


def trajectory_cosine(
    previous_transition: torch.Tensor, current_transition: torch.Tensor
) -> float | None:
    """Cosine between [features] transitions; None if either has zero length.

    Inputs are transitions (s[t-1]-s[t-2], s[t]-s[t-1]), not absolute states.
    No epsilon threshold silently reclassifies a small nonzero transition.
    """
    a, b = _vectors(previous_transition, current_transition)
    sa, sb = float(a.abs().max()), float(b.abs().max())
    if sa == 0 or sb == 0:
        return None
    a, b = a / sa, b / sb
    cosine = float(torch.dot(a / torch.linalg.vector_norm(a), b / torch.linalg.vector_norm(b)))
    return max(-1.0, min(1.0, _finite(cosine)))


def predictive_entropy(logits: torch.Tensor) -> float:
    """Mean categorical entropy in nats for [..., classes] logits.

    Last axis is classes; all preceding axes are averaged. Softmax defines the
    distribution. No temperature or division by log(class count) is applied.
    One class has zero entropy; extreme finite logits permit zero probabilities.
    """
    x = _tensor(logits)
    if x.ndim < 1:
        raise ValueError("logits must have shape [..., classes]")
    shifted = x - x.amax(dim=-1, keepdim=True)
    probabilities = shifted.exp()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1).mean()
    return max(0.0, _finite(float(entropy)))
