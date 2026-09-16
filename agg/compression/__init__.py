"""Proposal-only structural compression and logical precision simulation.

All tensors remain ordinary dense tensors. Fake quantization makes no storage or
accelerated compute claims; compute and accumulator fields are declared targets.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

PRECISIONS = ("FP32", "FP16", "BF16", "INT8", "INT4", "ternary")


@dataclass(frozen=True)
class PrecisionConfig:
    weight: str = "FP32"
    activation: str = "FP32"
    compute: str = "FP32"
    accumulator: str = "FP32"

    def __post_init__(self) -> None:
        if any(
            p not in PRECISIONS
            for p in (self.weight, self.activation, self.compute, self.accumulator)
        ):
            raise ValueError("Unknown logical precision")


def _validate(tensor: Tensor) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Expected finite floating point tensor")


def fake_quantize(tensor: Tensor, precision: str, ternary_threshold: float = 0.5) -> Tensor:
    """Symmetric per-tensor rounding; ternary values use the max absolute scale."""
    _validate(tensor)
    if precision not in PRECISIONS or not 0 <= ternary_threshold <= 1:
        raise ValueError("Unknown precision or invalid threshold")
    if tensor.numel() == 0:
        return tensor.clone()
    if precision in ("FP32", "FP16", "BF16"):
        dtype = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}
        result = tensor.to(dtype[precision]).to(tensor.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Precision overflow")
        return result.clone()
    # Normalize before multiplying by the integer range. Computing scale / range
    # first can underflow even in float64 for its smallest subnormal values.
    # Wider intermediates also avoid premature rounding for FP16/BF16 inputs.
    work = tensor.to(torch.float64)
    scale = work.abs().max()
    if scale.detach().item() == 0:
        return tensor.clone()
    normalized = work / scale
    if precision == "ternary":
        result = torch.where(
            normalized.abs() > ternary_threshold, work.sign() * scale, torch.zeros_like(work)
        )
    else:
        maximum = 127 if precision == "INT8" else 7
        levels = torch.clamp(torch.round(normalized * maximum), -maximum, maximum)
        result = (levels / maximum) * scale
    result = result.to(tensor.dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Precision overflow")
    return result


def magnitude_prune(tensor: Tensor, fraction: float) -> Tensor:
    """Zero exactly floor(fraction * count) smallest entries, stable on ties."""
    _validate(tensor)
    if not 0 <= fraction <= 1:
        raise ValueError("Pruning fraction must lie in [0,1]")
    result = tensor.clone().reshape(-1)
    indices = torch.argsort(result.abs(), stable=True)[: int(fraction * result.numel())]
    result[indices] = 0
    return result.reshape(tensor.shape)


def low_rank(tensor: Tensor, rank: int) -> Tensor:
    _validate(tensor)
    if tensor.ndim != 2 or rank < 0 or rank > min(tensor.shape):
        raise ValueError("Rank must fit a matrix")
    u, s, vh = torch.linalg.svd(tensor.float(), full_matrices=False)
    return ((u[:, :rank] * s[:rank]) @ vh[:rank]).to(tensor.dtype)
