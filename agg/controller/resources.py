"""Conjunctive resource requirements and explicitly scoped CPU measurements."""

import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ResourceProfile:
    min_id: float | None = None
    min_ood: float | None = None
    max_latency_ms: float | None = None
    max_model_bytes: float | None = None
    max_peak_memory_bytes: float | None = None
    max_inference_cpu_ms: float | None = None
    max_parameters: float | None = None
    max_training_updates: float | None = None
    max_training_tokens: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name.startswith("min_") and value > 1)
            ):
                raise ValueError(f"invalid resource requirement: {name}")

    def assess(self, metrics: dict[str, float]) -> dict[str, Any]:
        names = {
            "min_id": "performance.validation_score",
            "min_ood": "performance.ood_score",
            "max_latency_ms": "resources.latency_ms",
            "max_model_bytes": "resources.model_bytes",
            "max_peak_memory_bytes": "resources.peak_memory_bytes",
            "max_inference_cpu_ms": "resources.inference_cpu_ms",
            "max_parameters": "resources.parameter_count",
            "max_training_updates": "resources.training_updates",
            "max_training_tokens": "resources.training_tokens",
        }
        requirements = {}
        for name, limit in asdict(self).items():
            if limit is None:
                continue
            value = metrics.get(names[name])
            available = (
                value is not None
                and type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
            )
            if name.startswith("min_") and available and value is not None:
                available = value <= 1
            margin = (
                ((value - limit) if name.startswith("min_") else (limit - value))
                if available
                else None
            )
            requirements[name] = {
                "metric": names[name],
                "limit": limit,
                "value": value if available else None,
                "margin": margin,
                "passed": margin is not None and margin >= 0,
                "missing_reason": None
                if available
                else "required measurement unavailable or invalid",
            }
        return {
            "requirements": requirements,
            "joint_feasible": all(r["passed"] for r in requirements.values()),
        }


def measure_resources(
    model: nn.Module, inputs: torch.Tensor, *, repeats: int = 9, warmup: int = 2
) -> dict[str, Any]:
    """Measure resident tensor storage and synchronous CPU forward batch execution.

    No FLOPs/energy/peak-RSS estimate is substituted for missing instrumentation.
    Process CPU time includes other process threads; small values may quantize to zero.
    """
    if type(repeats) is not int or repeats < 3 or type(warmup) is not int or warmup < 0:
        raise ValueError("at least three timed repeats and nonnegative warmup required")
    tensors = list(model.parameters()) + list(model.buffers())
    if inputs.device.type != "cpu" or any(t.device.type != "cpu" for t in tensors):
        raise ValueError("resource probe supports synchronous CPU execution only")
    if inputs.ndim != 2 or inputs.shape[0] < 1:
        raise ValueError("nonempty batch of token sequences required")
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    modes = [(m, m.training) for m in model.modules()]
    wall, cpu = [], []
    try:
        model.eval()
        with torch.inference_mode():
            for i in range(warmup + repeats):
                start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
                logits = model(inputs)
                elapsed_cpu = (time.process_time_ns() - start_cpu) / 1e6
                elapsed_wall = (time.perf_counter_ns() - start_wall) / 1e6
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("nonfinite resource probe output")
                if i >= warmup:
                    wall.append(elapsed_wall)
                    cpu.append(elapsed_cpu)
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "schema_version": "agg.resource-measurement/1",
        "metrics": {
            "resources.latency_ms": statistics.median(wall),
            "resources.inference_cpu_ms": statistics.median(cpu),
            "resources.model_bytes": sum(storages.values()),
            "resources.parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "latency_samples_ms": wall,
        "cpu_samples_ms": cpu,
        "batch_shape": list(inputs.shape),
        "warmup": warmup,
        "repeats": repeats,
        "hardware": {
            "device": "cpu",
            "machine": platform.machine(),
            "system": platform.system(),
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "scope": "median synchronous forward batch; unique parameter and buffer storage only",
        "unavailable": {
            "resources.peak_memory_bytes": "peak runtime allocation not instrumented",
            "flops": "operator FLOPs not instrumented",
            "energy": "not instrumented",
        },
    }
