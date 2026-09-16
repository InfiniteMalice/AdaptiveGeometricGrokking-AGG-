"""Versioned, weights-only inference artifacts with explicit model architectures.

Only the named AGG model types are reconstructed. Loading restores CPU tensors,
their dtypes and train/eval modes; it does not resume an optimizer or training RNG.
"""

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from agg.compression import PRECISIONS
from agg.geometry import AdaptedModel, GeometryAdapter
from agg.models import TinyTransformer


def _architecture(model: nn.Module) -> dict[str, Any]:
    from .candidates import ActivationPrecisionModel

    if type(model) is TinyTransformer:
        return {
            "type": "TinyTransformer",
            "kwargs": {
                "vocab_size": model.embedding.num_embeddings,
                "classes": model.readout.out_features,
                "width": model.width,
                "layers": len(model.blocks),
                "heads": cast(nn.TransformerEncoderLayer, model.blocks[0]).self_attn.num_heads,
                "max_length": model.position.num_embeddings,
            },
        }
    if type(model) is AdaptedModel:
        adapter = model.adapter
        if type(adapter) is not GeometryAdapter or adapter.gate._forced is not None:
            raise ValueError("Unsupported adapter or active transient gate intervention")
        curved = adapter.geometry != "euclidean"
        return {
            "type": "AdaptedModel",
            "base": _architecture(model.base),
            "adapter": {
                "width": adapter.width,
                "dimension": adapter.dimension,
                "geometry": adapter.geometry,
                # Learned gate weights are restored from state, including saturation.
                "gate": 0.5 if adapter.gate.logit is not None else float(adapter.gate.fixed),
                "learned_gate": adapter.gate.logit is not None,
                "curvature": float(adapter.manifold.c.detach()) if curved else 1.0,
                "learned_curvature": (
                    adapter.manifold.isp_c.requires_grad
                    if curved
                    else adapter.raw_scale is not None
                ),
            },
        }
    if type(model) is ActivationPrecisionModel:
        if model.precision not in PRECISIONS:
            raise ValueError("Unsupported activation precision")
        return {
            "type": "ActivationPrecisionModel",
            "base": _architecture(model.base),
            "precision": model.precision,
        }
    raise ValueError(f"Unsupported checkpoint model type: {type(model).__name__}")


def _construct(architecture: Any, depth: int = 0) -> nn.Module:
    from .candidates import ActivationPrecisionModel

    if not isinstance(architecture, dict) or depth > 16:
        raise ValueError("Invalid or excessively nested architecture")
    kind = architecture.get("type")
    if kind == "TinyTransformer" and set(architecture) == {"type", "kwargs"}:
        return TinyTransformer(**architecture["kwargs"])
    if kind == "AdaptedModel" and set(architecture) == {"type", "base", "adapter"}:
        base = _construct(architecture["base"], depth + 1)
        if not isinstance(base, TinyTransformer):
            raise ValueError("AdaptedModel requires a TinyTransformer base")
        return AdaptedModel(base, GeometryAdapter(**architecture["adapter"]))
    if kind == "ActivationPrecisionModel" and set(architecture) == {"type", "base", "precision"}:
        if architecture["precision"] not in PRECISIONS:
            raise ValueError("Invalid activation precision")
        return ActivationPrecisionModel(
            _construct(architecture["base"], depth + 1), architecture["precision"]
        )
    raise ValueError(f"Unsupported checkpoint architecture: {kind}")


def save_model(model: nn.Module, path: str | Path) -> None:
    """Save architecture and CPU state without pickling model or Python classes."""
    architecture = _architecture(model)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Cannot save nonfinite model state")
    torch.save(
        {
            "format_version": 1,
            "architecture": architecture,
            "state_dict": state,
            "training_modes": {name: module.training for name, module in model.named_modules()},
            "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
        },
        Path(path),
    )


def load_model(path: str | Path) -> nn.Module:
    """Reconstruct a supported CPU model, fail closed on incompatible artifacts.

    This is an inference artifact reader, not a sandbox for untrusted tensor files.
    Optional geometry dependencies retain their actionable ImportError.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "architecture", "state_dict", "training_modes", "requires_grad"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or type(payload["format_version"]) is not int
        or payload["format_version"] != 1
    ):
        raise ValueError("Unsupported checkpoint format or version")
    try:
        state = payload["state_dict"]
        if not isinstance(state, dict) or any(
            not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError("Invalid or nonfinite checkpoint state")
        with torch.random.fork_rng(devices=[]):
            model = _construct(payload["architecture"])
        # assign preserves saved floating dtypes, rather than silently casting to FP32.
        model.load_state_dict(state, strict=True, assign=True)
        modules = dict(model.named_modules())
        parameters = dict(model.named_parameters())
        for field, targets in (("training_modes", modules), ("requires_grad", parameters)):
            values = payload[field]
            if (
                not isinstance(values, dict)
                or set(values) != set(targets)
                or any(type(value) is not bool for value in values.values())
            ):
                raise ValueError(f"Invalid checkpoint {field}")
        for name, module in modules.items():
            module.training = payload["training_modes"][name]
        for name, parameter in parameters.items():
            parameter.requires_grad_(payload["requires_grad"][name])
        return model
    except (TypeError, KeyError, RuntimeError, AttributeError) as exc:
        raise ValueError(f"Invalid checkpoint architecture or state: {exc}") from exc
