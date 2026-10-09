"""Explicit candidate construction and equal-budget fitting."""

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from agg.compression import fake_quantize, magnitude_prune
from agg.consolidation import Proposal
from agg.distillation import distillation_loss
from agg.geometry import AdaptedModel, GeometryAdapter
from agg.models import TinyTransformer
from agg.tasks import TaskData

from .config import ExperimentConfig


def fit_candidate(
    student: nn.Module,
    teacher: nn.Module,
    data: TaskData,
    config: ExperimentConfig,
    distill: bool,
    optimizer_observer: Callable[[dict[str, Any]], None] | None = None,
) -> nn.Module:
    modes = [(module, module.training) for module in teacher.modules()]
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.training.seed + 77)
    try:
        teacher.eval()
        student.train()
        for _ in range(config.candidate_steps):
            indices = torch.randperm(len(data.train.y), generator=generator)[
                : config.training.batch_size
            ]
            x, y = data.train.x[indices], data.train.y[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = student(x)
            if distill:
                with torch.no_grad():
                    target = teacher(x)
                loss = distillation_loss(logits, y, target, teacher_weight=0.5)
            else:
                loss = F.cross_entropy(logits, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite candidate fitting loss")
            loss.backward()
            optimizer.step()
    finally:
        for module, mode in modes:
            module.training = mode
        if optimizer_observer is not None:
            optimizer_observer(
                {
                    "schema_version": "agg.candidate-optimizer/1",
                    "optimizer": optimizer.state_dict(),
                    "model": student.state_dict(),
                    "batch_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "optimizer_kind": "AdamW",
                    "resumed_parent_optimizer": False,
                }
            )
    return student


class ActivationPrecisionModel(nn.Module):
    """Inference-only fake rounding before the classifier; compute remains FP32."""

    def __init__(self, base: nn.Module, precision: str) -> None:
        super().__init__()
        self.base, self.precision = base, precision

    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False) -> Any:
        _, hidden = self.base(tokens, return_hidden=True)
        rounded = fake_quantize(hidden[-1], self.precision)
        root = self.base
        while isinstance(root, (AdaptedModel, ActivationPrecisionModel)):
            root = root.base
        if not isinstance(root, TinyTransformer):
            raise TypeError("activation rounding requires TinyTransformer readout")
        logits = root.readout(root.norm(rounded[:, -1]))
        return (logits, [*hidden[:-1], rounded]) if return_hidden else logits


def proposals(
    model: nn.Module,
    data: TaskData,
    config: ExperimentConfig,
    *,
    optimizer_observer: Callable[[dict[str, Any]], None] | None = None,
) -> list[Proposal]:
    result = []
    f = config.features
    if f.distillation:
        teacher = copy.deepcopy(model)
        result.append(
            Proposal(
                "self_distillation",
                "model",
                {"steps": config.candidate_steps},
                lambda m: fit_candidate(m, teacher, data, config, True, optimizer_observer),
            )
        )
    if f.geometry or f.dimension or f.gating:
        geometries = config.geometries if f.geometry else ("euclidean",)
        dimensions = config.dimensions if f.dimension else (config.training.width,)
        for geometry in geometries:
            for dimension in dimensions:
                if geometry == "product" and dimension < 2:
                    continue

                def adapt(
                    m: nn.Module, geometry: str = geometry, dimension: int = dimension
                ) -> nn.Module:
                    adapter = GeometryAdapter(
                        config.training.width,
                        dimension,
                        geometry,
                        config.gate if f.gating else 1.0,
                        config.learned_gate if f.gating else False,
                        config.curvature,
                        config.learned_curvature,
                    )
                    # Wrapping the root avoids nesting adapters during an independent sweep.
                    if isinstance(m, AdaptedModel):
                        m = m.base
                    if not isinstance(m, TinyTransformer):
                        raise TypeError("geometry candidate requires TinyTransformer")
                    student = AdaptedModel(m, adapter)
                    return fit_candidate(
                        student, model, data, config, f.distillation, optimizer_observer
                    )

                result.append(
                    Proposal(
                        "geometry",
                        "final_hidden",
                        {
                            "geometry": geometry,
                            "dimension": dimension,
                            "gate": config.gate if f.gating else 1.0,
                            "curvature": config.curvature,
                            "steps": config.candidate_steps,
                        },
                        adapt,
                    )
                )
    if f.pruning:

        def prune(m: nn.Module) -> nn.Module:
            with torch.no_grad():
                for p in m.parameters():
                    if p.ndim >= 2:
                        p.copy_(magnitude_prune(p, config.prune_fraction))
            return m

        result.append(
            Proposal("pruning", "matrix_weights", {"fraction": config.prune_fraction}, prune)
        )
    if f.quantization:
        # Each tensor gets its own trial; a sensitive component can remain FP32.
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2:
                continue
            for precision in config.precisions:

                def quantize(
                    m: nn.Module, name: str = name, precision: str = precision
                ) -> nn.Module:
                    with torch.no_grad():
                        p = dict(m.named_parameters())[name]
                        p.copy_(fake_quantize(p, precision))
                    return m

                result.append(
                    Proposal(
                        "precision",
                        name,
                        {
                            "weight": precision,
                            "activation": "FP32",
                            "compute": "FP32",
                            "accumulator": "FP32",
                            "simulation": True,
                        },
                        quantize,
                    )
                )
        for precision in config.activation_precisions:

            def round_activations(m: nn.Module, p: str = precision) -> nn.Module:
                return ActivationPrecisionModel(m, p)

            result.append(
                Proposal(
                    "activation_precision",
                    "final_hidden",
                    {
                        "activation": precision,
                        "weight": "FP32",
                        "compute": "FP32",
                        "accumulator": "FP32",
                        "simulation": True,
                    },
                    round_activations,
                )
            )
    return result
