from dataclasses import replace
from itertools import product

from agg.training import TrainConfig

from .config import ExperimentConfig, Features


def ablation_matrix() -> dict[str, Features]:
    matrix = {"baseline": Features.baseline()}
    for name, flag in (
        ("telemetry_only", "telemetry"),
        ("self_distillation_only", "distillation"),
        ("geometry_only", "geometry"),
        ("dimension_only", "dimension"),
        ("gating_only", "gating"),
        ("epiplexity_only", "epiplexity"),
        ("pruning_only", "pruning"),
        ("quantization_only", "quantization"),
        ("storage_only", "storage"),
        ("executable_only", "executable"),
        ("teacher_only", "teacher"),
        ("process_only", "process"),
        ("verified_only", "verified"),
    ):
        flags = Features.baseline()
        setattr(flags, flag, True)
        flags.consolidation = flag in {
            "distillation",
            "geometry",
            "dimension",
            "gating",
            "pruning",
            "quantization",
        }
        matrix[name] = flags
    phase_a = replace(
        Features.baseline(),
        telemetry=True,
        consolidation=True,
        distillation=True,
        epiplexity=True,
        executable=True,
        verified=True,
        teacher=True,
        process=True,
    )
    phase_b = replace(phase_a, geometry=True, dimension=True, gating=True)
    matrix.update(
        full_a=phase_a,
        a_b=phase_b,
        a_c=replace(phase_a, pruning=True, quantization=True, storage=True),
        a_b_c=Features(),
        immediate_credit=replace(Features.baseline(), executable=True, verified=True),
        discounted_credit=replace(Features.baseline(), executable=True, verified=True),
    )
    return matrix


def retrieval_factorial(
    *,
    gates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    distances: tuple[int, ...] = (1, 4),
    densities: tuple[float, ...] = (0.0, 0.5),
    lengths: tuple[int, ...] = (8, 16),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 100,
) -> list[ExperimentConfig]:
    if not lengths or not distances or any(d >= min(lengths) - 1 or d < 1 for d in distances):
        raise ValueError(
            "every distance must leave room for a longer OOD condition in every context"
        )
    flags = replace(Features.baseline(), telemetry=True, gating=True, dimension=True)
    return [
        ExperimentConfig(
            task="retrieval",
            gate=g,
            distance=d,
            density=rho,
            context_length=n,
            training=TrainConfig(steps=steps, seed=seed),
            features=replace(flags),
            dimensions=(8,),
            geometries=("euclidean",),
            adapt_during_training=True,
        )
        for g, d, rho, n, seed in product(gates, distances, densities, lengths, seeds)
    ]
