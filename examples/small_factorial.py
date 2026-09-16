"""Run 32 short retrieval cells; this verifies orchestration, not a phase transition."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from agg.experiments import run_experiment
from agg.experiments.analysis import phase_surface
from agg.experiments.matrix import retrieval_factorial
from agg.experiments.runner import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = retrieval_factorial(
        gates=(0.0, 1.0),
        distances=(1, 3),
        densities=(0.0, 0.5),
        lengths=(8, 16),
        seeds=(0, 1),
        steps=12,
    )
    rows = []
    for index, config in enumerate(configs):
        config.training = replace(config.training, eval_every=4)
        config.samples = 64
        summary = run_experiment(config, args.output / f"cell-{index:03d}")
        rows.append(
            {
                "gate": config.gate,
                "distance": config.distance,
                "density": config.density,
                "context_length": config.context_length,
                "seed": config.training.seed,
                "crossing": summary["crossing"],
            }
        )
    write_json(args.output / "matrix.json", [asdict(c) for c in configs])
    write_json(args.output / "phase-surface.json", phase_surface(rows))
    print(f"Completed {len(configs)} cells; summary: {args.output / 'phase-surface.json'}")


if __name__ == "__main__":
    main()
