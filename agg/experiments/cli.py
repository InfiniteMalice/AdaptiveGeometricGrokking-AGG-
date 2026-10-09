import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .analysis import phase_surface
from .config import ExperimentConfig
from .matrix import ablation_matrix, retrieval_factorial
from .runner import run_experiment, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Geometric Grokking research harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one declarative experiment")
    run.add_argument("--config", type=Path)
    run.add_argument("--task", choices=["modular", "hierarchy", "retrieval"])
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--steps", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--ablation", choices=list(ablation_matrix()))
    matrix = sub.add_parser("matrix", help="emit or execute a retrieval factorial")
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--execute", action="store_true")
    matrix.add_argument("--steps", type=int, default=100)
    independent = sub.add_parser(
        "independent", help="report on frozen audit or authorized final data"
    )
    independent.add_argument("--run", type=Path, required=True)
    independent.add_argument("--role", choices=["audit", "final"], default="audit")
    independent.add_argument("--authorization")
    independent.add_argument("--manifest-sha256")
    independent.add_argument("--bootstrap-seed", type=int, default=0)
    causal = sub.add_parser("causal", help="report oracle-validated frozen audit interventions")
    causal.add_argument("--run", type=Path, required=True)
    causal.add_argument("--seed", type=int, default=0)
    milestones = sub.add_parser("milestones", help="diagnose milestones from frozen audit reports")
    milestones.add_argument("--run", type=Path, required=True)
    blocks = sub.add_parser("blocks", help="run guarded executable-summary curriculum")
    blocks.add_argument("--config", type=Path)
    blocks.add_argument("--output", type=Path, required=True)
    block_audit = sub.add_parser("block-audit", help="audit a frozen block curriculum")
    block_audit.add_argument("--run", type=Path, required=True)
    block_audit.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command in {"blocks", "block-audit"}:
        from .blocks import BlockConfig, audit_blocks, run_blocks

        if args.command == "blocks":
            block_config = (
                BlockConfig.from_dict(json.loads(args.config.read_text()))
                if args.config
                else BlockConfig()
            )
            result = run_blocks(block_config, args.output)
        else:
            result = audit_blocks(args.run, seed=args.seed)
        print(json.dumps(result, indent=2))
        return
    if args.command == "milestones":
        from .milestones import milestone_report

        print(json.dumps(milestone_report(args.run), indent=2))
        return
    if args.command == "causal":
        from .causal import causal_report

        print(json.dumps(causal_report(args.run, seed=args.seed), indent=2))
        return
    if args.command == "independent":
        from .independent import report_run

        report = report_run(
            args.run,
            role=args.role,
            authorization=args.authorization,
            manifest_sha256=args.manifest_sha256,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "matrix":
        configs = retrieval_factorial(steps=args.steps)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "matrix.json", [asdict(c) for c in configs])
        if args.execute:
            rows = [run_experiment(c, args.output / f"cell-{i:04d}") for i, c in enumerate(configs)]
            write_json(args.output / "results.json", rows)
            cells = [
                {
                    "gate": c.gate,
                    "distance": c.distance,
                    "density": c.density,
                    "context_length": c.context_length,
                    "seed": c.training.seed,
                    "crossing": row["crossing"],
                }
                for c, row in zip(configs, rows, strict=True)
            ]
            write_json(args.output / "phase-surface.json", phase_surface(cells))
        print(f"Wrote {len(configs)} factorial cells to {args.output}")
        return
    config = (
        ExperimentConfig.from_dict(json.loads(args.config.read_text()))
        if args.config
        else (ExperimentConfig())
    )
    if args.task:
        config.task = args.task
    if args.steps is not None:
        config.training = replace(config.training, steps=args.steps)
    if args.seed is not None:
        config.training = replace(config.training, seed=args.seed)
    if args.ablation:
        config.features = ablation_matrix()[args.ablation]
        if not (config.features.geometry or config.features.dimension or config.features.gating):
            config.adapt_during_training = False
        if args.ablation == "immediate_credit":
            config.gamma = 0.0
    config.__post_init__()
    print(json.dumps(run_experiment(config, args.output), indent=2))


if __name__ == "__main__":
    main()
