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
    empowerment = sub.add_parser("empowerment", help="calculate finite-MDP empowerment references")
    empowerment.add_argument("--config", type=Path)
    empowerment.add_argument("--output", type=Path, required=True)
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
    resources = sub.add_parser("resources", help="run measured resource controller comparison")
    resources.add_argument("--config", type=Path)
    resources.add_argument("--output", type=Path, required=True)
    resource_audit = sub.add_parser("resource-audit", help="audit frozen resource comparison")
    resource_audit.add_argument("--run", type=Path, required=True)
    resource_audit.add_argument("--seed", type=int, default=0)
    scaling = sub.add_parser("scaling", help="train controlled small/larger geometry grid")
    scaling.add_argument("--config", type=Path)
    scaling.add_argument("--output", type=Path, required=True)
    scaling_audit = sub.add_parser(
        "scaling-audit", help="audit frozen grid and fit small-only scaling forms"
    )
    scaling_audit.add_argument("--run", type=Path, required=True)
    scaling_audit.add_argument("--seed", type=int, default=0)
    multistep = sub.add_parser("multistep", help="run executed finite-MDP credit comparison")
    multistep.add_argument("--config", type=Path)
    multistep.add_argument("--output", type=Path, required=True)
    integration = sub.add_parser("integration", help="run fixed matched-control integration study")
    integration.add_argument("--config", type=Path)
    integration.add_argument("--output", type=Path, required=True)
    bundle = sub.add_parser("evidence-bundle", help="copy and hash explicitly named source runs")
    bundle.add_argument("--sources", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser(
        "prepare-release", help="freeze final evaluation scope without inference"
    )
    prepare.add_argument("--run", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("release-final", help="evaluate an explicitly authorized frozen release")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--authorization", required=True)
    final.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "empowerment":
        from .empowerment import EmpowermentConfig, run_empowerment

        empowerment_config = (
            EmpowermentConfig.from_dict(json.loads(args.config.read_text()))
            if args.config else EmpowermentConfig()
        )
        print(json.dumps(run_empowerment(empowerment_config, args.output), indent=2))
        return
    if args.command in {"integration", "evidence-bundle", "prepare-release", "release-final"}:
        from .integration import (
            IntegrationConfig,
            build_bundle,
            prepare_release,
            release_final,
            run_integration,
        )

        if args.command == "integration":
            cfg_integrated = (
                IntegrationConfig.from_dict(json.loads(args.config.read_text()))
                if args.config
                else IntegrationConfig()
            )
            result_integrated = run_integration(cfg_integrated, args.output)
        elif args.command == "evidence-bundle":
            sources = {
                key: (args.sources.parent / value).resolve()
                for key, value in json.loads(args.sources.read_text()).items()
            }
            result_integrated = build_bundle(sources, args.output)
        elif args.command == "prepare-release":
            result_integrated = prepare_release(args.run, args.output)
        else:
            result_integrated = release_final(
                args.manifest, authorization=args.authorization, digest=args.sha256
            )
        print(json.dumps(result_integrated, indent=2))
        return
    if args.command == "multistep":
        from .multistep import MultistepConfig, run_multistep

        config_credit = (
            MultistepConfig.from_dict(json.loads(args.config.read_text()))
            if args.config
            else MultistepConfig()
        )
        print(json.dumps(run_multistep(config_credit, args.output), indent=2))
        return
    if args.command in {"scaling", "scaling-audit"}:
        from .scaling import ScalingConfig, audit_scaling, run_scaling

        if args.command == "scaling":
            scaling_config = (
                ScalingConfig.from_dict(json.loads(args.config.read_text()))
                if args.config
                else ScalingConfig()
            )
            result = run_scaling(scaling_config, args.output)
        else:
            result = audit_scaling(args.run, seed=args.seed)
        print(json.dumps(result, indent=2))
        return
    if args.command in {"resources", "resource-audit"}:
        from .resources import ResourceConfig, audit_resources, run_resources

        if args.command == "resources":
            resource_config = (
                ResourceConfig.from_dict(json.loads(args.config.read_text()))
                if args.config
                else ResourceConfig()
            )
            result = run_resources(resource_config, args.output)
        else:
            result = audit_resources(args.run, seed=args.seed)
        print(json.dumps(result, indent=2))
        return
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
