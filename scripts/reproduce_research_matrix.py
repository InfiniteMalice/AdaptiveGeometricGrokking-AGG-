"""Execute all small integration components; never authorize final-test inference.

Run from the installed repository: python scripts/reproduce_research_matrix.py --output runs/matrix
"""

import argparse
import json
from pathlib import Path

from agg.experiments.blocks import BlockConfig, audit_blocks, run_blocks
from agg.experiments.config import ExperimentConfig
from agg.experiments.independent import report_run, selection_comparison
from agg.experiments.integration import (
    IntegrationConfig,
    build_bundle,
    prepare_release,
    run_integration,
)
from agg.experiments.multistep import MultistepConfig, run_multistep
from agg.experiments.resources import ResourceConfig, audit_resources, run_resources
from agg.experiments.runner import run_experiment
from agg.experiments.scaling import ScalingConfig, audit_scaling, run_scaling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = Path(__file__).resolve().parents[1] / "configs"

    def read(name):
        return json.loads((configs / name).read_text())

    integrated = args.output / "integrated"
    result = run_integration(
        IntegrationConfig.from_dict(read("integration-smoke.json")), integrated
    )
    selection = args.output / "selection"
    for seed in (11, 13):
        cfg = ExperimentConfig.from_dict(read("independent-smoke.json"))
        cfg.training.seed = seed
        for arm, variant in selection_comparison(cfg).items():
            root = selection / f"{arm}-{seed}"
            run_experiment(variant, root)
            report_run(root, bootstrap_seed=41)
    blocks = args.output / "blocks"
    run_blocks(BlockConfig.from_dict(read("block-smoke.json")), blocks)
    audit_blocks(blocks, seed=41)
    resources = args.output / "resources"
    run_resources(ResourceConfig.from_dict(read("resource-smoke.json")), resources)
    audit_resources(resources, seed=41)
    scaling = args.output / "scaling"
    run_scaling(ScalingConfig.from_dict(read("scaling-smoke.json")), scaling)
    audit_scaling(scaling, seed=41)
    credit = args.output / "credit"
    run_multistep(MultistepConfig.from_dict(read("multistep-smoke.json")), credit)
    bundle = args.output / "evidence"
    build_bundle(
        {
            "integrated": integrated,
            "selection": selection,
            "blocks": blocks,
            "resources": resources,
            "scaling": scaling,
            "credit": credit,
        },
        bundle,
    )
    roots = [bundle / "sources" / "integrated" / row["id"] for row in result["cells"]]
    if any(row["status"] != "complete" for row in result["cells"]):
        raise RuntimeError("Incomplete integration study; evidence retained, no release prepared")
    prepare_release(roots, bundle / "final-release.json")
    print(f"Evidence and unauthorised final-release manifest: {bundle}")


if __name__ == "__main__":
    main()
