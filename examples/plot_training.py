"""Plot a saved trajectory without smoothing or claiming phase transitions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for split, color in (("train", "#376A8A"), ("id", "#BC641E"), ("ood", "#855299")):
        axes[0].plot(steps, [r[split]["accuracy"] for r in rows], label=split.upper(), color=color)
        axes[1].plot(steps, [r[split]["loss"] for r in rows], label=split.upper(), color=color)
    axes[0].set(ylabel="Accuracy", ylim=(0, 1.04))
    axes[1].set(ylabel="Cross entropy", yscale="log")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(
        "AGG modular addition: recorded single-seed trajectory\n"
        "No smoothing; not a replicated grokking result",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
