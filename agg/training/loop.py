import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agg.models import TinyTransformer
from agg.tasks import Split, TaskData


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    eval_every: int = 20
    width: int = 32
    layers: int = 2
    heads: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    seed: int = 0
    batch_size: int = 128

    def __post_init__(self) -> None:
        if (
            min(self.steps, self.eval_every, self.width, self.layers, self.heads, self.batch_size)
            < 1
        ):
            raise ValueError("training counts and dimensions must be positive")
        if self.width % self.heads or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid head width or optimizer parameters")


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, Any]]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(model: nn.Module, split: Split) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        logits = model(split.x)
        predicted = logits.argmax(-1)
        recalls = [(predicted[split.y == c] == c).float().mean() for c in split.y.unique()]
        return {
            "loss": float(F.cross_entropy(logits, split.y)),
            "accuracy": float((predicted == split.y).float().mean()),
            "balanced_accuracy": float(torch.stack(recalls).mean()),
        }
    finally:
        model.train(was_training)


def train(
    data: TaskData,
    config: TrainConfig,
    output: Path,
    callback: Callable[[int, nn.Module, dict[str, Any]], None] | None = None,
    model: nn.Module | None = None,
) -> TrainingResult:
    seed_all(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if model is None:
        model = TinyTransformer(
            data.vocab_size,
            data.classes,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            max_length=data.train.x.shape[1],
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed + 1)
    history = []

    def record(step: int) -> None:
        metrics: dict[str, Any] = {"step": step}
        for name, split in (("train", data.train), ("id", data.id), ("ood", data.ood)):
            metrics[name] = evaluate(model, split)
        history.append(metrics)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "batch_rng": generator.get_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "config": asdict(config),
                "task": data.metadata,
            },
            output / f"checkpoint-{step}.pt",
        )
        if callback:
            callback(step, model, metrics)

    (output / "config.json").write_text(
        json.dumps(
            {
                "training": asdict(config),
                "task": data.metadata,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    record(0)
    for step in range(1, config.steps + 1):
        model.train()
        indices = torch.randperm(len(data.train.y), generator=generator)[: config.batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data.train.x[indices]), data.train.y[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % config.eval_every == 0 or step == config.steps:
            record(step)
    return TrainingResult(model, history)
