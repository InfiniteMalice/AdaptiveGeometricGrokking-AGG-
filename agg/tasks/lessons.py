"""Executable summary templates grounded in verified episodes, not prose rewards."""

import hashlib
from typing import Any

import torch

from agg.controller.memory import Abstraction

from .causal import intervention_pairs, oracle
from .synthetic import Split

OPERATIONS = {
    "retrieval": ("irrelevant_permutation", "value_cycle"),
    "modular": ("operand_swap",),
    "hierarchy": ("ancestry_preserving_query",),
}
CONTROLS = ("incorrect", "irrelevant", "contradictory", "obsolete", "random", "out_of_scope")


def lesson_examples(
    split: Split, metadata: dict[str, Any], operation: str, *, seed: int, control: str | None = None
) -> tuple[Split, dict[str, Any]]:
    if control not in (None, *CONTROLS):
        raise ValueError("unknown lesson control")
    if operation not in OPERATIONS.get(metadata["task"], ()) or control == "out_of_scope":
        raise ValueError("lesson operation is outside task scope")
    pairs = intervention_pairs(
        split,
        tuple("source" for _ in split.y),
        metadata,
        kind="decisive" if operation == "value_cycle" else "invariant",
        seed=seed,
    )
    x, y = split.x.clone(), split.y.clone()
    # Replacement preserves exact row and token budgets, rather than appending data.
    indices = pairs.source_indices[::2]
    if indices and control != "irrelevant":
        x[indices], y[indices] = pairs.transformed.x[::2], pairs.transformed.y[::2]
    classes = (
        int(metadata["modulus"])
        if metadata["task"] == "modular"
        else int(metadata["values"])
        if metadata["task"] == "retrieval"
        else 2
    )
    if control == "incorrect":
        y = (y + 1) % classes
    elif control == "contradictory":
        y[::2] = (y[::2] + 1) % classes
    elif control == "obsolete":
        # Deliberately stale bindings: pair current examples with previous labels.
        y = y.roll(1)
    elif control == "random":
        y = torch.randint(classes, y.shape, generator=torch.Generator().manual_seed(seed))
    agreement = float((oracle(x, metadata) == y).float().mean())
    return Split(x, y), {
        "schema_version": "agg.lesson-examples/1",
        "operation": operation,
        "control": control,
        "oracle_agreement": agreement,
        "rows": len(y),
        "changed_inputs": int((x != split.x).any(1).sum()),
        "changed_labels": int((y != split.y).sum()),
        "tokens": x.numel(),
        "nontrivial_support": len(pairs.source_indices),
        "missing_transforms": pairs.missing,
    }


def extract_lessons(
    split: Split,
    metadata: dict[str, Any],
    *,
    source_block: int,
    episode_prefix: str,
    predictions: list[int],
    seed: int,
) -> list[Abstraction]:
    if source_block < 0 or not episode_prefix or len(predictions) != len(split.y):
        raise ValueError("lesson extraction requires actual aligned episode results")
    if not torch.equal(oracle(split.x, metadata), split.y):
        raise ValueError("source episode labels fail oracle validation")
    result = []
    fingerprint = hashlib.sha256(split.x.numpy().tobytes() + split.y.numpy().tobytes()).hexdigest()
    for operation in OPERATIONS[metadata["task"]]:
        _, details = lesson_examples(split, metadata, operation, seed=seed)
        if details["nontrivial_support"] == 0:
            continue
        result.append(
            Abstraction(
                id=f"{episode_prefix}:{operation}",
                proposition=f"Oracle-supported {operation}",
                block=metadata["task"],
                conditions={"task": metadata["task"]},
                status="experimental",
                source_episodes=tuple(f"{episode_prefix}:episode:{i}" for i in range(len(split.y))),
                provenance={
                    "schema_version": "agg.executable-summary/1",
                    "source_block": source_block,
                    "operation": operation,
                    "source_sha256": fingerprint,
                    "source_accuracy": float((torch.tensor(predictions) == split.y).float().mean()),
                    "oracle_support": details,
                    "signature": f"{metadata['task']}:{operation}",
                    "generation": "enumerated task templates; not extracted internal understanding",
                },
            )
        )
    return result
