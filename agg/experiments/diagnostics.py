"""Executed observational probes. None of these statistics is optimized by AGG."""

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from agg.complexity import PrequentialCodeLengthProxy
from agg.depth import trajectory_statistics
from agg.dimension import observation_matrix, spectral_statistics
from agg.gating import ScalarGate
from agg.horizon import effective_horizons
from agg.tasks import TaskData
from agg.topology import connectivity, geometry_statistics


@torch.no_grad()
def observe(
    model: nn.Module,
    data: TaskData,
    *,
    attribution: bool = True,
    memorization: bool = True,
    gates: bool = True,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Collect enabled probes; disabled mechanisms do no work and emit no fields."""
    was_training = model.training
    model.eval()
    try:
        logits, hidden = model(data.id.x, return_hidden=True)
        ood_correct = model(data.ood.x).argmax(-1) == data.ood.y
        metrics: dict[str, Any] = {
            "generalization": {
                "score": float(ood_correct.float().mean()),
                "method": "held-out structural OOD accuracy",
            },
            "sparsity": {
                "parameters": sum(p.numel() for p in model.parameters()),
                "active_parameters": sum(int(torch.count_nonzero(p)) for p in model.parameters()),
            },
        }
        if memorization:
            _, train_hidden = model(data.train.x, return_hidden=True)
            nearest = torch.cdist(
                hidden[-1][:, -1].float(), train_hidden[-1][:, -1].float()
            ).argmin(1)
            neighbors = data.train.y[nearest]
            metrics["memorization"] = {
                "score": float((logits.argmax(-1) == neighbors).float().mean()),
                "method": "training-representation 1NN agreement proxy",
            }
        if gates:
            metrics["gates"] = {
                name: module.statistics()
                for name, module in model.named_modules()
                if isinstance(module, ScalarGate)
            }
        if attribution and data.metadata["task"] == "retrieval":
            pos = data.metadata["relevant_position"]
            ablated = data.id.x.clone()
            ablated[:, pos] = 0
            ablated_logits = model(ablated)
            original_accuracy = float((logits.argmax(-1) == data.id.y).float().mean())
            retrieval_delta = original_accuracy - float(
                (ablated_logits.argmax(-1) == data.id.y).float().mean()
            )
            metrics["retrieval"] = {
                "score": retrieval_delta,
                "method": "relevant-memory ablation accuracy delta proxy",
            }
            values = data.metadata["values"]
            replacement = data.id.x.clone()
            replacement[:, pos] = replacement[:, pos] - data.id.y + (data.id.y + 1) % values
            counterfactual = model(replacement).argmax(-1)
            distractor_removed = torch.zeros_like(data.id.x)
            distractor_removed[:, pos] = data.id.x[:, pos]
            distractor_removed[:, -1] = data.id.x[:, -1]
            metrics["attribution"] = {
                "relevant_ablation_accuracy_delta": retrieval_delta,
                "counterfactual_replacement_accuracy": float(
                    (counterfactual == (data.id.y + 1) % values).float().mean()
                ),
                "distractor_removal_accuracy_delta": original_accuracy
                - float((model(distractor_removed).argmax(-1) == data.id.y).float().mean()),
                "counterfactual_changed_prediction_fraction": float(
                    (counterfactual != logits.argmax(-1)).float().mean()
                ),
            }
        return metrics, [h[:, -1].detach() for h in hidden]
    finally:
        model.train(was_training)


def representation_diagnostics(
    hidden: list[torch.Tensor],
    *,
    reference_hidden: list[torch.Tensor] | None = None,
    max_samples: int = 128,
) -> dict[str, Any]:
    """Offline summaries on aligned final-token representations, without model calls.

    Radius graphs use the median pairwise Euclidean distance at each layer. They
    describe connectivity at that declared scale, not persistent homology. For
    comparisons, report raw median radii as well: per-layer scaling can conceal
    absolute distance changes. Reconstruction compares final states only, allowing
    an adapter to append a depth sample without implying aligned layer counts.
    """
    if not hidden or max_samples < 2:
        raise ValueError("need representations and max_samples >= 2")
    matrices = [observation_matrix(h)[:max_samples] for h in hidden]
    if any(x.shape != matrices[0].shape for x in matrices):
        raise ValueError("trajectory representations must have aligned observation and width axes")
    graphs = []
    for x in matrices:
        distances = torch.pdist(x)
        radius = float(distances.median()) if distances.numel() else 0.0
        graphs.append({**connectivity(x, radius), "sample_count": len(x)})
    result: dict[str, Any] = {
        "spectra": [spectral_statistics(x) for x in matrices],
        "geometry": [geometry_statistics(x, max_samples=max_samples) for x in matrices],
        "topology": graphs,
        "trajectory": trajectory_statistics(torch.stack(matrices))
        if len(matrices) >= 3
        else {
            "available": False,
            "reason": "requires at least three aligned depth samples",
        },
        "sampling": "first aligned observations, deterministic cap",
        "max_samples": max_samples,
    }
    if reference_hidden is not None:
        if not reference_hidden or hidden[-1].shape != reference_hidden[-1].shape:
            raise ValueError("reference final states must have aligned observations and width")
        reference = observation_matrix(reference_hidden[-1])[:max_samples]
        final = matrices[-1]
        ref_distances, final_distances = torch.pdist(reference), torch.pdist(final)
        denominator = float(ref_distances.norm())
        relative_error = (
            float((final_distances - ref_distances).norm()) / denominator
            if denominator > 0
            else None
        )
        result["reconstruction"] = {
            "final_hidden_mse": float(F.mse_loss(final, reference)),
            "pairwise_distance_relative_error": relative_error,
            "distance_metric": "euclidean final lifted representation",
            "zero_reference_distance": denominator == 0,
            "sample_count": len(final),
        }
    return result


@torch.no_grad()
def horizon_sweep(model: nn.Module, data: TaskData) -> dict[str, Any]:
    if data.metadata["task"] != "retrieval":
        return {"available": False, "reason": "requires movable relevant memory"}
    was_training = model.training
    model.eval()
    try:
        length, pos = data.id.x.shape[1], data.metadata["relevant_position"]
        distances = sorted({1, data.metadata["distance"], length // 2, length - 1})
        accuracies, sensitivity = [], []
        for distance in distances:
            moved = data.id.x.clone()
            target = length - 1 - distance
            moved[:, pos], moved[:, target] = data.id.x[:, target], data.id.x[:, pos]
            accuracy = float((model(moved).argmax(-1) == data.id.y).float().mean())
            moved[:, target] = 0
            ablated = float((model(moved).argmax(-1) == data.id.y).float().mean())
            accuracies.append(accuracy)
            sensitivity.append(accuracy - ablated)
        return {
            **effective_horizons(distances, accuracies, length - 1, threshold=0.75),
            "causal_threshold_0_1": effective_horizons(
                distances, accuracies, length - 1, threshold=0.1, causal=sensitivity
            )["causal_horizon"],
            "distances": distances,
            "accuracy": accuracies,
            "ablation_delta": sensitivity,
        }
    finally:
        model.train(was_training)


def coding_proxy(model: nn.Module, data: TaskData, seed: int, budget: int = 16) -> dict[str, Any]:
    """Ordered online linear probe on frozen representations, scored before each update."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, h = model(data.train.x, return_hidden=True)
            z = h[-1][:, -1].detach()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            probe = nn.Linear(z.shape[-1], data.classes)
            optimizer = torch.optim.SGD(probe.parameters(), lr=0.01)
            probabilities = []
            count = min(budget, len(z))
            for index in range(count):
                logits = probe(z[index : index + 1])
                probabilities.append(float(logits.detach().softmax(-1)[0, data.train.y[index]]))
                optimizer.zero_grad()
                F.cross_entropy(logits, data.train.y[index : index + 1]).backward()
                optimizer.step()
            return {
                **PrequentialCodeLengthProxy().estimate(probabilities),
                "compute_budget_updates": count,
                "seed": seed,
                "structural_description_bits": None,
                "probe": "online linear SGD",
                "residual_predictive_bits": -sum(torch.log2(torch.tensor(probabilities)).tolist()),
            }
    finally:
        model.train(was_training)


def execution_demo(
    *, executable: bool, teacher: bool, process: bool, verified: bool, gamma: float
) -> dict[str, Any]:
    from agg.credit import assign_credit
    from agg.supervision import Signal, ToyEnvironment

    env = ToyEnvironment(1)
    signals: list[Signal] = []
    if executable or verified:
        env.execute("add", 2, intent="reach 3", expected_state=3)
        env.execute("divide", 0, intent="I successfully reached 9", expected_state=9)
        env.execute("multiply", 2, intent="reach 6", expected_state=6)
        if verified:
            signals.extend(event.signal() for event in env.trace)
        elif executable:
            signals.extend(
                Signal(
                    "execution_log",
                    1 if e.error is None else -1,
                    0.95,
                    e.order,
                    "toy executor status",
                )
                for e in env.trace
            )
    if teacher:
        signals.append(Signal("teacher", 1, 0.5, 1, "deliberately incorrect teacher fixture"))
    if process:
        signals.append(Signal("process", 1, 0.4, 1, "deliberately incorrect process fixture"))
    return {
        "trace": [asdict(e) for e in env.trace],
        "credit": [asdict(c) for c in assign_credit(signals, 3, gamma)],
        "gamma": gamma,
        "scope": "execution and credit demonstration, not agent learning",
    }
