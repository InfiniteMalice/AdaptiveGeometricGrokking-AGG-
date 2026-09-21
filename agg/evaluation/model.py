"""Shared classifier evaluation used by research trials and controller trials."""

from collections.abc import Callable

import torch
from torch import nn

from agg.attribution import linear_cka
from agg.tasks import TaskData
from agg.training import evaluate

from . import Evaluation


def make_evaluator(reference: nn.Module, data: TaskData) -> Callable[[nn.Module], Evaluation]:
    """Freeze the original mechanism anchor; validation splits remain development data."""
    mode = reference.training
    reference.eval()
    try:
        with torch.no_grad():
            _, hidden = reference(data.id.x, return_hidden=True)
            reference_hidden = hidden[-1][:, -1].detach().clone()
    finally:
        reference.train(mode)
    metric = "balanced_accuracy" if data.metadata.get("task") == "hierarchy" else "accuracy"

    def evaluator(candidate: nn.Module) -> Evaluation:
        previous_mode = candidate.training
        candidate.eval()
        try:
            with torch.no_grad():
                first, hidden = candidate(data.id.x, return_hidden=True)
                second = candidate(data.id.x)
                mechanism = linear_cka(reference_hidden, hidden[-1][:, -1])
                stable = all(bool(torch.isfinite(p).all()) for p in candidate.parameters())
                stable &= bool(torch.isfinite(first).all())
                return Evaluation(
                    evaluate(candidate, data.id)[metric],
                    evaluate(candidate, data.ood)[metric],
                    mechanism,
                    cost=float(sum(int(torch.count_nonzero(p)) for p in candidate.parameters())),
                    stable=stable,
                    reproducible=torch.equal(first, second),
                )
        finally:
            candidate.train(previous_mode)

    return evaluator
