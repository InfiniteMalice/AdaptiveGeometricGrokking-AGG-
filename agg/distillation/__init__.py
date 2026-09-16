"""Ground-truth-anchored distillation; acceptance needs independent ID/OOD checks."""

import math

import torch.nn.functional as F
from torch import Tensor


def distillation_loss(
    student_logits: Tensor,
    labels: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
    teacher_weight: float = 0.5,
    student_hidden: Tensor | None = None,
    teacher_hidden: Tensor | None = None,
    representation_weight: float = 0.0,
    relation_weight: float = 0.0,
) -> Tensor:
    """CE + weighted T^2 KL, with optional MSE and normalized Gram preservation.

    Ground truth always retains unit weight. Teacher logits/representations are
    detached. This training objective is not evidence of candidate acceptance.
    """
    weights = (teacher_weight, representation_weight, relation_weight)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("distillation weights must be finite and nonnegative")
    loss = F.cross_entropy(student_logits, labels)
    if teacher_weight:
        loss = loss + teacher_weight * temperature**2 * F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        )
    if representation_weight or relation_weight:
        if student_hidden is None or teacher_hidden is None:
            raise ValueError("representation terms require student and teacher hidden states")
        s = student_hidden.reshape(-1, student_hidden.shape[-1])
        t = teacher_hidden.detach().reshape(-1, teacher_hidden.shape[-1])
        if s.shape[0] != t.shape[0]:
            raise ValueError("student and teacher representations need aligned observations")
        if representation_weight:
            if s.shape != t.shape:
                raise ValueError("representation MSE requires equal feature dimensions")
            loss = loss + representation_weight * F.mse_loss(s, t)
        if relation_weight:
            s, t = F.normalize(s, dim=-1), F.normalize(t, dim=-1)
            loss = loss + relation_weight * F.mse_loss(s @ s.T, t @ t.T)
    return loss
