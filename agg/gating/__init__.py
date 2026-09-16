"""Explicit scalar mixing gates and reversible causal interventions."""

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn


class ScalarGate(nn.Module):
    fixed: Tensor

    def __init__(self, value: float = 0.5, learned: bool = False) -> None:
        super().__init__()
        self._validate(value)
        if learned:
            if value in (0, 1):
                raise ValueError("learned gate initialization must be strictly between 0 and 1")
            self.logit = nn.Parameter(torch.tensor(math.log(value / (1 - value))))
        else:
            self.register_parameter("logit", None)
        self.register_buffer("fixed", torch.tensor(float(value)))
        self._forced: float | None = None

    @staticmethod
    def _validate(value: float) -> None:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("gate must be finite and in [0, 1]")

    def value(self) -> Tensor:
        if self._forced is not None:
            return self.fixed.new_tensor(self._forced)
        return self.fixed if self.logit is None else self.logit.sigmoid()

    def forward(self, base: Tensor, adapted: Tensor) -> Tensor:
        g = self.value()
        return (1 - g) * base + g * adapted

    @contextmanager
    def force(self, value: float) -> Iterator[None]:
        """Restore the prior intervention even on nested contexts or exceptions."""
        self._validate(value)
        old = self._forced
        self._forced = value
        try:
            yield
        finally:
            self._forced = old

    def statistics(self) -> dict[str, float | bool]:
        g = float(self.value().detach())
        entropy = 0.0 if g in (0, 1) else -g * math.log(g) - (1 - g) * math.log1p(-g)
        return {"value": g, "entropy": entropy, "saturated": g <= 0.01 or g >= 0.99}
