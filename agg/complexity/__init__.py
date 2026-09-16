"""Compute-bounded structure interface with an explicitly limited coding proxy."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class EpiplexityEstimator(ABC):
    @abstractmethod
    def estimate(self, predictive_probabilities: Sequence[float]) -> dict[str, Any]:
        """Estimate from probabilities scored before learning each observation."""


class PrequentialCodeLengthProxy(EpiplexityEstimator):
    """Predictive code length PROXY, not a true epiplexity or model-MDL estimator.

    The caller must supply pre-update probabilities from a fixed ordered learning
    protocol. This statistic cannot establish that the protocol was obeyed.
    """

    def estimate(self, predictive_probabilities: Sequence[float]) -> dict[str, Any]:
        if not predictive_probabilities or any(
            not math.isfinite(p) or not 0 < p <= 1 for p in predictive_probabilities
        ):
            raise ValueError("Need nonempty finite pre-update probabilities in (0, 1]")
        bits = -sum(math.log2(p) for p in predictive_probabilities)
        return {
            "bits": bits,
            "bits_per_observation": bits / len(predictive_probabilities),
            "observations": len(predictive_probabilities),
            "proxy": True,
            "method": "prequential_predictive_code_length_proxy",
            "limitation": "Excludes model/compute cost; not true epiplexity",
        }
