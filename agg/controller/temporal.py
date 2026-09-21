"""Windowed time regression, not second differences between noisy single samples."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from agg.telemetry.controller import Observation

from .config import TemporalConfig


@dataclass(frozen=True)
class TemporalSummary:
    current: float
    ema: float
    rolling_mean: float
    rolling_variance: float
    first_derivative: float | None
    smoothed_derivative: float | None
    second_derivative: float | None
    residual_std: float | None
    samples: int
    confidence: float
    trend: str


class TimeSeries:
    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self.values: deque[tuple[int, float]] = deque(maxlen=self.config.window)
        self.ema: float | None = None

    def update(self, step: int, value: float) -> TemporalSummary:
        if type(step) is not int or step < 0 or not math.isfinite(value):
            raise ValueError("finite value and nonnegative integer step required")
        previous = self.values[-1] if self.values else None
        if previous and step <= previous[0]:
            raise ValueError("time-series steps must increase strictly")
        if previous and step - previous[0] > self.config.max_gap:
            self.values.clear()
            self.ema = None
            previous = None
        first = (value - previous[1]) / (step - previous[0]) if previous else None
        self.values.append((step, value))
        alpha = self.config.ema_alpha
        self.ema = value if self.ema is None else alpha * value + (1 - alpha) * self.ema
        y = np.array([v for _, v in self.values], dtype=float)
        slope = acceleration = residual = None
        confidence = min(len(y) / self.config.min_samples, 1.0)
        trend = "insufficient_evidence"
        if len(y) >= self.config.min_samples:
            # Center and scale time to avoid ill-conditioning for large global steps.
            t = np.array([s - step for s, _ in self.values], dtype=float)
            scale = float(-t[0])
            x = t / scale
            design = np.column_stack((np.ones_like(x), x, x * x))
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            slope = float(coefficients[1] / scale)
            acceleration = float(2 * coefficients[2] / scale**2)
            residual = float(np.sqrt(np.mean((y - design @ coefficients) ** 2)))
            confidence /= 1 + (residual / self.config.noise_threshold) ** 2
            if residual > self.config.noise_threshold:
                trend = "unstable"
            elif abs(slope) <= self.config.plateau_slope:
                trend = "plateauing"
            elif slope < 0:
                trend = "degrading"
            elif acceleration < -self.config.acceleration_threshold:
                trend = "improving_but_decelerating"
            else:
                trend = "improving"
        else:
            confidence *= 0.5
        return TemporalSummary(
            value,
            self.ema,
            float(y.mean()),
            float(y.var()),
            first,
            slope,
            acceleration,
            residual,
            len(y),
            confidence,
            trend,
        )


class TemporalTelemetry:
    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self.series: dict[str, TimeSeries] = {}
        self.last_step = -1
        self.component: str | None = None
        self._proxy_metrics: set[str] = set()

    def update(self, observation: Observation) -> dict[str, TemporalSummary]:
        if observation.step <= self.last_step:
            raise ValueError("observation steps must increase strictly")
        if self.component is not None and self.component != observation.component:
            raise ValueError("use a separate controller per component")
        metrics = observation.metrics()
        self.component = observation.component
        self.last_step = observation.step
        if observation.out_of_distribution or observation.contradictory:
            self.series.clear()
            self._proxy_metrics.clear()
            return {}
        proxies = set(observation.proxy_metrics)
        # A measured sample cannot upgrade a window of proxy observations into
        # measured evidence (or mix protocols across a provenance transition).
        for name in self._proxy_metrics ^ proxies:
            self.series.pop(name, None)
        self._proxy_metrics = proxies
        # A missing observation breaks support rather than reusing stale evidence.
        for name in self.series.keys() - metrics.keys():
            del self.series[name]
        return {
            name: self.series.setdefault(name, TimeSeries(self.config)).update(
                observation.step, value
            )
            for name, value in metrics.items()
        }
