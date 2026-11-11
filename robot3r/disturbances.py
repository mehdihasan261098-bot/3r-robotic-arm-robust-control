"""
External joint disturbance torques (docs/THEORY.md §8.1).

A disturbance is any callable ``d(t) -> np.ndarray (3,)``. The provided classes
are smooth in time (pulses use tanh edges) so the ODE solver is never asked to
integrate across a discontinuity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class Sinusoid:
    """``A_i sin(2 pi f_i t + phase_i)`` on every joint."""

    amplitude: Sequence[float]
    frequency: Sequence[float]
    phase: Sequence[float] = (0.0, 0.0, 0.0)

    def __call__(self, t: float) -> np.ndarray:
        A, f, p = (np.asarray(v, dtype=float) for v in (self.amplitude, self.frequency, self.phase))
        return A * np.sin(2.0 * np.pi * f * t + p)


@dataclass(frozen=True)
class SmoothPulse:
    """Rectangular pulse with tanh edges on a single joint (an impact / push)."""

    joint: int
    amplitude: float
    t_start: float
    t_end: float
    rise_time: float = 0.01

    def __call__(self, t: float) -> np.ndarray:
        w = 0.5 * (np.tanh((t - self.t_start) / self.rise_time) - np.tanh((t - self.t_end) / self.rise_time))
        out = np.zeros(3)
        out[self.joint] = self.amplitude * w
        return out


@dataclass
class Disturbance:
    """Sum of disturbance components."""

    components: List = field(default_factory=list)
    name: str = "disturbance"

    def __call__(self, t: float) -> np.ndarray:
        out = np.zeros(3)
        for c in self.components:
            out = out + c(t)
        return out


def default_disturbance() -> Disturbance:
    """Disturbance profile used by the robustness study.

    * Low-frequency sinusoids on all joints (vibration / cable forces).
    * Three short pushes: 8 N m on joint 1, -4 N m on joint 2 during the motion,
      and 2 N m on joint 3 while holding the final pose.
    """
    return Disturbance([
        Sinusoid(amplitude=(3.0, 1.5, 0.5), frequency=(0.8, 1.3, 2.0), phase=(0.0, 0.5, 1.0)),
        SmoothPulse(joint=0, amplitude=8.0, t_start=1.0, t_end=1.3),
        SmoothPulse(joint=1, amplitude=-4.0, t_start=2.0, t_end=2.3),
        SmoothPulse(joint=2, amplitude=2.0, t_start=3.4, t_end=3.6),
    ], name="sinusoids + pulses")
