"""
Joint-space controllers for the 3R arm (docs/THEORY.md Part 7).

All controllers share the :class:`Controller` interface:

* ``torque(t, q, qd, ref, z)`` returns the commanded torque;
* ``n_states`` / ``state_derivative(...)`` / ``initial_state()`` describe
  internal states (e.g. the PID integral). The simulator integrates them.

Controllers are pure functions of time and state, which keeps them correct
with adaptive ODE solvers. Model-based controllers hold their own
:class:`~robot3r.robot.RobotArm3R` (the *nominal* model), which may differ from
the simulated plant.

Switch controllers with :func:`make_controller`::

    ctrl = make_controller("smc", nominal_model, switching="tanh")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

from .robot import RobotArm3R

Gain = Union[float, Sequence[float], np.ndarray]


@dataclass(frozen=True)
class Reference:
    """Desired joint position, velocity and acceleration at one instant."""

    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray


def _gain(k: Gain, n: int = 3) -> np.ndarray:
    k = np.broadcast_to(np.asarray(k, dtype=float), (n,)).copy()
    if np.any(k < 0):
        raise ValueError("gains must be non-negative")
    return k


class Controller(ABC):
    """Base class for joint-space controllers."""

    name: str = "controller"
    n_states: int = 0

    @abstractmethod
    def torque(self, t: float, q: np.ndarray, qd: np.ndarray, ref: Reference, z: np.ndarray) -> np.ndarray:
        """Commanded joint torque (3,)."""

    def state_derivative(self, t, q, qd, ref, z) -> np.ndarray:
        return np.zeros(0)

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.n_states)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class ZeroTorqueController(Controller):
    """No actuation (passive motion), used for model validation."""

    name = "Passive"

    def torque(self, t, q, qd, ref, z):
        return np.zeros(3)


# --------------------------------------------------------------------------- #
# 1. Independent-joint PID (§7.1)
# --------------------------------------------------------------------------- #
class PIDController(Controller):
    """Decentralised PID: ``tau = Kp e + Ki int(e) + Kd e'``.

    Parameters
    ----------
    kp, ki, kd : float or array_like, shape (3,)
        Diagonal gains.
    integral_limit : float or array_like, optional
        Anti-windup bound on ``|int(e)|`` per joint (conditional integration).
    """

    n_states = 3

    def __init__(self, kp: Gain, ki: Gain, kd: Gain, *, integral_limit: Optional[Gain] = None,
                 name: str = "PID") -> None:
        self.kp, self.ki, self.kd = _gain(kp), _gain(ki), _gain(kd)
        self.integral_limit = None if integral_limit is None else _gain(integral_limit)
        self.name = name

    def torque(self, t, q, qd, ref, z):
        return self.kp * (ref.q - q) + self.ki * z + self.kd * (ref.qd - qd)

    def state_derivative(self, t, q, qd, ref, z):
        e = ref.q - q
        if self.integral_limit is None:
            return e
        frozen = (np.abs(z) >= self.integral_limit) & (np.sign(e) == np.sign(z))
        return np.where(frozen, 0.0, e)


# --------------------------------------------------------------------------- #
# 2. Computed-torque control with friction compensation (§7.2)
# --------------------------------------------------------------------------- #
class ComputedTorqueController(Controller):
    """``tau = M^(qdd_d + Kd e' + Kp e) + C^ q' + G^ + F^(q')``.

    Parameters
    ----------
    model : RobotArm3R
        Nominal model (its friction model provides the compensation term).
    kp, kd : float or array_like
        Outer-loop gains [1/s^2], [1/s].
    compensate_friction : bool
        Include ``F^(q')`` (default True).
    """

    def __init__(self, model: RobotArm3R, kp: Gain, kd: Gain, *, compensate_friction: bool = True,
                 name: str = "CTC") -> None:
        self.model = model
        self.kp, self.kd = _gain(kp), _gain(kd)
        self.compensate_friction = compensate_friction
        self.name = name

    @classmethod
    def from_bandwidth(cls, model: RobotArm3R, wn: float = 10.0, zeta: float = 1.0, **kw):
        """``Kp = wn^2``, ``Kd = 2 zeta wn`` (critically damped for zeta = 1)."""
        return cls(model, kp=wn**2, kd=2.0 * zeta * wn, **kw)

    def torque(self, t, q, qd, ref, z):
        v = ref.qdd + self.kd * (ref.qd - qd) + self.kp * (ref.q - q)
        tau = self.model.mass_matrix(q) @ v + self.model.coriolis_vector(q, qd) + self.model.gravity_vector(q)
        if self.compensate_friction:
            tau = tau + self.model.friction_torque(qd)
        return tau


# --------------------------------------------------------------------------- #
# 3. Sliding-mode control (§7.3)
# --------------------------------------------------------------------------- #
class SlidingModeController(Controller):
    """Slotine-Li sliding-mode controller with boundary layer.

    ``s = e' + Lambda e``,  ``qd_r = qd_d + Lambda e``,
    ``tau = M^ qdd_r + C^(q, q') qd_r + G^ + F^ + K_D s + K sw(s / Phi)``

    Parameters
    ----------
    model : RobotArm3R
        Nominal model for the equivalent control.
    lam : float or array_like
        Sliding-surface slopes ``Lambda`` [1/s].
    kd : float or array_like
        Linear damping on ``s`` [N m s/rad].
    k : float or array_like
        Switching gains [N m]; must exceed the uncertainty bound ``|Delta_i|``.
    phi : float or array_like
        Boundary-layer widths [rad/s] (ignored for ``switching="sign"``).
    switching : {"sat", "tanh", "sign"}
        Switching function: boundary-layer saturation, smooth tanh, or ideal sign.
    """

    def __init__(self, model: RobotArm3R, lam: Gain, kd: Gain, k: Gain, phi: Gain = 0.2, *,
                 switching: Literal["sat", "tanh", "sign"] = "sat", name: Optional[str] = None) -> None:
        if switching not in ("sat", "tanh", "sign"):
            raise ValueError("switching must be 'sat', 'tanh' or 'sign'")
        self.model = model
        self.lam, self.kd, self.k = _gain(lam), _gain(kd), _gain(k)
        self.phi = _gain(phi)
        if switching != "sign" and np.any(self.phi <= 0):
            raise ValueError("boundary-layer width must be positive")
        self.switching = switching
        self.name = name or ("SMC" if switching == "sat" else f"SMC-{switching}")

    def sliding_variable(self, q, qd, ref) -> np.ndarray:
        return (ref.qd - qd) + self.lam * (ref.q - q)

    def switch(self, s: np.ndarray) -> np.ndarray:
        if self.switching == "sign":
            return np.sign(s)
        if self.switching == "tanh":
            return np.tanh(s / self.phi)
        return np.clip(s / self.phi, -1.0, 1.0)

    def torque(self, t, q, qd, ref, z):
        e, ed = ref.q - q, ref.qd - qd
        s = ed + self.lam * e
        qd_r = ref.qd + self.lam * e
        qdd_r = ref.qdd + self.lam * ed
        m = self.model
        tau_eq = (m.mass_matrix(q) @ qdd_r + m.coriolis_matrix(q, qd) @ qd_r
                  + m.gravity_vector(q) + m.friction_torque(qd))
        return tau_eq + self.kd * s + self.k * self.switch(s)


# --------------------------------------------------------------------------- #
# Factory and default gains
# --------------------------------------------------------------------------- #
#: Default gains (see README "Controller tuning").
DEFAULT_GAINS = {
    "pid": dict(kp=(400.0, 200.0, 40.0), ki=(200.0, 100.0, 20.0), kd=(60.0, 25.0, 4.0),
                integral_limit=(1.0, 1.0, 1.0)),
    "ctc": dict(wn=20.0, zeta=1.0),
    "smc": dict(lam=(10.0, 10.0, 10.0), kd=(20.0, 8.0, 1.0), k=(35.0, 18.0, 7.0), phi=(0.2, 0.3, 0.5)),
}


def make_controller(kind: str, model: RobotArm3R, **overrides) -> Controller:
    """Create a controller by name: ``"pid"``, ``"ctc"``, ``"smc"`` or ``"passive"``.

    ``overrides`` replace default gains, e.g. ``make_controller("ctc", m, wn=15)`` or
    ``make_controller("smc", m, switching="sign")``.
    """
    kind = kind.lower()
    if kind == "pid":
        return PIDController(**{**DEFAULT_GAINS["pid"], **overrides})
    if kind == "ctc":
        params = {**DEFAULT_GAINS["ctc"], **overrides}
        if "kp" in params and "kd" in params:
            params.pop("wn", None), params.pop("zeta", None)
            return ComputedTorqueController(model, **params)
        return ComputedTorqueController.from_bandwidth(model, **params)
    if kind == "smc":
        return SlidingModeController(model, **{**DEFAULT_GAINS["smc"], **overrides})
    if kind in ("passive", "none"):
        return ZeroTorqueController()
    raise ValueError(f"unknown controller '{kind}' (pid, ctc, smc, passive)")
