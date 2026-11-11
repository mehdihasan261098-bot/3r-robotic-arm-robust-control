"""
Optimisation-based trajectory planning: minimum energy and minimum time
(docs/THEORY.md §6.2-6.3).

The joint path is a quintic "backbone" plus a correction built from basis
functions that vanish together with their first two derivatives at both ends:

    q(s) = q_quintic(s) + sum_k c_k psi_k(s),   psi_k(s) = s^3 (1-s)^3 P_{k-1}(2s-1)

so any coefficient vector keeps the rest-to-rest boundary conditions.
Torques along the path come from the vectorised nominal inverse dynamics, and
the problem is solved with SLSQP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from numpy.polynomial import Legendre, Polynomial
from scipy.integrate import simpson
from scipy.optimize import minimize

from .robot import RobotArm3R


# =========================================================================== #
# Trajectory representation
# =========================================================================== #
class _Basis:
    """Quintic blend and correction basis functions with derivatives (in s)."""

    def __init__(self, K: int) -> None:
        self.K = K
        s = Polynomial([0, 1])
        bump = s**3 * (1 - s) ** 3
        self.psi = []
        for k in range(K):
            leg = Legendre.basis(k).convert(kind=Polynomial)         # P_k(x)
            self.psi.append(bump * leg(2 * s - 1))                   # P_k(2s - 1)
        self.blend = Polynomial([0, 0, 0, 10, -15, 6])               # 10 s^3 - 15 s^4 + 6 s^5

    def evaluate(self, s: np.ndarray):
        """Return blend (3 derivs) and basis matrices Psi, dPsi, ddPsi of shape (N, K)."""
        h = [self.blend.deriv(d)(s) for d in range(3)]
        Psi = [np.column_stack([p.deriv(d)(s) for p in self.psi]) if self.K else np.zeros((s.size, 0))
               for d in range(3)]
        return h, Psi


class OptimizedTrajectory:
    """Rest-to-rest joint trajectory ``q0 -> qf`` shaped by correction coefficients.

    Parameters
    ----------
    q0, qf : array_like, shape (3,)
    t0, duration : float
    coeffs : array_like, shape (K, 3)
        Correction coefficients (zeros reproduce the classical quintic).
    """

    def __init__(self, q0, qf, t0: float, duration: float, coeffs) -> None:
        self.q0 = np.asarray(q0, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self.coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
        self.t0, self.duration = float(t0), float(duration)
        self.tf = self.t0 + self.duration
        self.n_joints = self.q0.size
        self._basis = _Basis(self.coeffs.shape[0])

    def evaluate(self, t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        s = np.clip((t_arr - self.t0) / self.duration, 0.0, 1.0)
        (h, hd, hdd), (P, Pd, Pdd) = self._basis.evaluate(s)
        dq = self.qf - self.q0
        T = self.duration
        q = self.q0 + np.outer(h, dq) + P @ self.coeffs
        qd = (np.outer(hd, dq) + Pd @ self.coeffs) / T
        qdd = (np.outer(hdd, dq) + Pdd @ self.coeffs) / T**2
        before, after = (t_arr < self.t0)[:, None], (t_arr > self.tf)[:, None]
        q = np.where(before, self.q0, np.where(after, self.qf, q))
        qd = np.where(before | after, 0.0, qd)
        qdd = np.where(before | after, 0.0, qdd)
        return (q[0], qd[0], qdd[0]) if np.ndim(t) == 0 else (q, qd, qdd)

    __call__ = evaluate


# =========================================================================== #
# Planning problem
# =========================================================================== #
@dataclass
class PlanningLimits:
    """Constraints of the trajectory optimisation problem (THEORY.md §6.2)."""

    tau_max: Tuple[float, float, float] = (60.0, 30.0, 10.0)   # [N m]
    qd_max: float = 3.0                                        # [rad/s]
    y_floor: Optional[float] = 0.0                             # elbow/wrist/tool height [m]
    floor_margin: float = 0.02     # safety margin: constraints are only checked at collocation points
    use_joint_limits: bool = True


@dataclass
class PlanResult:
    """Outcome of :func:`plan_trajectory`."""

    trajectory: OptimizedTrajectory
    objective: str
    success: bool
    message: str
    n_iter: int
    report: Dict[str, float] = field(default_factory=dict)


def trajectory_report(robot: RobotArm3R, traj, n: int = 401, limits: Optional[PlanningLimits] = None) -> Dict[str, float]:
    """Energy and constraint statistics of a joint trajectory under the nominal model."""
    limits = limits or PlanningLimits()
    ts = np.linspace(traj.t0, traj.tf, n)
    q, qd, qdd = traj.evaluate(ts)
    tau = robot.inverse_dynamics(q, qd, qdd)
    y = robot.joint_positions(q)[:, 1:, 1]
    return {
        "duration_s": float(traj.tf - traj.t0),
        "energy_tau2": float(simpson(np.sum(tau**2, axis=1), x=ts)),
        "energy_mech_J": float(simpson(np.sum(np.abs(tau * qd), axis=1), x=ts)),
        "peak_tau_1": float(np.max(np.abs(tau[:, 0]))),
        "peak_tau_2": float(np.max(np.abs(tau[:, 1]))),
        "peak_tau_3": float(np.max(np.abs(tau[:, 2]))),
        "peak_qd": float(np.max(np.abs(qd))),
        "min_height_m": float(np.min(y)),
        "min_manipulability": float(np.min(robot.manipulability(q))),
        "torque_limit_ratio": float(np.max(np.abs(tau) / np.asarray(limits.tau_max))),
    }


def plan_trajectory(
    robot: RobotArm3R,
    q0,
    qf,
    duration: float,
    *,
    objective: Literal["energy", "time"] = "energy",
    n_basis: int = 5,
    n_colloc: int = 81,
    limits: Optional[PlanningLimits] = None,
    duration_bounds: Tuple[float, float] = (0.3, 10.0),
    maxiter: int = 1000,
    t0: float = 0.0,
) -> PlanResult:
    """Solve the minimum-energy or minimum-time planning problem with SLSQP.

    Parameters
    ----------
    robot : RobotArm3R
        Nominal model used to evaluate torques.
    q0, qf : array_like, shape (3,)
        Rest-to-rest boundary configurations (e.g. from redundancy-resolved IK).
    duration : float
        Motion time (fixed for ``"energy"``, initial guess for ``"time"``).
    objective : {"energy", "time"}
        ``"energy"`` minimises ``int tau^T tau dt``; ``"time"`` minimises the duration
        (plus a tiny energy regulariser).
    n_basis : int
        Number of correction functions per joint (decision variables = 3 n_basis [+1]).
    n_colloc : int
        Odd number of collocation points for the integral and the constraints.
    limits : PlanningLimits, optional
    """
    limits = limits or PlanningLimits()
    if n_colloc % 2 == 0:
        n_colloc += 1
    q0, qf = np.asarray(q0, float), np.asarray(qf, float)
    basis = _Basis(n_basis)
    s = np.linspace(0.0, 1.0, n_colloc)
    (h, hd, hdd), (P, Pd, Pdd) = basis.evaluate(s)
    dq = qf - q0
    tau_max = np.asarray(limits.tau_max, dtype=float)
    interior = slice(1, -1)  # endpoints are fixed by the boundary conditions
    nz = 3 * n_basis

    def unpack(z):
        C = z[:nz].reshape(n_basis, 3)
        T = z[nz] if objective == "time" else duration
        return C, T

    def path(z):
        C, T = unpack(z)
        q = q0 + np.outer(h, dq) + P @ C
        qd = (np.outer(hd, dq) + Pd @ C) / T
        qdd = (np.outer(hdd, dq) + Pdd @ C) / T**2
        return q, qd, qdd, T

    def energy(z):
        q, qd, qdd, T = path(z)
        tau = robot.inverse_dynamics(q, qd, qdd)
        return simpson(np.sum(tau**2, axis=1), x=s * T)

    z0 = np.zeros(nz + (1 if objective == "time" else 0))
    if objective == "time":
        z0[nz] = duration
    E_ref = max(energy(z0), 1e-9)

    def cost(z):
        if objective == "energy":
            return energy(z) / E_ref
        return z[nz] / duration + 1e-3 * energy(z) / E_ref

    def constraints(z):
        q, qd, qdd, _ = path(z)
        tau = robot.inverse_dynamics(q, qd, qdd)
        g = [(tau_max - tau).ravel() / tau_max.max(), (tau_max + tau).ravel() / tau_max.max(),
             (limits.qd_max - qd[interior]).ravel(), (limits.qd_max + qd[interior]).ravel()]
        if limits.use_joint_limits:
            lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
            g += [(q[interior] - lo).ravel(), (hi - q[interior]).ravel()]
        if limits.y_floor is not None:
            y = robot.joint_positions(q[interior])[:, 1:, 1]
            g.append((y - limits.y_floor - limits.floor_margin).ravel())
        return np.concatenate(g)

    bounds = [(None, None)] * nz
    if objective == "time":
        bounds.append(duration_bounds)
    res = minimize(cost, z0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": constraints}],
                   options={"maxiter": maxiter, "ftol": 1e-9})
    C, T = unpack(res.x)
    traj = OptimizedTrajectory(q0, qf, t0, T, C)
    feasible = bool(np.min(constraints(res.x)) > -1e-6)
    return PlanResult(traj, objective, bool(res.success) and feasible, str(res.message), int(res.nit),
                      trajectory_report(robot, traj, limits=limits))
