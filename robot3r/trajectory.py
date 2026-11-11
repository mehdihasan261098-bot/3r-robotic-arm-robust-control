"""
Classical joint-space trajectories (docs/THEORY.md §6.1).

Every trajectory object exposes ``t0``, ``tf`` and ``evaluate(t) -> (q, qd, qdd)``
(scalar ``t`` -> arrays of shape ``(n,)``, array ``t`` -> ``(N, n)``). Outside
``[t0, tf]`` the end positions are held with zero velocity/acceleration.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .robot import Elbow, RobotArm3R, SingularityWarning


def _hold(t_arr: np.ndarray, t0: float, tf: float, q0, qf, q, qd, qdd):
    """Hold boundary positions (zero velocity / acceleration) outside [t0, tf]."""
    before = (t_arr < t0)[:, None]
    after = (t_arr > tf)[:, None]
    q = np.where(before, q0, np.where(after, qf, q))
    qd = np.where(before | after, 0.0, qd)
    qdd = np.where(before | after, 0.0, qdd)
    return q, qd, qdd


class QuinticTrajectory:
    """Quintic polynomial between two joint-space states (general boundary conditions).

    Parameters
    ----------
    q0, qf : array_like, shape (n,)
    t0, tf : float
    v0, vf, a0, af : array_like, optional
        Boundary velocities / accelerations (default zero: rest-to-rest).
    """

    def __init__(self, q0, qf, t0: float, tf: float, v0=None, vf=None, a0=None, af=None) -> None:
        if tf <= t0:
            raise ValueError("tf must be greater than t0")
        self.q0 = np.atleast_1d(np.asarray(q0, dtype=float))
        self.qf = np.atleast_1d(np.asarray(qf, dtype=float))
        n = self.q0.size
        z = np.zeros(n)
        v0 = z if v0 is None else np.asarray(v0, float)
        vf = z if vf is None else np.asarray(vf, float)
        a0 = z if a0 is None else np.asarray(a0, float)
        af = z if af is None else np.asarray(af, float)
        self.t0, self.tf, self.n_joints = float(t0), float(tf), n
        T = self.tf - self.t0
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
            [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
        ], dtype=float)
        self.coeffs = np.linalg.solve(A, np.vstack([self.q0, v0, a0, self.qf, vf, af]))  # (6, n)

    @property
    def duration(self) -> float:
        return self.tf - self.t0

    def evaluate(self, t) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        tau = np.clip(t_arr - self.t0, 0.0, self.duration)[:, None]
        k = np.arange(6)
        P = tau**k
        dP = np.where(k > 0, k * tau ** np.maximum(k - 1, 0), 0.0)
        ddP = np.where(k > 1, k * (k - 1) * tau ** np.maximum(k - 2, 0), 0.0)
        out = _hold(t_arr, self.t0, self.tf, self.q0, self.qf, P @ self.coeffs, dP @ self.coeffs,
                    ddP @ self.coeffs)
        return tuple(o[0] for o in out) if np.ndim(t) == 0 else out

    __call__ = evaluate


class PiecewiseQuinticTrajectory:
    """C^2 trajectory through via points, one quintic per segment.

    Interior via-point velocities follow Craig's heuristic (average of adjacent
    slopes when they have the same sign, otherwise zero); accelerations are zero.
    """

    def __init__(self, waypoints, times, via_velocities: Union[str, np.ndarray] = "heuristic") -> None:
        W = np.asarray(waypoints, dtype=float)
        T = np.asarray(times, dtype=float)
        if W.ndim != 2 or len(W) < 2 or T.shape != (len(W),) or np.any(np.diff(T) <= 0):
            raise ValueError("need K >= 2 waypoints and strictly increasing times")
        if isinstance(via_velocities, str):
            V = np.zeros_like(W)
            if via_velocities == "heuristic":
                d = np.diff(W, axis=0) / np.diff(T)[:, None]
                for k in range(1, len(T) - 1):
                    V[k] = np.where(np.sign(d[k - 1]) == np.sign(d[k]), 0.5 * (d[k - 1] + d[k]), 0.0)
            elif via_velocities != "zero":
                raise ValueError("via_velocities must be 'heuristic', 'zero' or an array")
        else:
            V = np.asarray(via_velocities, dtype=float)
        self.waypoints, self.times = W, T
        self.segments = [QuinticTrajectory(W[k], W[k + 1], T[k], T[k + 1], v0=V[k], vf=V[k + 1])
                         for k in range(len(T) - 1)]
        self.t0, self.tf, self.n_joints = float(T[0]), float(T[-1]), W.shape[1]

    def evaluate(self, t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.clip(np.searchsorted(self.times, t_arr, side="right") - 1, 0, len(self.segments) - 1)
        q = np.empty((t_arr.size, self.n_joints))
        qd, qdd = np.empty_like(q), np.empty_like(q)
        for k in np.unique(idx):
            m = idx == k
            q[m], qd[m], qdd[m] = self.segments[k].evaluate(t_arr[m])
        return (q[0], qd[0], qdd[0]) if np.ndim(t) == 0 else (q, qd, qdd)

    __call__ = evaluate


def joint_waypoints_from_cartesian(robot: RobotArm3R, points: Sequence[Sequence[float]],
                                   elbow: Elbow = "up", k_cont: float = 0.5) -> np.ndarray:
    """IK for a sequence of tool positions with redundancy resolution and continuity.

    The first point maximises manipulability; later points add a penalty on the
    distance to the previous configuration so the joint path does not jump
    between members of the self-motion family.
    """
    Q, q_prev = [], None
    for p in points:
        q = robot.inverse_kinematics(p, elbow=elbow, q_prev=q_prev, k_cont=k_cont if q_prev is not None else 0.0)
        Q.append(q)
        q_prev = q
    return np.array(Q)


def check_trajectory_singularities(robot: RobotArm3R, trajectory, tol: float = 0.05,
                                   n_samples: int = 1000, warn: bool = True) -> float:
    """Minimum positional manipulability along a trajectory (warns below ``tol``)."""
    ts = np.linspace(trajectory.t0, trajectory.tf, n_samples)
    w = robot.manipulability(trajectory.evaluate(ts)[0])
    k = int(np.argmin(w))
    if warn and w[k] < tol:
        warnings.warn(f"trajectory approaches a singularity at t = {ts[k]:.2f} s (w = {w[k]:.3g})",
                      SingularityWarning, stacklevel=2)
    return float(w[k])
