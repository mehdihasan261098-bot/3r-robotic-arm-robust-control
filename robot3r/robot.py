"""
Kinematic and dynamic model of a planar 3R serial manipulator (docs/THEORY.md, Parts 1-5).

Conventions
-----------
* ``q = [theta1, theta2, theta3]``; theta1 from the world x-axis, theta2 and theta3
  relative to the previous link.
* The world y-axis points up; gravity acts along -y.
* All dynamics functions are vectorised: ``q`` and ``qd`` may have shape ``(3,)``
  or ``(N, 3)``; matrices are then returned with shape ``(3, 3)`` or ``(N, 3, 3)``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

Elbow = Literal["up", "down"]
ArrayLike = Sequence[float] | np.ndarray


class WorkspaceError(ValueError):
    """Raised when a Cartesian target cannot be reached."""


class SingularityWarning(UserWarning):
    """Issued when an operation is evaluated at or near a kinematic singularity."""


# =========================================================================== #
# Parameters
# =========================================================================== #
@dataclass(frozen=True)
class RobotParams3R:
    """Physical parameters of the 3R arm (THEORY.md §1.1).

    Links are uniform slender rods (COM at mid-length, ``I = m l^2 / 12``).
    ``payload`` is a point mass at the tool tip merged with link 3 (§5.10).

    Attributes
    ----------
    lengths, masses : tuple of 3 floats
        Link lengths [m] and masses [kg].
    g : float
        Gravitational acceleration [m/s^2].
    payload : float
        Point mass at the tool tip [kg].
    viscous_friction : tuple of 3 floats
        Viscous coefficients ``b_i`` [N m s/rad].
    coulomb_friction : tuple of 3 floats
        Coulomb levels ``f_ci`` [N m] (set to zeros to disable).
    coulomb_eps : float
        Velocity scale of the ``tanh`` smoothing of Coulomb friction [rad/s].
    joint_limits : tuple of 3 (min, max) pairs
        Joint ranges [rad] used by IK and trajectory optimisation.
    floor : float or None
        Height of the table the arm is mounted on [m]. IK only returns postures
        whose tool lies above it and whose elbow and wrist are at least
        ``floor_clearance`` above it (``None`` disables the check).
    floor_clearance : float
        Required clearance of elbow and wrist above the table for IK [m].
    """

    lengths: Tuple[float, float, float] = (1.0, 0.8, 0.6)
    masses: Tuple[float, float, float] = (1.0, 0.8, 0.5)
    g: float = 9.81
    payload: float = 0.0
    viscous_friction: Tuple[float, float, float] = (0.25, 0.20, 0.15)
    coulomb_friction: Tuple[float, float, float] = (0.30, 0.20, 0.10)
    coulomb_eps: float = 0.05
    joint_limits: Tuple[Tuple[float, float], ...] = (
        (-0.5 * np.pi, 1.5 * np.pi),        # table-mounted shoulder: upper half-plane plus margin
        (np.deg2rad(-165.0), np.deg2rad(165.0)),
        (np.deg2rad(-165.0), np.deg2rad(165.0)),
    )
    floor: Optional[float] = 0.0
    floor_clearance: float = 0.1

    def __post_init__(self) -> None:
        if any(v <= 0 for v in (*self.lengths, *self.masses)):
            raise ValueError("lengths and masses must be positive")
        if self.payload < 0 or self.g < 0 or self.coulomb_eps <= 0:
            raise ValueError("payload, g must be >= 0 and coulomb_eps > 0")

    # -- derived quantities (with payload merged into link 3) -------------------
    @property
    def l(self) -> np.ndarray:
        return np.asarray(self.lengths, dtype=float)

    @property
    def m_eff(self) -> np.ndarray:
        """Masses with the payload added to link 3."""
        m = np.asarray(self.masses, dtype=float).copy()
        m[2] += self.payload
        return m

    @property
    def lc(self) -> np.ndarray:
        """COM distances from the proximal joint, payload included in link 3."""
        lc = 0.5 * self.l
        m3 = self.masses[2]
        lc[2] = (m3 * 0.5 * self.lengths[2] + self.payload * self.lengths[2]) / (m3 + self.payload)
        return lc

    @property
    def I(self) -> np.ndarray:
        """Inertias about each body's COM (parallel-axis theorem for link 3 + payload)."""
        m, l = np.asarray(self.masses, dtype=float), self.l
        inertia = m * l**2 / 12.0
        lc3 = self.lc[2]
        inertia[2] += m[2] * (0.5 * l[2] - lc3) ** 2 + self.payload * (l[2] - lc3) ** 2
        return inertia

    # -- convenience constructors ----------------------------------------------------
    def with_payload(self, payload: float) -> "RobotParams3R":
        return replace(self, payload=payload)

    def scaled(self, mass_scale: float = 1.0, friction_scale: float = 1.0) -> "RobotParams3R":
        """Copy with all link masses and friction coefficients scaled (uncertainty studies)."""
        return replace(
            self,
            masses=tuple(mass_scale * np.asarray(self.masses)),
            viscous_friction=tuple(friction_scale * np.asarray(self.viscous_friction)),
            coulomb_friction=tuple(friction_scale * np.asarray(self.coulomb_friction)),
        )


# =========================================================================== #
# Robot model
# =========================================================================== #
class RobotArm3R:
    """Planar three-link revolute (3R) robot arm.

    Provides forward kinematics, the geometric Jacobian with singularity
    measures, redundancy-resolving inverse kinematics (geometric and numerical),
    the Lagrangian dynamic model with friction
    ``M(q) q'' + C(q, q') q' + G(q) + F(q') = tau`` and drawing utilities.

    Parameters
    ----------
    params : RobotParams3R, optional
        Physical parameters; defaults to the project robot.
    singularity_tol : float
        Threshold on the smallest singular value of the positional Jacobian
        below which a configuration is reported as singular [m].

    Examples
    --------
    >>> robot = RobotArm3R()
    >>> q = robot.inverse_kinematics([1.2, 0.8])        # redundancy resolved automatically
    >>> np.allclose(robot.forward_kinematics(q), [1.2, 0.8])
    True
    """

    n_joints = 3

    def __init__(self, params: Optional[RobotParams3R] = None, singularity_tol: float = 1e-3) -> None:
        self.params = params if params is not None else RobotParams3R()
        self.singularity_tol = float(singularity_tol)
        p = self.params
        self.l1, self.l2, self.l3 = p.l
        m1, m2, m3 = p.m_eff
        lc1, lc2, lc3 = p.lc
        I1, I2, I3 = p.I
        l1, l2, g = self.l1, self.l2, p.g

        # Inertia constants (THEORY.md §5.5)
        self.a1 = I1 + I2 + I3 + m1 * lc1**2 + m2 * (l1**2 + lc2**2) + m3 * (l1**2 + l2**2 + lc3**2)
        self.a2 = I2 + I3 + m2 * lc2**2 + m3 * (l2**2 + lc3**2)
        self.a3 = I3 + m3 * lc3**2
        self.b1 = l1 * (m2 * lc2 + m3 * l2)
        self.b2 = m3 * l1 * lc3
        self.b3 = m3 * l2 * lc3
        # Gravity constants (§5.7)
        self.gamma1 = g * (m1 * lc1 + (m2 + m3) * l1)
        self.gamma2 = g * (m2 * lc2 + m3 * l2)
        self.gamma3 = g * m3 * lc3
        # Friction (§5.8)
        self.B = np.asarray(p.viscous_friction, dtype=float)
        self.Fc = np.asarray(p.coulomb_friction, dtype=float)
        self.eps = p.coulomb_eps
        self.joint_limits = np.asarray(p.joint_limits, dtype=float)
        self.floor = p.floor
        self.reach = float(np.sum(p.l))

    def __repr__(self) -> str:
        p = self.params
        return f"RobotArm3R(lengths={p.lengths}, masses={p.masses}, payload={p.payload})"

    # ======================================================================= #
    # Forward kinematics (Part 2)
    # ======================================================================= #
    @staticmethod
    def _abs_angles(q: np.ndarray) -> np.ndarray:
        """Absolute link angles [theta1, theta12, theta123], shape (..., 3)."""
        return np.cumsum(q, axis=-1)

    def joint_positions(self, q: ArrayLike) -> np.ndarray:
        """Base, elbow, wrist and tool positions, shape ``(..., 4, 2)``."""
        q = np.asarray(q, dtype=float)
        phi = self._abs_angles(q)
        l = self.params.l
        seg = np.stack([l * np.cos(phi), l * np.sin(phi)], axis=-1)   # (..., 3, 2) link vectors
        pts = np.cumsum(seg, axis=-2)                                 # elbow, wrist, tool
        base = np.zeros(pts.shape[:-2] + (1, 2))
        return np.concatenate([base, pts], axis=-2)

    def forward_kinematics(self, q: ArrayLike, return_orientation: bool = False) -> np.ndarray:
        """Tool position ``[x, y]`` (or pose ``[x, y, phi]``), shape ``(..., 2)`` / ``(..., 3)``."""
        q = np.asarray(q, dtype=float)
        p = self.joint_positions(q)[..., 3, :]
        if return_orientation:
            return np.concatenate([p, np.sum(q, axis=-1, keepdims=True)], axis=-1)
        return p

    # ======================================================================= #
    # Jacobian and singularities (Part 3)
    # ======================================================================= #
    def jacobian(self, q: ArrayLike) -> np.ndarray:
        """Planar geometric Jacobian (3x3) mapping ``q'`` to ``[x', y', phi']``."""
        q = np.asarray(q, dtype=float)
        phi = self._abs_angles(q)
        ls = self.params.l * np.sin(phi)
        lc = self.params.l * np.cos(phi)
        # Column i sums the contributions of links i..3 (reverse cumulative sum).
        Jx = -np.cumsum(ls[..., ::-1], axis=-1)[..., ::-1]
        Jy = np.cumsum(lc[..., ::-1], axis=-1)[..., ::-1]
        ones = np.ones_like(Jx)
        return np.stack([Jx, Jy, ones], axis=-2)

    def position_jacobian(self, q: ArrayLike) -> np.ndarray:
        """Positional Jacobian ``J_p`` (2x3)."""
        return self.jacobian(q)[..., :2, :]

    def spatial_jacobian(self, q: ArrayLike) -> np.ndarray:
        """Full 6x3 geometric Jacobian ``[v; omega]`` (rows vz, wx, wy are zero)."""
        J = self.jacobian(q)
        out = np.zeros(J.shape[:-2] + (6, 3))
        out[..., 0:2, :] = J[..., 0:2, :]
        out[..., 5, :] = 1.0
        return out

    def jacobian_dot(self, q: ArrayLike, qd: ArrayLike) -> np.ndarray:
        """Time derivative of the 3x3 Jacobian."""
        q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
        phi, w = self._abs_angles(q), self._abs_angles(qd)
        lcw = self.params.l * np.cos(phi) * w
        lsw = self.params.l * np.sin(phi) * w
        Jx = -np.cumsum(lcw[..., ::-1], axis=-1)[..., ::-1]
        Jy = -np.cumsum(lsw[..., ::-1], axis=-1)[..., ::-1]
        return np.stack([Jx, Jy, np.zeros_like(Jx)], axis=-2)

    def manipulability(self, q: ArrayLike) -> np.ndarray | float:
        """Yoshikawa manipulability of the position task, ``sqrt(det(J_p J_p^T))`` (§3.3)."""
        q = np.asarray(q, dtype=float)
        l1, l2, l3 = self.l1, self.l2, self.l3
        s2, s3, s23 = np.sin(q[..., 1]), np.sin(q[..., 2]), np.sin(q[..., 1] + q[..., 2])
        w2 = (l1 * l2 * s2 + l1 * l3 * s23) ** 2 + (l1 * l3 * s23 + l2 * l3 * s3) ** 2 + (l2 * l3 * s3) ** 2
        w = np.sqrt(w2)
        return float(w) if np.ndim(w) == 0 else w

    def singularity_index(self, q: ArrayLike, task: Literal["position", "pose"] = "position") -> float:
        """Smallest singular value of ``J_p`` (position task) or ``J`` (pose task)."""
        J = self.position_jacobian(q) if task == "position" else self.jacobian(q)
        return float(np.linalg.svd(J, compute_uv=False)[-1])

    def is_singular(self, q: ArrayLike, task: Literal["position", "pose"] = "position",
                    tol: Optional[float] = None) -> bool:
        """True if the configuration is (numerically) singular for the given task."""
        tol = self.singularity_tol if tol is None else tol
        return self.singularity_index(q, task) < tol

    @staticmethod
    def dls_pinv(J: np.ndarray, eps: float = 0.05, lambda_max: float = 0.05) -> np.ndarray:
        """Damped least-squares pseudo-inverse with singular-value-dependent damping (§3.4)."""
        sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
        lam2 = 0.0 if sigma_min >= eps else lambda_max**2 * (1.0 - (sigma_min / eps) ** 2)
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(J.shape[0]))

    # ======================================================================= #
    # Inverse kinematics (Part 4)
    # ======================================================================= #
    def ik_given_orientation(self, p: ArrayLike, phi: float, elbow: Elbow = "up") -> np.ndarray:
        """Closed-form IK for a full pose ``(x, y, phi)`` (§4.2-4.4).

        Raises
        ------
        WorkspaceError
            If the wrist point is not reachable by links 1-2.
        """
        x, y = float(p[0]), float(p[1])
        xw, yw = x - self.l3 * np.cos(phi), y - self.l3 * np.sin(phi)
        l1, l2 = self.l1, self.l2
        c2 = (xw**2 + yw**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
        if abs(c2) > 1.0 + 1e-9:
            raise WorkspaceError(f"wrist point ({xw:.3f}, {yw:.3f}) unreachable for phi = {phi:.3f} rad")
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2 = np.sqrt(1.0 - c2**2) * (-1.0 if elbow == "up" else 1.0)
        th2 = np.arctan2(s2, c2)
        th1 = np.arctan2(yw, xw) - np.arctan2(l2 * s2, l1 + l2 * c2)
        th3 = phi - th1 - th2
        q = _wrap(np.array([th1, th2, th3]))
        lo1 = self.joint_limits[0, 0]
        q[0] = lo1 + (q[0] - lo1) % (2.0 * np.pi)   # theta1 expressed inside its joint range
        return q

    def within_limits(self, q: ArrayLike, margin: float = 0.0) -> bool:
        """True if all joints are inside their ranges (shrunk by ``margin``)."""
        q = np.asarray(q, dtype=float)
        lo, hi = self.joint_limits[:, 0] + margin, self.joint_limits[:, 1] - margin
        return bool(np.all(q >= lo) and np.all(q <= hi))

    def admissible(self, q: ArrayLike) -> bool:
        """Joint limits respected, tool above the table and elbow/wrist at least
        ``params.floor_clearance`` above it."""
        if not self.within_limits(q):
            return False
        if self.floor is None:
            return True
        y = self.joint_positions(q)[1:, 1]
        c = self.params.floor_clearance
        return bool(y[0] >= self.floor + c and y[1] >= self.floor + c and y[2] >= self.floor - 1e-9)

    def feasible_orientations(self, p: ArrayLike, elbow: Elbow = "up", n: int = 721) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Grid of tool angles for which IK exists and the posture is admissible (§4.5).

        Returns
        -------
        phis : np.ndarray
            Admissible tool angles.
        sols : list of np.ndarray
            Corresponding joint solutions.
        """
        phis, sols = [], []
        for phi in np.linspace(-np.pi, np.pi, n, endpoint=False):
            try:
                q = self.ik_given_orientation(p, phi, elbow)
            except WorkspaceError:
                continue
            if self.admissible(q):
                phis.append(phi)
                sols.append(q)
        return np.asarray(phis), sols

    def _ik_objective(self, q: np.ndarray, q_prev: Optional[np.ndarray], k_cont: float, k_lim: float) -> float:
        """Secondary criterion to MINIMISE (negative of the §4.6 objective)."""
        cost = -self.manipulability(q)
        if q_prev is not None:
            cost += k_cont * float(np.sum(_wrap(q - q_prev) ** 2))
        if k_lim > 0:
            lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            cost += k_lim * float(np.sum(((q - mid) / half) ** 8))  # smooth, flat in the middle
        return cost

    def inverse_kinematics(
        self,
        p: ArrayLike,
        phi: Optional[float] = None,
        elbow: Elbow = "up",
        *,
        q_prev: Optional[ArrayLike] = None,
        k_cont: float = 0.0,
        k_lim: float = 0.1,
        n_grid: int = 721,
    ) -> np.ndarray:
        """Redundancy-resolving inverse kinematics (§4.6).

        Parameters
        ----------
        p : array_like, shape (2,)
            Desired tool position.
        phi : float, optional
            Tool orientation. If given, the problem is not redundant and the
            closed form is returned directly.
        elbow : {"up", "down"}
            Branch for links 1-2.
        q_prev : array_like, optional
            Previous configuration; with ``k_cont > 0`` the solution closest to it is preferred.
        k_cont, k_lim : float
            Weights of the continuity and joint-limit terms.

        Returns
        -------
        np.ndarray, shape (3,)

        Raises
        ------
        WorkspaceError
            If no admissible configuration exists.
        """
        if phi is not None:
            q = self.ik_given_orientation(p, phi, elbow)
            if self.manipulability(q) < self.singularity_tol:
                warnings.warn("IK solution is singular for positioning", SingularityWarning, stacklevel=2)
            return q
        if np.hypot(p[0], p[1]) > self.reach + 1e-12:
            raise WorkspaceError(f"target {tuple(p)} beyond reach {self.reach:.2f} m")
        qp = None if q_prev is None else np.asarray(q_prev, dtype=float)
        phis, sols = self.feasible_orientations(p, elbow, n_grid)
        if len(phis) == 0:
            raise WorkspaceError(f"no admissible configuration for target {tuple(p)} (elbow {elbow})")
        costs = np.array([self._ik_objective(q, qp, k_cont, k_lim) for q in sols])
        k = int(np.argmin(costs))
        # Brent refinement inside the neighbouring grid cells.
        step = 2.0 * np.pi / n_grid

        def f(ph: float) -> float:
            try:
                q = self.ik_given_orientation(p, ph, elbow)
            except WorkspaceError:
                return 1e6        # finite penalty keeps Brent's method well defined
            return self._ik_objective(q, qp, k_cont, k_lim) if self.admissible(q) else 1e6

        res = minimize_scalar(f, bounds=(phis[k] - step, phis[k] + step), method="bounded",
                              options={"xatol": 1e-8})
        phi_best = res.x if res.success and res.fun <= costs[k] else phis[k]
        q = self.ik_given_orientation(p, phi_best, elbow)
        if qp is not None:  # keep angles continuous with the previous configuration
            q = qp + _wrap(q - qp)
        return q

    def ik_numerical(
        self,
        p: ArrayLike,
        q0: ArrayLike,
        *,
        k_null: float = 0.5,
        tol: float = 1e-10,
        max_iter: int = 500,
    ) -> Tuple[np.ndarray, bool]:
        """Differential IK with null-space manipulability optimisation (§4.7).

        Returns the solution and a convergence flag. The position error is
        driven to zero with a DLS pseudo-inverse; null-space motion climbs the
        manipulability gradient without moving the tool.
        """
        q = np.asarray(q0, dtype=float).copy()
        pd = np.asarray(p, dtype=float)
        converged = False
        for _ in range(max_iter):
            err = pd - self.forward_kinematics(q)
            Jp = self.position_jacobian(q)
            Jpinv = self.dls_pinv(Jp)
            grad = _num_grad(self.manipulability, q)
            dq = Jpinv @ err + (np.eye(3) - Jpinv @ Jp) @ (k_null * grad)
            step = np.linalg.norm(dq)
            if step > 0.3:          # trust region for robustness far from the solution
                dq *= 0.3 / step
            q = q + dq
            if np.linalg.norm(err) < tol and np.linalg.norm((np.eye(3) - Jpinv @ Jp) @ grad) < 1e-6:
                converged = True
                break
        return q, converged

    # ======================================================================= #
    # Dynamics (Part 5) — vectorised over leading dimensions
    # ======================================================================= #
    def mass_matrix(self, q: ArrayLike) -> np.ndarray:
        """Inertia matrix ``M(q)`` (§5.5)."""
        q = np.asarray(q, dtype=float)
        c2, c3, c23 = np.cos(q[..., 1]), np.cos(q[..., 2]), np.cos(q[..., 1] + q[..., 2])
        m11 = self.a1 + 2 * self.b1 * c2 + 2 * self.b2 * c23 + 2 * self.b3 * c3
        m12 = self.a2 + self.b1 * c2 + self.b2 * c23 + 2 * self.b3 * c3
        m13 = self.a3 + self.b2 * c23 + self.b3 * c3
        m22 = self.a2 + 2 * self.b3 * c3
        m23 = self.a3 + self.b3 * c3
        m33 = np.full_like(m11, self.a3)
        return np.stack([
            np.stack([m11, m12, m13], -1),
            np.stack([m12, m22, m23], -1),
            np.stack([m13, m23, m33], -1),
        ], -2)

    def coriolis_matrix(self, q: ArrayLike, qd: ArrayLike) -> np.ndarray:
        """Coriolis/centrifugal matrix from Christoffel symbols (§5.6).

        With this choice ``M' - 2C`` is skew-symmetric.
        """
        q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
        be1 = self.b1 * np.sin(q[..., 1])
        be2 = self.b2 * np.sin(q[..., 1] + q[..., 2])
        be3 = self.b3 * np.sin(q[..., 2])
        w1, w2, w3 = qd[..., 0], qd[..., 1], qd[..., 2]
        w12, w123 = w1 + w2, w1 + w2 + w3
        z = np.zeros_like(be1)
        return np.stack([
            np.stack([-be1 * w2 - be2 * (w2 + w3) - be3 * w3,
                      -(be1 + be2) * w12 - (be2 + be3) * w3,
                      -(be2 + be3) * w123], -1),
            np.stack([(be1 + be2) * w1 - be3 * w3, -be3 * w3, -be3 * w123], -1),
            np.stack([(be2 + be3) * w1 + be3 * w2, be3 * w12, z], -1),
        ], -2)

    def coriolis_vector(self, q: ArrayLike, qd: ArrayLike) -> np.ndarray:
        """``h = C(q, q') q'`` without forming ``C`` (§5.6)."""
        q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
        be1 = self.b1 * np.sin(q[..., 1])
        be2 = self.b2 * np.sin(q[..., 1] + q[..., 2])
        be3 = self.b3 * np.sin(q[..., 2])
        w1, w2, w3 = qd[..., 0], qd[..., 1], qd[..., 2]
        h1 = (-be1 * w2 * (2 * w1 + w2) - be2 * (w2 + w3) * (2 * w1 + w2 + w3)
              - be3 * w3 * (2 * w1 + 2 * w2 + w3))
        h2 = (be1 + be2) * w1**2 - be3 * w3 * (2 * w1 + 2 * w2 + w3)
        h3 = be2 * w1**2 + be3 * (w1 + w2) ** 2
        return np.stack([h1, h2, h3], -1)

    def gravity_vector(self, q: ArrayLike) -> np.ndarray:
        """Gravity torques ``G(q) = dV/dq`` (§5.7)."""
        q = np.asarray(q, dtype=float)
        c = np.cos(self._abs_angles(q))
        g3 = self.gamma3 * c[..., 2]
        g2 = self.gamma2 * c[..., 1] + g3
        g1 = self.gamma1 * c[..., 0] + g2
        return np.stack([g1, g2, g3], -1)

    def friction_torque(self, qd: ArrayLike) -> np.ndarray:
        """Viscous + smoothed Coulomb friction ``F(q') = B q' + Fc tanh(q'/eps)`` (§5.8)."""
        qd = np.asarray(qd, dtype=float)
        return self.B * qd + self.Fc * np.tanh(qd / self.eps)

    def dynamics(self, q: ArrayLike, qd: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(M, C, G, F)`` at one or many states."""
        return self.mass_matrix(q), self.coriolis_matrix(q, qd), self.gravity_vector(q), self.friction_torque(qd)

    def inverse_dynamics(self, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike, include_friction: bool = True) -> np.ndarray:
        """``tau = M q'' + C q' + G (+ F)``, vectorised."""
        q, qd, qdd = (np.asarray(v, dtype=float) for v in (q, qd, qdd))
        tau = np.einsum("...ij,...j->...i", self.mass_matrix(q), qdd)
        tau = tau + self.coriolis_vector(q, qd) + self.gravity_vector(q)
        if include_friction:
            tau = tau + self.friction_torque(qd)
        return tau

    def forward_dynamics(self, q: ArrayLike, qd: ArrayLike, tau: ArrayLike,
                         tau_ext: Optional[ArrayLike] = None) -> np.ndarray:
        """``q'' = M^{-1}(tau + tau_ext - C q' - G - F)`` (§5.9)."""
        q, qd, tau = (np.asarray(v, dtype=float) for v in (q, qd, tau))
        rhs = tau - self.coriolis_vector(q, qd) - self.gravity_vector(q) - self.friction_torque(qd)
        if tau_ext is not None:
            rhs = rhs + np.asarray(tau_ext, dtype=float)
        return np.linalg.solve(self.mass_matrix(q), rhs)

    # -- energy --------------------------------------------------------------------
    def kinetic_energy(self, q: ArrayLike, qd: ArrayLike) -> np.ndarray | float:
        qd = np.asarray(qd, dtype=float)
        T = 0.5 * np.einsum("...i,...ij,...j->...", qd, self.mass_matrix(q), qd)
        return float(T) if np.ndim(T) == 0 else T

    def potential_energy(self, q: ArrayLike) -> np.ndarray | float:
        """``V`` with zero at the base height; consistent with ``G = dV/dq``."""
        s = np.sin(self._abs_angles(np.asarray(q, dtype=float)))
        V = self.gamma1 * s[..., 0] + self.gamma2 * s[..., 1] + self.gamma3 * s[..., 2]
        return float(V) if np.ndim(V) == 0 else V

    def total_energy(self, q: ArrayLike, qd: ArrayLike) -> np.ndarray | float:
        return self.kinetic_energy(q, qd) + self.potential_energy(q)

    # ======================================================================= #
    # Drawing
    # ======================================================================= #
    LINK_COLORS = ("#3b6ea8", "#5b9bd5", "#9cc3e6")

    def plot(self, q: ArrayLike, ax=None, *, color: Optional[str] = None, alpha: float = 1.0,
             linewidth: float = 7.0, show_frame: bool = True, label: Optional[str] = None):
        """Draw the arm in configuration ``q`` (links, joints, tool and base pedestal)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 6.5))
        pts = self.joint_positions(q)
        for i in range(3):
            c = color or self.LINK_COLORS[i]
            ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], "-", color=c, lw=linewidth, alpha=alpha,
                    solid_capstyle="round", zorder=3, label=label if i == 0 else None)
        ax.plot(pts[:3, 0], pts[:3, 1], "o", ms=9, mfc="white", mec="#1f2d3d", mew=2, alpha=alpha, zorder=4)
        ax.plot(pts[3, 0], pts[3, 1], "o", ms=7, color="#c0392b", alpha=alpha, zorder=4)
        if show_frame:
            self.draw_base(ax)
            self.format_axes(ax)
        return ax

    def draw_base(self, ax) -> None:
        """Draw the table (at ``params.floor``) and the shoulder mount."""
        from matplotlib.patches import Wedge

        lim = 1.08 * self.reach
        top = 0.0 if self.floor is None else self.floor
        ax.fill_between([-lim, lim], -lim, top, color="#f0f0f0", zorder=0, hatch="//", edgecolor="#d9d9d9")
        ax.axhline(top, color="#7f7f7f", lw=1.2, zorder=1)
        ax.add_patch(Wedge((0.0, top), 0.11, 0, 180, color="#4d4d4d", zorder=2))

    def format_axes(self, ax, lim: Optional[float] = None) -> None:
        lim = lim or 1.08 * self.reach
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.35, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.25)


# =========================================================================== #
# Helpers
# =========================================================================== #
def _wrap(a):
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _num_grad(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function."""
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g
