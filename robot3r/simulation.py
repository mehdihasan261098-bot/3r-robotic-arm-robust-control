"""
Closed-loop simulation with ``scipy.integrate.solve_ivp`` and performance metrics
(docs/THEORY.md §8.2-8.3).

Two modes:

* **continuous** (``control_period=None``): the controller is evaluated inside the
  ODE right-hand side; controller states are part of the ODE state.
* **sampled-data** (``control_period=Ts``): the controller runs every ``Ts``
  seconds with zero-order hold; ``solve_ivp`` integrates the plant between
  samples. This reproduces chattering of discontinuous control laws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .controllers import Controller, Reference
from .robot import RobotArm3R

_trapz = getattr(np, "trapezoid", None) or np.trapz


def detrend(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove the least-squares linear trend along ``axis`` (NumPy-only, like scipy.signal.detrend)."""
    x = np.moveaxis(np.asarray(x, dtype=float), axis, 0)
    n = x.shape[0]
    A = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    coef, *_ = np.linalg.lstsq(A, x.reshape(n, -1), rcond=None)
    return np.moveaxis((x.reshape(n, -1) - A @ coef).reshape(x.shape), 0, axis)


@dataclass
class SimulationResult:
    """Time histories on a uniform grid ``t`` (joint arrays have shape ``(N, 3)``)."""

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_ref: np.ndarray
    qd_ref: np.ndarray
    qdd_ref: np.ndarray
    tau: np.ndarray          # applied torque (after saturation)
    tau_d: np.ndarray        # external disturbance torque
    robot: RobotArm3R = field(repr=False)
    controller_name: str = "controller"
    scenario: str = "nominal"
    nfev: int = 0

    # -- derived signals -------------------------------------------------------
    @property
    def e(self) -> np.ndarray:
        return self.q_ref - self.q

    @property
    def ee(self) -> np.ndarray:
        return self.robot.forward_kinematics(self.q)

    @property
    def ee_ref(self) -> np.ndarray:
        return self.robot.forward_kinematics(self.q_ref)

    @property
    def ee_error(self) -> np.ndarray:
        return np.linalg.norm(self.ee_ref - self.ee, axis=1)

    @property
    def energy(self) -> np.ndarray:
        """Mechanical energy T + V of the plant [J]."""
        return self.robot.total_energy(self.q, self.qd)

    # -- metrics ---------------------------------------------------------------
    def chattering(self, f_cut: float = 20.0) -> Dict[str, float]:
        """Total-variation rate [N m/s] and high-frequency power fraction of the torque."""
        dt = self.t[1] - self.t[0]
        T = self.t[-1] - self.t[0]
        tv = float(np.sum(np.abs(np.diff(self.tau, axis=0))) / T)
        x = detrend(self.tau, axis=0)
        P = np.abs(np.fft.rfft(x, axis=0)) ** 2
        f = np.fft.rfftfreq(len(self.t), dt)
        hf = float(P[f > f_cut].sum() / max(P[1:].sum(), 1e-30))
        return {"tv_rate": tv, "hf_fraction": hf}

    def metrics(self) -> Dict[str, object]:
        e, tau, t = self.e, self.tau, self.t
        ee_err = self.ee_error
        ch = self.chattering()
        return {
            "controller": self.controller_name,
            "scenario": self.scenario,
            "rmse_joint": np.sqrt(np.mean(e**2, axis=0)),
            "max_joint_error": np.max(np.abs(e), axis=0),
            "rmse_ee": float(np.sqrt(np.mean(ee_err**2))),
            "max_ee_error": float(np.max(ee_err)),
            "final_ee_error": float(ee_err[-1]),
            "rms_torque": np.sqrt(np.mean(tau**2, axis=0)),
            "peak_torque": np.max(np.abs(tau), axis=0),
            "energy_mech": float(_trapz(np.sum(np.abs(tau * self.qd), axis=1), t)),
            "energy_tau2": float(_trapz(np.sum(tau**2, axis=1), t)),
            "tv_rate": ch["tv_rate"],
            "hf_fraction": ch["hf_fraction"],
            # lowest point reached by elbow, wrist or tool (table at y = 0): negative = collision
            "min_height": float(np.min(self.robot.joint_positions(self.q)[:, 1:, 1])),
        }


class Simulator:
    """Simulate the plant under feedback control while tracking a trajectory.

    Parameters
    ----------
    robot : RobotArm3R
        The plant (true dynamics, possibly with payload/parameter errors).
    controller : Controller
    trajectory : object with ``t0``, ``tf``, ``evaluate(t)``, optional
        Reference; if omitted the initial configuration is regulated.
    disturbance : callable ``d(t) -> (3,)``, optional
        External joint torque added to the plant.
    torque_limit : float or array_like, optional
        Actuator saturation [N m].
    control_period : float, optional
        Sample time of a digital controller (zero-order hold). ``None`` = continuous.
    rtol, atol, max_step : float
        ``solve_ivp`` settings (continuous mode).
    """

    def __init__(self, robot: RobotArm3R, controller: Controller, trajectory=None, *,
                 disturbance: Optional[Callable[[float], np.ndarray]] = None,
                 torque_limit: Optional[Sequence[float] | float] = None,
                 control_period: Optional[float] = None,
                 rtol: float = 1e-7, atol: float = 1e-9, max_step: float = 0.01,
                 scenario: str = "nominal") -> None:
        self.robot, self.controller, self.trajectory = robot, controller, trajectory
        self.disturbance = disturbance
        self.torque_limit = None if torque_limit is None else np.broadcast_to(
            np.asarray(torque_limit, dtype=float), (3,)).copy()
        self.control_period = control_period
        self.rtol, self.atol, self.max_step = rtol, atol, max_step
        self.scenario = scenario
        self._hold = np.zeros(3)

    # ------------------------------------------------------------------ helpers
    def reference(self, t: float) -> Reference:
        if self.trajectory is None:
            return Reference(self._hold, np.zeros(3), np.zeros(3))
        return Reference(*self.trajectory.evaluate(t))

    def tau_ext(self, t: float) -> np.ndarray:
        return np.zeros(3) if self.disturbance is None else np.asarray(self.disturbance(t), dtype=float)

    def saturate(self, tau: np.ndarray) -> np.ndarray:
        return tau if self.torque_limit is None else np.clip(tau, -self.torque_limit, self.torque_limit)

    def _rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        q, qd, z = x[0:3], x[3:6], x[6:]
        ref = self.reference(t)
        tau = self.saturate(self.controller.torque(t, q, qd, ref, z))
        qdd = self.robot.forward_dynamics(q, qd, tau, self.tau_ext(t))
        return np.concatenate([qd, qdd, self.controller.state_derivative(t, q, qd, ref, z)])

    # ---------------------------------------------------------------------- run
    def run(self, t_final: Optional[float] = None, *, q0=None, qd0=None, dt: float = 0.001) -> SimulationResult:
        """Simulate and return a :class:`SimulationResult` sampled every ``dt``.

        Defaults: ``t_final`` = trajectory end, ``q0``/``qd0`` = reference at t0.
        In sampled-data mode the output grid equals the control period.
        """
        traj = self.trajectory
        t0 = traj.t0 if traj is not None else 0.0
        if t_final is None:
            if traj is None:
                raise ValueError("t_final required without a trajectory")
            t_final = traj.tf
        if q0 is None:
            if traj is None:
                raise ValueError("q0 required without a trajectory")
            q0 = traj.evaluate(t0)[0]
        if qd0 is None:
            qd0 = traj.evaluate(t0)[1] if traj is not None else np.zeros(3)
        q0, qd0 = np.asarray(q0, float), np.asarray(qd0, float)
        self._hold = q0.copy()
        if self.control_period is not None:
            return self._run_sampled(t0, t_final, q0, qd0)

        x0 = np.concatenate([q0, qd0, self.controller.initial_state()])
        t_eval = t0 + dt * np.arange(int(round((t_final - t0) / dt)) + 1)
        sol = solve_ivp(self._rhs, (t0, t_eval[-1]), x0, t_eval=t_eval, method="RK45",
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        X = sol.y.T
        q, qd, z = X[:, :3], X[:, 3:6], X[:, 6:]
        refs = [self.reference(tk) for tk in sol.t]
        tau = np.array([self.saturate(self.controller.torque(tk, q[k], qd[k], refs[k], z[k]))
                        for k, tk in enumerate(sol.t)])
        return self._result(sol.t, q, qd, refs, tau, sol.nfev)

    def _run_sampled(self, t0: float, t_final: float, q0: np.ndarray, qd0: np.ndarray) -> SimulationResult:
        Ts = float(self.control_period)
        n = int(round((t_final - t0) / Ts))
        ts = t0 + Ts * np.arange(n + 1)
        x = np.concatenate([q0, qd0])
        z = self.controller.initial_state()
        Q, QD, TAU, refs, nfev = [q0], [qd0], [], [], 0
        for k in range(n):
            tk = ts[k]
            ref = self.reference(tk)
            refs.append(ref)
            tau = self.saturate(self.controller.torque(tk, x[:3], x[3:], ref, z))
            z = z + Ts * self.controller.state_derivative(tk, x[:3], x[3:], ref, z)   # forward Euler
            TAU.append(tau)

            def f(t, s, tau=tau):
                return np.concatenate([s[3:], self.robot.forward_dynamics(s[:3], s[3:], tau, self.tau_ext(t))])

            sol = solve_ivp(f, (tk, ts[k + 1]), x, method="RK45", rtol=self.rtol, atol=self.atol)
            x = sol.y[:, -1]
            nfev += sol.nfev
            Q.append(x[:3])
            QD.append(x[3:])
        refs.append(self.reference(ts[-1]))
        TAU.append(TAU[-1])
        return self._result(ts, np.array(Q), np.array(QD), refs, np.array(TAU), nfev)

    def _result(self, t, q, qd, refs, tau, nfev) -> SimulationResult:
        return SimulationResult(
            t=t, q=q, qd=qd,
            q_ref=np.array([r.q for r in refs]), qd_ref=np.array([r.qd for r in refs]),
            qdd_ref=np.array([r.qdd for r in refs]), tau=tau,
            tau_d=np.array([self.tau_ext(tk) for tk in t]),
            robot=self.robot, controller_name=self.controller.name, scenario=self.scenario, nfev=nfev,
        )


def format_metrics_table(results: Sequence[SimulationResult], label: str = "controller") -> str:
    """Markdown table of the main metrics of several runs."""
    deg = 180.0 / np.pi
    head = (f"| {label} | RMSE θ1/θ2/θ3 [deg] | Max \\|e\\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] "
            "| RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |")
    lines = [head, "|" + "---|" * 12]
    for r in results:
        m = r.metrics()
        name = r.controller_name if label == "controller" else r.scenario
        fmt = lambda v: "/".join(f"{x:.3g}" for x in v)  # noqa: E731
        lines.append(
            f"| {name} | {fmt(m['rmse_joint'] * deg)} | {fmt(m['max_joint_error'] * deg)} "
            f"| {m['rmse_ee'] * 1e3:.3g} | {m['max_ee_error'] * 1e3:.3g} "
            f"| {'/'.join(f'{x:.2f}' for x in m['rms_torque'])} | {'/'.join(f'{x:.1f}' for x in m['peak_torque'])} "
            f"| {m['energy_mech']:.1f} | {m['energy_tau2']:.0f} | {m['tv_rate']:.0f} | {100 * m['hf_fraction']:.2g} "
            f"| {m['min_height'] * 1e3:.0f} |")
    return "\n".join(lines)
