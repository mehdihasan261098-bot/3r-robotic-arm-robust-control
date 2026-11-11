"""
Figures and animation for the 3R project.

Every function returns the Matplotlib figure (or animation); the caller saves or shows it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from .robot import RobotArm3R
from .simulation import SimulationResult

DEG = 180.0 / np.pi
COLORS = {"PID": "#d62728", "CTC": "#1f77b4", "SMC": "#2ca02c", "SMC-sign": "#9467bd",
          "SMC-tanh": "#17becf", "Quintic": "#7f7f7f", "Min-energy": "#2ca02c", "Min-time": "#ff7f0e"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "figure.dpi": 100})


def color(name: str, i: int = 0) -> str:
    for key in sorted(COLORS, key=len, reverse=True):
        if name.startswith(key):
            return COLORS[key]
    return f"C{i}"


# =========================================================================== #
# Inverse kinematics / redundancy
# =========================================================================== #
def plot_ik_redundancy(robot: RobotArm3R, p: Sequence[float], q_star: Optional[np.ndarray] = None,
                       q_numeric: Optional[np.ndarray] = None) -> plt.Figure:
    """Self-motion family at one target and the manipulability criterion over the tool angle."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    robot.draw_base(ax1)
    for elbow, ls in (("up", "-"), ("down", "--")):
        phis, sols = robot.feasible_orientations(p, elbow, n=361)
        if len(phis) == 0:
            continue
        for q in sols[:: max(1, len(sols) // 9)]:
            pts = robot.joint_positions(q)
            ax1.plot(pts[:, 0], pts[:, 1], ls, color="0.6", lw=1.5, alpha=0.6, marker="o", ms=3)
        w = robot.manipulability(np.array(sols))
        ax2.plot(np.rad2deg(phis), w, ".", ms=3, label=f"elbow-{elbow} (admissible)",
                 color="#1f77b4" if elbow == "up" else "#ff7f0e")
    if q_star is not None:
        robot.plot(q_star, ax1, show_frame=False, label="selected (max. manipulability)")
        ax2.axvline(np.rad2deg(np.sum(q_star)), color="k", lw=1.2, ls=":")
        ax2.plot(np.rad2deg(np.sum(q_star)), robot.manipulability(q_star), "k*", ms=14, label="selected")
    if q_numeric is not None:
        robot.plot(q_numeric, ax1, color="#ff7f0e", alpha=0.8, linewidth=4, show_frame=False,
                   label="numerical IK (null-space)")
    ax1.plot(*p, "r*", ms=16, zorder=6, label="target")
    robot.format_axes(ax1)
    ax1.set_title("Self-motion family: all postures reaching the same point")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("tool angle φ [deg]")
    ax2.set_ylabel("manipulability w")
    ax2.set_title("Redundancy resolution criterion")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =========================================================================== #
# Trajectory planning
# =========================================================================== #
def plot_trajectory_comparison(robot: RobotArm3R, trajectories: Mapping[str, object],
                               reports: Mapping[str, Dict[str, float]], tau_max=None) -> plt.Figure:
    """Joint paths, velocities and torques (nominal inverse dynamics), tool paths and energies."""
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.5])
    ax_q = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_t = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_v = fig.add_subplot(gs[2, 0])
    ax_p = fig.add_subplot(gs[0:2, 3])
    ax_b = fig.add_subplot(gs[2, 1:3])
    ax_tab = fig.add_subplot(gs[2, 3])
    robot.draw_base(ax_p)
    for i, (name, tr) in enumerate(trajectories.items()):
        c = color(name, i)
        ts = np.linspace(tr.t0, tr.tf, 400)
        q, qd, qdd = tr.evaluate(ts)
        tau = robot.inverse_dynamics(q, qd, qdd)
        for j in range(3):
            ax_q[j].plot(ts, q[:, j] * DEG, color=c, lw=1.8, label=name)
            ax_t[j].plot(ts, tau[:, j], color=c, lw=1.8, label=name)
        ax_v.plot(ts, np.max(np.abs(qd), axis=1), color=c, lw=1.8, label=name)
        ee = robot.forward_kinematics(q)
        ax_p.plot(ee[:, 0], ee[:, 1], color=c, lw=2, label=name)
        for k in np.linspace(0, len(ts) - 1, 5).astype(int)[1:-1]:
            robot.plot(q[k], ax_p, color=c, alpha=0.35, linewidth=3, show_frame=False)
    first = next(iter(trajectories.values()))
    robot.plot(first.evaluate(first.t0)[0], ax_p, show_frame=False)
    robot.plot(first.evaluate(first.tf)[0], ax_p, show_frame=False)
    for j in range(3):
        ax_q[j].set_title(f"θ{j+1} [deg]")
        ax_t[j].set_title(f"τ{j+1} [N·m] (nominal inverse dynamics)")
        if tau_max is not None:
            for sgn in (-1, 1):
                ax_t[j].axhline(sgn * tau_max[j], color="k", ls=":", lw=1)
        ax_t[j].set_xlabel("time [s]")
    ax_v.set_title("max |joint speed| [rad/s]")
    ax_v.set_xlabel("time [s]")
    robot.format_axes(ax_p)
    ax_p.set_title("Tool paths")
    ax_p.legend(fontsize=8, loc="upper right")
    for ax in (*ax_q, *ax_t, ax_v):
        ax.grid(alpha=0.3)
    ax_q[0].legend(fontsize=8)

    names = list(reports)
    x = np.arange(2)
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        r = reports[n]
        vals = [r["energy_tau2"], r["energy_mech_J"] * 10]
        bars = ax_b.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, color=color(n, i), label=n)
        ax_b.bar_label(bars, labels=[f"{r['energy_tau2']:.0f}", f"{r['energy_mech_J']:.1f} J"], fontsize=8)
    ax_b.set_xticks(x, ["∫τᵀτ dt  [N²m²s]", "mechanical work ×10  [J]"])
    ax_b.set_title("Energy measures")
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3, axis="y")

    ax_tab.axis("off")
    rows = [[n, f"{reports[n]['duration_s']:.2f}", f"{reports[n]['torque_limit_ratio']:.2f}",
             f"{reports[n]['min_height_m']:.2f}", f"{reports[n]['min_manipulability']:.2f}"] for n in names]
    tab = ax_tab.table(cellText=rows, colLabels=["", "T\n[s]", "peak\nτ/τmax", "min\nheight [m]", "min\nw"],
                       loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 2.2)
    fig.suptitle("Trajectory planning: quintic vs optimisation-based", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# =========================================================================== #
# Tracking
# =========================================================================== #
def plot_tracking(result: SimulationResult, title: Optional[str] = None) -> plt.Figure:
    """Joint references vs actual, errors and torques (3x3 grid)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 8.5), sharex=True)
    c = color(result.controller_name)
    t = result.t
    for j in range(3):
        axes[0, j].plot(t, result.q_ref[:, j] * DEG, "k--", lw=1.2, label="reference")
        axes[0, j].plot(t, result.q[:, j] * DEG, color=c, lw=1.6, label="actual")
        axes[0, j].set_title(f"Joint {j+1}")
        axes[0, j].set_ylabel("θ [deg]")
        axes[1, j].plot(t, result.e[:, j] * DEG, color=c, lw=1.4)
        axes[1, j].axhline(0, color="0.6", lw=0.8)
        axes[1, j].set_ylabel("error [deg]")
        axes[2, j].plot(t, result.tau[:, j], color=c, lw=1.4, label="control τ")
        if np.any(result.tau_d[:, j]):
            axes[2, j].plot(t, result.tau_d[:, j], color="0.5", lw=1.0, ls="--", label="disturbance τ_d")
        axes[2, j].set_ylabel("torque [N·m]")
        axes[2, j].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(title or f"{result.controller_name} — {result.scenario}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_controller_comparison(results: Sequence[SimulationResult], title: str) -> plt.Figure:
    """Tool error (log), joint errors and torques of several controllers on one scenario."""
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3)
    ax_ee = fig.add_subplot(gs[0, :2])
    ax_e = [fig.add_subplot(gs[1, j], sharex=ax_ee) for j in range(3)]
    ax_t = [fig.add_subplot(gs[2, j], sharex=ax_ee) for j in range(3)]
    ax_tab = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(results):
        c = color(r.controller_name, i)
        ax_ee.semilogy(r.t, np.maximum(r.ee_error * 1e3, 1e-9), color=c, lw=1.6, label=r.controller_name)
        for j in range(3):
            ax_e[j].plot(r.t, r.e[:, j] * DEG, color=c, lw=1.3)
            ax_t[j].plot(r.t, r.tau[:, j], color=c, lw=1.3)
    if np.any(results[0].tau_d):
        for j in range(3):
            ax_t[j].plot(results[0].t, results[0].tau_d[:, j], color="0.4", ls="--", lw=1.0, label="τ_d")
        ax_t[0].legend(fontsize=8)
    ax_ee.set_title("Tool-tip position error ‖p_d − p‖ [mm] (log scale)")
    ax_ee.legend(fontsize=9, ncol=len(results))
    for j in range(3):
        ax_e[j].set_title(f"Joint {j+1} error [deg]")
        ax_t[j].set_title(f"Joint {j+1} torque [N·m]")
        ax_t[j].set_xlabel("time [s]")
    for ax in (ax_ee, *ax_e, *ax_t):
        ax.grid(alpha=0.3, which="both")
    ax_tab.axis("off")
    rows = []
    for r in results:
        m = r.metrics()
        rows.append([r.controller_name, f"{m['rmse_ee'] * 1e3:.3g}", f"{m['max_ee_error'] * 1e3:.3g}",
                     f"{m['energy_mech']:.1f}", f"{m['tv_rate']:.0f}"])
    tab = ax_tab.table(cellText=rows, colLabels=["", "RMSE\n[mm]", "max\n[mm]", "E_mech\n[J]", "TV\n[N·m/s]"],
                       loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.0)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# =========================================================================== #
# Robustness
# =========================================================================== #
def plot_robustness(results: Mapping[str, Mapping[str, SimulationResult]]) -> plt.Figure:
    """Heat map of tool RMSE and bar charts of worst-case error, peak torque and energy."""
    ctrls = list(results)
    scen = list(next(iter(results.values())))
    M = {c: {s: results[c][s].metrics() for s in scen} for c in ctrls}
    rmse = np.array([[M[c][s]["rmse_ee"] * 1e3 for s in scen] for c in ctrls])

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax_h = fig.add_subplot(gs[0, :2])
    ax_w = fig.add_subplot(gs[0, 2])
    ax_pk = fig.add_subplot(gs[1, 0])
    ax_en = fig.add_subplot(gs[1, 1])
    ax_mx = fig.add_subplot(gs[1, 2])

    im = ax_h.imshow(np.maximum(rmse, 1e-6), cmap="viridis_r", norm=LogNorm(vmin=max(rmse.min(), 1e-6), vmax=rmse.max()),
                     aspect="auto")
    ax_h.set_xticks(range(len(scen)), scen, rotation=15)
    ax_h.set_yticks(range(len(ctrls)), ctrls)
    for i in range(len(ctrls)):
        for j in range(len(scen)):
            v = rmse[i, j]
            ax_h.text(j, i, f"{v:.3g}" if v >= 1e-3 else f"{v:.1e}", ha="center", va="center",
                      color="white" if v > np.sqrt(rmse.max() * max(rmse.min(), 1e-6)) * 3 else "black", fontsize=10,
                      fontweight="bold")
    fig.colorbar(im, ax=ax_h, label="tool RMSE [mm] (log)")
    ax_h.set_title("Tool-tip RMSE [mm] by controller and scenario")
    for spine in ax_h.spines.values():
        spine.set_visible(False)

    x = np.arange(len(scen))
    w = 0.8 / len(ctrls)

    def bars(ax, key, f, ylabel, title, log=False):
        for i, c in enumerate(ctrls):
            ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, [f(M[c][s][key]) for s in scen], w, color=color(c, i), label=c)
        ax.set_xticks(x, [s.replace(" ", "\n", 1) for s in scen], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y", which="both")

    bars(ax_pk, "peak_torque", lambda v: v[0], "N·m", "Peak shoulder torque |τ1|")
    bars(ax_en, "energy_mech", lambda v: v, "J", "Mechanical energy ∫Σ|τᵢθ̇ᵢ| dt")
    bars(ax_mx, "max_ee_error", lambda v: max(v * 1e3, 1e-6), "mm", "Max tool error (log)", log=True)
    ax_pk.legend(fontsize=8)

    worst = [rmse[i].max() for i in range(len(ctrls))]
    b = ax_w.barh(ctrls, worst, color=[color(c, i) for i, c in enumerate(ctrls)])
    ax_w.bar_label(b, fmt="%.3g mm", fontsize=9)
    ax_w.set_xscale("log")
    ax_w.invert_yaxis()
    ax_w.set_title("Robustness: worst-case tool RMSE")
    ax_w.grid(alpha=0.3, axis="x", which="both")
    fig.suptitle("Robustness evaluation under payload, parameter error and disturbance", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_chattering(results: Sequence[SimulationResult], window=(1.6, 2.0), joint: int = 2) -> plt.Figure:
    """Torque zoom, spectrum and chattering indices of sampled-data controllers."""
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3)
    ax_full = fig.add_subplot(gs[0, :2])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[1, 1])
    ax_err = fig.add_subplot(gs[0, 2])
    ax_tab = fig.add_subplot(gs[1, 2])
    rows = []
    for i, r in enumerate(results):
        c = color(r.controller_name, i)
        ax_full.plot(r.t, r.tau[:, joint], color=c, lw=0.9, alpha=0.85, label=r.controller_name)
        m = (r.t >= window[0]) & (r.t <= window[1])
        ax_zoom.step(r.t[m], r.tau[m, joint], where="post", color=c, lw=1.0, label=r.controller_name)
        x = r.tau[:, joint] - r.tau[:, joint].mean()
        f = np.fft.rfftfreq(len(x), r.t[1] - r.t[0])
        P = np.abs(np.fft.rfft(x)) ** 2
        ax_spec.loglog(f[1:], P[1:] / P[1:].sum(), color=c, lw=1.0, label=r.controller_name)
        ax_err.semilogy(r.t, np.maximum(r.ee_error * 1e3, 1e-6), color=c, lw=1.2, label=r.controller_name)
        mt = r.metrics()
        rows.append([r.controller_name, f"{mt['tv_rate']:.0f}", f"{100 * mt['hf_fraction']:.2g} %",
                     f"{mt['rmse_ee'] * 1e3:.3g}", f"{mt['energy_mech']:.1f}"])
    ax_full.axvspan(*window, color="0.9", zorder=0)
    ax_full.set_title(f"Joint {joint + 1} torque [N·m] (1 kHz zero-order-hold controller)")
    ax_full.set_xlabel("time [s]")
    ax_full.legend(fontsize=8, ncol=len(results))
    ax_zoom.set_title(f"Zoom {window[0]}–{window[1]} s")
    ax_zoom.set_xlabel("time [s]")
    ax_spec.axvline(20, color="k", ls=":", lw=1)
    ax_spec.set_title("Normalised torque power spectrum")
    ax_spec.set_xlabel("frequency [Hz]")
    ax_err.set_title("Tool error [mm]")
    ax_err.set_xlabel("time [s]")
    for ax in (ax_full, ax_zoom, ax_spec, ax_err):
        ax.grid(alpha=0.3, which="both")
    ax_tab.axis("off")
    tab = ax_tab.table(cellText=rows, colLabels=["", "TV\n[N·m/s]", "power\n>20 Hz", "RMSE\n[mm]", "E_mech\n[J]"],
                       loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.0)
    fig.suptitle("Chattering: switching function of the sliding-mode controller", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_energy_balance(result: SimulationResult, dissipated: np.ndarray, residual: np.ndarray) -> plt.Figure:
    """Passive-motion validation: energy, cumulative friction dissipation and balance residual."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    E = result.energy
    a1.plot(result.t, E, "k", lw=2, label="mechanical energy E = T + V")
    a1.plot(result.t, E[0] - dissipated, "C1--", lw=1.6, label="E₀ − ∫ θ̇ᵀF dt (friction loss)")
    a1.set_ylabel("energy [J]")
    a1.legend()
    a2.plot(result.t, residual, "C3")
    a2.set_ylabel("balance residual [J]")
    a2.set_xlabel("time [s]")
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle("Model validation: passive motion with friction (τ = 0)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# =========================================================================== #
# Animation
# =========================================================================== #
def animate(results: SimulationResult | Sequence[SimulationResult], *, fps: int = 30, speed: float = 1.0,
            save_path: Optional[str | Path] = None, titles: Optional[Sequence[str]] = None,
            dpi: int = 90) -> FuncAnimation:
    """Side-by-side animation of one or several simulations.

    Each panel shows the arm (coloured links, joints, tool), a translucent
    reference arm, the desired and actual tool paths, an arrow while the
    disturbance torque deviates strongly from its median (pulses and sinusoid peaks), live error read-outs and a warning when any part of the arm is
    below the table. ``.gif`` files use Pillow; ``.mp4`` needs ffmpeg.
    """
    if isinstance(results, SimulationResult):
        results = [results]
    robot = results[0].robot
    t0, tf = results[0].t[0], results[0].t[-1]
    ft = np.arange(t0, tf + 1e-9, speed / fps)

    def interp(a, t):
        return np.column_stack([np.interp(ft, t, a[:, j]) for j in range(a.shape[1])])

    data = []
    for r in results:
        P = robot.joint_positions(interp(r.q, r.t))
        R = robot.joint_positions(interp(r.q_ref, r.t))
        data.append((P, R, np.interp(ft, r.t, r.ee_error), interp(r.tau_d, r.t)))
    # Axis limits fitted to the motion of all panels (plus margin).
    allpts = np.concatenate([np.concatenate([P, R]).reshape(-1, 2) for P, R, _, _ in data])
    xlo, xhi = allpts[:, 0].min() - 0.35, allpts[:, 0].max() + 0.35
    ylo, yhi = min(allpts[:, 1].min(), 0.0) - 0.25, allpts[:, 1].max() + 0.45
    aspect = (yhi - ylo) / (xhi - xlo)

    n = len(results)
    w = 5.0
    fig, axes = plt.subplots(1, n, figsize=(w * n, w * aspect + 0.9), squeeze=False)
    panels = []
    for i, (ax, r) in enumerate(zip(axes[0], results)):
        P, R, err, td = data[i]
        robot.draw_base(ax)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x [m]")
        ax.plot(r.ee_ref[:, 0], r.ee_ref[:, 1], color="k", ls="--", lw=1.0, alpha=0.5)
        c = color(r.controller_name, i)
        ghost, = ax.plot([], [], "-", color="0.55", lw=6, alpha=0.25, solid_capstyle="round")
        links = [ax.plot([], [], "-", color=lc, lw=7, solid_capstyle="round", zorder=3)[0] for lc in robot.LINK_COLORS]
        joints, = ax.plot([], [], "o", ms=8, mfc="white", mec="#1f2d3d", mew=2, zorder=4)
        tip, = ax.plot([], [], "o", ms=7, color="#c0392b", zorder=5)
        trail, = ax.plot([], [], "-", color=c, lw=1.8, zorder=2)
        dist = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2.5))
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8.5,
                      bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
        warn = ax.text(0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right", color="white", fontsize=9,
                       fontweight="bold", bbox=dict(boxstyle="round", fc="#c0392b", ec="none"))
        ax.set_title(titles[i] if titles else f"{r.controller_name} — {r.scenario}", color=c)
        pulse = np.abs(td - np.median(td, axis=0))  # highlight pulses rather than the sinusoids
        panels.append((ghost, links, joints, tip, trail, dist, txt, warn, P, R, err, td, pulse.sum(axis=1)))
    axes[0][0].set_ylabel("y [m]")
    fig.tight_layout()

    def update(k):
        artists = []
        for ghost, links, joints, tip, trail, dist, txt, warn, P, R, err, td, pulse in panels:
            ghost.set_data(R[k, :, 0], R[k, :, 1])
            for i, ln in enumerate(links):
                ln.set_data(P[k, i:i + 2, 0], P[k, i:i + 2, 1])
            joints.set_data(P[k, :3, 0], P[k, :3, 1])
            tip.set_data([P[k, 3, 0]], [P[k, 3, 1]])
            trail.set_data(P[: k + 1, 3, 0], P[: k + 1, 3, 1])
            if pulse[k] > 1.5:          # disturbance well above its typical level
                dist.set_visible(True)
                dist.xy = (P[k, 3, 0], P[k, 3, 1])
                dist.set_position((P[k, 3, 0] - 0.35, P[k, 3, 1] + 0.35))
            else:
                dist.set_visible(False)
            below = robot.floor is not None and P[k, 1:, 1].min() < robot.floor
            warn.set_text("below table!" if below else "")
            warn.set_visible(bool(below))
            txt.set_text(f"t        = {ft[k]:5.2f} s\n|e_tool| = {err[k] * 1e3:7.2f} mm\n"
                         f"|τ_d|    = {np.abs(td[k]).sum():5.1f} N·m")
            artists += [ghost, *links, joints, tip, trail, dist, txt, warn]
        return artists

    anim = FuncAnimation(fig, update, frames=len(ft), interval=1000 / fps, blit=False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".gif":
            anim.save(str(save_path), writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(str(save_path), writer="ffmpeg", fps=fps, dpi=dpi)
    return anim
