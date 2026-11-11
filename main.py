#!/usr/bin/env python
"""
Run the studies of the 3R project.

    python main.py --case all --no-show            # everything (figures + metrics.md in ./results)
    python main.py --case plan                      # quintic vs minimum-energy vs minimum-time
    python main.py --case track --controller smc --scenario Combined --animate
    python main.py --case robust --animate          # 3 controllers x 5 scenarios
    python main.py --case chattering                # SMC sign / sat / tanh at 1 kHz

Cases
-----
validate    SymPy check of M, C, G and passive energy balance with friction
ik          Redundancy resolution at the goal point (geometric vs numerical IK)
plan        Quintic, minimum-energy and minimum-time trajectories
track       One controller on one scenario
compare     PID vs CTC vs SMC on the nominal and combined scenarios
robust      Full robustness matrix (3 controllers x 5 scenarios)
chattering  Sampled-data sliding-mode control with sign / sat / tanh switching
all         Everything above
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- task definition ---------------------------------------------------------
TASK_START = (1.6, 0.6)      # tool position at t = 0 [m]
TASK_GOAL = (-1.2, 0.9)      # tool position at the end of the move [m]
MOVE_TIME = 3.0              # [s]
HOLD_TIME = 1.0              # regulation after the move [s]
CONTROL_PERIOD = 1e-3        # digital controller sample time for the chattering study [s]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="robust",
                   choices=["validate", "ik", "plan", "track", "compare", "robust", "chattering", "all"])
    p.add_argument("--controller", default="smc", choices=["pid", "ctc", "smc"], help="for --case track")
    p.add_argument("--scenario", default="Combined",
                   help="for --case track: Nominal, 'Payload +0.5 kg', 'Payload +1.0 kg', Disturbance, Combined")
    p.add_argument("--trajectory", default="energy", choices=["energy", "quintic", "time"],
                   help="reference used by the tracking studies")
    p.add_argument("--animate", action="store_true", help="also write GIF animations")
    p.add_argument("--save-dir", default="results")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


# =========================================================================== #
# Shared helpers
# =========================================================================== #
class Output:
    def __init__(self, d: str):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sections, self.anims = [], []

    def fig(self, fig, name):
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=130)
        print(f"  saved {path}")

    def md(self, title, text):
        self.sections.append(f"## {title}\n\n{text}\n")

    def write(self):
        if self.sections:
            path = self.dir / "metrics.md"
            path.write_text("# 3R arm — simulation results\n\n" + "\n".join(self.sections), encoding="utf-8")
            print(f"\nmetrics written to {path}")


_PLANS = {}


def task_endpoints(robot):
    qA = robot.inverse_kinematics(TASK_START)
    qB = robot.inverse_kinematics(TASK_GOAL)
    return qA, qB


def plans(robot):
    """Quintic, minimum-energy and minimum-time trajectories (cached)."""
    if not _PLANS:
        from robot3r import QuinticTrajectory, plan_trajectory, trajectory_report

        qA, qB = task_endpoints(robot)
        quin = QuinticTrajectory(qA, qB, 0.0, MOVE_TIME)
        _PLANS["Quintic"] = (quin, trajectory_report(robot, quin), None)
        for obj, name in (("energy", "Min-energy"), ("time", "Min-time")):
            t0 = time.time()
            res = plan_trajectory(robot, qA, qB, MOVE_TIME, objective=obj)
            print(f"  {name:<10s}: success={res.success}, {res.n_iter} SLSQP iterations, {time.time() - t0:.1f} s")
            _PLANS[name] = (res.trajectory, res.report, res)
    return _PLANS


def reference(robot, kind):
    key = {"energy": "Min-energy", "quintic": "Quintic", "time": "Min-time"}[kind]
    return plans(robot)[key][0]


# =========================================================================== #
# Cases
# =========================================================================== #
def case_validate(args, out):
    from robot3r import RobotArm3R, Simulator, ZeroTorqueController
    from robot3r.symbolic import verify_against_numeric
    from robot3r.visualization import plot_energy_balance

    print("\n[validate] symbolic dynamics and energy balance")
    robot = RobotArm3R()
    t0 = time.time()
    err = verify_against_numeric(robot, n_samples=100)
    print(f"  SymPy vs closed form (100 random states): max |ΔM| = {err['M']:.1e}, |ΔC| = {err['C']:.1e}, "
          f"|ΔG| = {err['G']:.1e}  ({time.time() - t0:.1f} s)")
    sim = Simulator(robot, ZeroTorqueController(), rtol=1e-11, atol=1e-12, max_step=0.002)
    res = sim.run(5.0, q0=[0.5, 0.8, -0.6], dt=0.0005)
    power = np.sum(res.qd * robot.friction_torque(res.qd), axis=1)
    dissipated = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(res.t))])
    residual = res.energy - res.energy[0] + dissipated
    print(f"  passive 5 s swing: energy dropped {res.energy[0] - res.energy[-1]:.4f} J, "
          f"friction dissipated {dissipated[-1]:.4f} J, max balance residual {np.max(np.abs(residual)):.2e} J")
    out.fig(plot_energy_balance(res, dissipated, residual), "validate_energy_balance")
    out.md("Model validation",
           f"- SymPy vs closed form: max |ΔM| = {err['M']:.1e}, |ΔC| = {err['C']:.1e}, |ΔG| = {err['G']:.1e}\n"
           f"- Passive swing with friction: energy loss {res.energy[0] - res.energy[-1]:.4f} J, "
           f"friction dissipation {dissipated[-1]:.4f} J, max residual {np.max(np.abs(residual)):.2e} J")


def case_ik(args, out):
    from robot3r import RobotArm3R
    from robot3r.visualization import plot_ik_redundancy

    print("\n[ik] redundancy resolution")
    robot = RobotArm3R()
    rows = ["| Target [m] | Branch | Selected q [deg] | φ [deg] | w | Numerical IK q [deg] | w (numerical) | Numerical admissible? |",
            "|---|---|---|---|---|---|---|---|"]
    for p in (TASK_START, TASK_GOAL):
        q = robot.inverse_kinematics(p)
        qn, ok = robot.ik_numerical(p, q + np.array([0.3, -0.2, 0.25]))
        print(f"  {p}: geometric q = {np.rad2deg(q).round(2)} (w = {robot.manipulability(q):.4f}), "
              f"numerical q = {np.rad2deg(qn).round(2)} (w = {robot.manipulability(qn):.4f}, converged={ok}, "
              f"admissible={robot.admissible(qn)})")
        rows.append(f"| {p} | up | {np.rad2deg(q).round(1)} | {np.rad2deg(q.sum()):.1f} | {robot.manipulability(q):.4f} "
                    f"| {np.rad2deg(qn).round(1)} | {robot.manipulability(qn):.4f} | {robot.admissible(qn)} |")
    q = robot.inverse_kinematics(TASK_GOAL)
    qn, _ = robot.ik_numerical(TASK_GOAL, q + np.array([0.3, -0.2, 0.25]))
    out.fig(plot_ik_redundancy(robot, TASK_GOAL, q, qn), "ik_redundancy")
    out.md("Inverse kinematics", "\n".join(rows))


def case_plan(args, out):
    from robot3r import PlanningLimits, RobotArm3R
    from robot3r.visualization import animate, plot_trajectory_comparison

    print("\n[plan] quintic vs optimisation-based trajectories")
    robot = RobotArm3R()
    P = plans(robot)
    rows = ["| Trajectory | T [s] | ∫τᵀτ dt | Mechanical work [J] | Peak τ1/τ2/τ3 [N·m] | Peak speed [rad/s] "
            "| Min height [m] | Min manipulability |", "|---|---|---|---|---|---|---|---|"]
    for name, (tr, rep, _) in P.items():
        print(f"  {name:<10s}: T = {rep['duration_s']:.3f} s, ∫τᵀτ = {rep['energy_tau2']:.1f}, "
              f"E_mech = {rep['energy_mech_J']:.2f} J, peak τ = ({rep['peak_tau_1']:.1f}, {rep['peak_tau_2']:.1f}, "
              f"{rep['peak_tau_3']:.2f}) N·m")
        rows.append(f"| {name} | {rep['duration_s']:.3f} | {rep['energy_tau2']:.1f} | {rep['energy_mech_J']:.2f} "
                    f"| {rep['peak_tau_1']:.1f}/{rep['peak_tau_2']:.1f}/{rep['peak_tau_3']:.2f} | {rep['peak_qd']:.2f} "
                    f"| {rep['min_height_m']:.3f} | {rep['min_manipulability']:.3f} |")
    out.md("Trajectory planning (nominal inverse dynamics)", "\n".join(rows))
    fig = plot_trajectory_comparison(robot, {k: v[0] for k, v in P.items()}, {k: v[1] for k, v in P.items()},
                                     tau_max=PlanningLimits().tau_max)
    out.fig(fig, "plan_comparison")


def case_track(args, out):
    from robot3r import RobotArm3R, Simulator, make_controller, standard_scenarios
    from robot3r.visualization import animate, plot_tracking

    robot = RobotArm3R()
    sc = {s.name.lower(): s for s in standard_scenarios()}[args.scenario.lower()]
    print(f"\n[track] {args.controller.upper()} on '{sc.name}', {args.trajectory} trajectory")
    traj = reference(robot, args.trajectory)
    ctrl = make_controller(args.controller, robot)
    res = Simulator(sc.plant(), ctrl, traj, disturbance=sc.disturbance, scenario=sc.name).run(traj.tf + HOLD_TIME)
    m = res.metrics()
    print(f"  tool RMSE {m['rmse_ee'] * 1e3:.3f} mm, max {m['max_ee_error'] * 1e3:.3f} mm, "
          f"peak τ {m['peak_torque'].round(2)} N·m, E_mech {m['energy_mech']:.2f} J")
    tag = f"track_{args.controller}_{sc.name.split()[0].lower()}"
    out.fig(plot_tracking(res), tag)
    if args.animate:
        out.anims.append(animate(res, save_path=out.dir / f"{tag}.gif"))


def case_compare(args, out):
    from robot3r import RobotArm3R, format_metrics_table, run_study, standard_scenarios
    from robot3r.visualization import plot_controller_comparison

    print("\n[compare] PID vs CTC vs SMC")
    robot = RobotArm3R()
    traj = reference(robot, args.trajectory)
    scen = [s for s in standard_scenarios() if s.name in ("Nominal", "Combined")]
    res = run_study(traj, ["pid", "ctc", "smc"], scen, nominal=robot, t_final=traj.tf + HOLD_TIME, progress=print)
    for s in scen:
        runs = [res[c][s.name] for c in res]
        out.fig(plot_controller_comparison(runs, f"Controller comparison — {s.name} scenario"),
                f"compare_{s.name.lower()}")
        out.md(f"Controller comparison — {s.name}", format_metrics_table(runs))


def case_robust(args, out):
    from robot3r import RobotArm3R, format_metrics_table, robustness_summary, run_study, standard_scenarios
    from robot3r.visualization import animate, plot_controller_comparison, plot_robustness, plot_tracking

    print("\n[robust] 3 controllers x 5 scenarios")
    robot = RobotArm3R()
    traj = reference(robot, args.trajectory)
    scen = standard_scenarios()
    t0 = time.time()
    res = run_study(traj, ["pid", "ctc", "smc"], scen, nominal=robot, t_final=traj.tf + HOLD_TIME, progress=print)
    print(f"  ({time.time() - t0:.0f} s)")
    out.fig(plot_robustness(res), "robust_matrix")
    for s in scen:
        runs = [res[c][s.name] for c in res]
        out.md(f"Robustness — {s.name} ({s.description})", format_metrics_table(runs))
        if s.name != "Nominal":
            out.fig(plot_controller_comparison(runs, f"Controller comparison — {s.name}"),
                    "robust_" + s.name.lower().replace(" ", "_").replace("+", "").replace(".", ""))
    summ = robustness_summary(res)
    rows = ["| Controller | Nominal [mm] | Combined [mm] | Mean over scenarios [mm] | Worst case [mm] | Worst scenario |",
            "|---|---|---|---|---|---|"]
    for c, s in summ.items():
        rows.append(f"| {c} | {s['nominal_rmse_ee'] * 1e3:.3g} | {s['combined_rmse_ee'] * 1e3:.3g} "
                    f"| {s['mean_rmse_ee'] * 1e3:.3g} | {s['worst_rmse_ee'] * 1e3:.3g} | {s['worst_scenario']} |")
    out.md("Robustness summary (tool-tip RMSE)", "\n".join(rows))
    print("\n".join(rows))
    out.fig(plot_tracking(res["SMC"]["Combined"]), "track_smc_combined")
    if args.animate:
        runs = [res[c]["Combined"] for c in res]
        out.anims.append(animate(runs, save_path=out.dir / "robust_combined.gif"))
        out.anims.append(animate([res[c]["Nominal"] for c in res], save_path=out.dir / "nominal.gif"))


def case_chattering(args, out):
    from robot3r import RobotArm3R, Simulator, format_metrics_table, make_controller, standard_scenarios
    from robot3r.visualization import plot_chattering

    print(f"\n[chattering] sampled-data SMC at {1 / CONTROL_PERIOD:.0f} Hz on the Combined scenario")
    robot = RobotArm3R()
    traj = reference(robot, args.trajectory)
    sc = {s.name: s for s in standard_scenarios()}["Combined"]
    runs = []
    for sw in ("sign", "sat", "tanh"):
        t0 = time.time()
        ctrl = make_controller("smc", robot, switching=sw)
        r = Simulator(sc.plant(), ctrl, traj, disturbance=sc.disturbance, control_period=CONTROL_PERIOD,
                      scenario=sc.name).run(traj.tf + HOLD_TIME)
        m = r.metrics()
        print(f"  {ctrl.name:<9s}: TV {m['tv_rate']:8.0f} N·m/s, HF {100 * m['hf_fraction']:6.2f} %, "
              f"tool RMSE {m['rmse_ee'] * 1e3:.3f} mm ({time.time() - t0:.0f} s)")
        runs.append(r)
    out.fig(plot_chattering(runs), "chattering_smc")
    out.md("Chattering study (sampled-data 1 kHz, Combined scenario)", format_metrics_table(runs))


def main(argv=None):
    args = parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.simplefilter("default")
    out = Output(args.save_dir)
    cases = {"validate": case_validate, "ik": case_ik, "plan": case_plan, "track": case_track,
             "compare": case_compare, "robust": case_robust, "chattering": case_chattering}
    selected = [c for c in cases if c not in ("track", "compare")] if args.case == "all" else [args.case]
    for name in selected:
        cases[name](args, out)
    out.write()
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
