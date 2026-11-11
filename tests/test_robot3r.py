"""
Verification tests for the 3R project.   Run:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot3r import (  # noqa: E402
    PlanningLimits, QuinticTrajectory, RobotArm3R, RobotParams3R, Simulator, WorkspaceError,
    ZeroTorqueController, default_disturbance, make_controller, plan_trajectory, trajectory_report,
)
from robot3r.optimization import OptimizedTrajectory  # noqa: E402

RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def robot():
    return RobotArm3R()


def _fd(f, x, h=1e-6):
    return np.column_stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


# ------------------------------------------------------------------ parameters
def test_constants_match_theory(robot):
    assert np.allclose([robot.a1, robot.a2, robot.a3], [2.184, 0.5506667, 0.06], atol=1e-6)
    assert np.allclose([robot.b1, robot.b2, robot.b3], [0.72, 0.15, 0.12])
    assert np.allclose([robot.gamma1, robot.gamma2, robot.gamma3], [17.658, 7.0632, 1.4715])
    assert np.all((0.1 <= robot.B) & (robot.B <= 0.3))


def test_payload_merging_reduces_to_base_model():
    a, b = RobotArm3R(), RobotArm3R(RobotParams3R(payload=0.0))
    assert np.isclose(a.a1, b.a1)
    p = RobotParams3R(payload=1.0)
    assert np.isclose(p.lc[2], 0.5) and np.isclose(p.m_eff[2], 1.5)


# ------------------------------------------------------------------ kinematics
def test_forward_kinematics_known_poses(robot):
    assert np.allclose(robot.forward_kinematics([0, 0, 0]), [2.4, 0])
    assert np.allclose(robot.forward_kinematics([np.pi / 2, 0, 0]), [0, 2.4])
    assert np.allclose(robot.forward_kinematics([0, np.pi / 2, -np.pi / 2]), [1.6, 0.8])
    assert robot.joint_positions(RNG.uniform(-3, 3, (5, 3))).shape == (5, 4, 2)


def test_jacobian_matches_finite_differences(robot):
    for _ in range(30):
        q = RNG.uniform(-np.pi, np.pi, 3)
        J_fd = _fd(lambda x: robot.forward_kinematics(x, return_orientation=True), q)
        assert np.allclose(robot.jacobian(q), J_fd, atol=1e-8)
        qd = RNG.uniform(-2, 2, 3)
        Jd_fd = (robot.jacobian(q + 1e-6 * qd) - robot.jacobian(q - 1e-6 * qd)) / 2e-6
        assert np.allclose(robot.jacobian_dot(q, qd), Jd_fd, atol=1e-7)


def test_singularity_formulas(robot):
    for _ in range(30):
        q = RNG.uniform(-np.pi, np.pi, 3)
        assert np.isclose(np.linalg.det(robot.jacobian(q)), robot.l1 * robot.l2 * np.sin(q[1]))
        Jp = robot.position_jacobian(q)
        assert np.isclose(robot.manipulability(q), np.sqrt(np.linalg.det(Jp @ Jp.T)))
    assert robot.is_singular([0.3, 0.0, 0.0])          # stretched: position-singular
    assert not robot.is_singular([0.3, 0.0, 1.0])      # theta2 = 0 alone is not position-singular...
    assert robot.is_singular([0.3, 0.0, 1.0], task="pose")  # ...but it is for the full pose task


def test_dls_bounded_near_singularity(robot):
    Jp = robot.position_jacobian([0.2, 1e-7, 1e-7])
    dq = robot.dls_pinv(Jp) @ np.array([0.1, 0.0])
    assert np.all(np.isfinite(dq)) and np.linalg.norm(dq) < 50


def test_geometric_ik_with_given_orientation(robot):
    for _ in range(200):
        q = RNG.uniform([-0.5, -2.5, -2.5], [3.6, 2.5, 2.5])
        p, phi = robot.forward_kinematics(q), q.sum()
        for elbow in ("up", "down"):
            try:
                qs = robot.ik_given_orientation(p, phi, elbow)
            except WorkspaceError:
                continue
            assert np.allclose(robot.forward_kinematics(qs, return_orientation=True)[:2], p, atol=1e-9)
            assert np.isclose(np.cos(qs.sum() - phi), 1.0)


def test_redundancy_resolution_is_optimal_and_admissible(robot):
    for p in [(1.6, 0.6), (-1.2, 0.9), (0.5, 1.5), (2.0, 0.4)]:
        q = robot.inverse_kinematics(p)
        assert np.allclose(robot.forward_kinematics(q), p, atol=1e-9)
        assert robot.admissible(q)
        phis, sols = robot.feasible_orientations(p, "up")
        w_grid = max(robot.manipulability(s) for s in sols)
        assert robot.manipulability(q) >= w_grid - 0.02     # near the best admissible grid point


def test_ik_errors(robot):
    with pytest.raises(WorkspaceError):
        robot.inverse_kinematics([3.0, 0.0])


def test_numerical_ik_converges_and_maximises_manipulability(robot):
    p = (1.6, 0.6)
    q_geo = robot.inverse_kinematics(p)
    q_num, ok = robot.ik_numerical(p, q_geo + np.array([0.3, -0.2, 0.25]))
    assert ok and np.allclose(robot.forward_kinematics(q_num), p, atol=1e-9)
    assert np.isclose(robot.manipulability(q_num), robot.manipulability(q_geo), atol=1e-3)


# -------------------------------------------------------------------- dynamics
def test_mass_matrix_spd(robot):
    M = robot.mass_matrix(RNG.uniform(-np.pi, np.pi, (2000, 3)))
    assert np.allclose(M, np.swapaxes(M, -1, -2))
    assert np.linalg.eigvalsh(M).min() > 0


def test_skew_symmetry(robot):
    for _ in range(30):
        q, qd = RNG.uniform(-np.pi, np.pi, 3), RNG.uniform(-3, 3, 3)
        Mdot = (robot.mass_matrix(q + 1e-6 * qd) - robot.mass_matrix(q - 1e-6 * qd)) / 2e-6
        N = Mdot - 2 * robot.coriolis_matrix(q, qd)
        assert np.allclose(N, -N.T, atol=1e-7)


def test_coriolis_vector_matches_matrix(robot):
    q, qd = RNG.uniform(-np.pi, np.pi, (50, 3)), RNG.uniform(-3, 3, (50, 3))
    assert np.allclose(robot.coriolis_vector(q, qd), np.einsum("nij,nj->ni", robot.coriolis_matrix(q, qd), qd))


def test_gravity_is_gradient_of_potential(robot):
    for _ in range(30):
        q = RNG.uniform(-np.pi, np.pi, 3)
        assert np.allclose(robot.gravity_vector(q), _fd(lambda x: np.atleast_1d(robot.potential_energy(x)), q)[0],
                           atol=1e-7)


def test_inverse_forward_dynamics_consistent(robot):
    for _ in range(30):
        q, qd, qdd = (RNG.uniform(-3, 3, 3) for _ in range(3))
        tau = robot.inverse_dynamics(q, qd, qdd)
        assert np.allclose(robot.forward_dynamics(q, qd, tau), qdd)


def test_symbolic_derivation_matches_closed_form():
    pytest.importorskip("sympy")
    from robot3r.symbolic import verify_against_numeric

    err = verify_against_numeric(RobotArm3R(RobotParams3R(payload=0.7)), n_samples=20)
    assert max(err.values()) < 1e-12


def test_energy_balance_with_friction(robot):
    res = Simulator(robot, ZeroTorqueController(), rtol=1e-11, atol=1e-12, max_step=0.002).run(
        2.0, q0=[0.5, 0.8, -0.6], dt=0.0005)
    P = np.sum(res.qd * robot.friction_torque(res.qd), axis=1)
    dissipated = np.concatenate([[0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(res.t))])
    assert np.max(np.abs(res.energy - res.energy[0] + dissipated)) < 1e-4
    assert dissipated[-1] > 1.0


# ---------------------------------------------------------------- trajectories
def test_quintic_boundaries():
    tr = QuinticTrajectory([0, 1, 2], [1, -1, 0], 0.5, 2.5)
    for t, q in ((0.5, [0, 1, 2]), (2.5, [1, -1, 0])):
        qq, qd, qdd = tr.evaluate(t)
        assert np.allclose(qq, q) and np.allclose(qd, 0) and np.allclose(qdd, 0)


def test_optimized_trajectory_keeps_boundary_conditions():
    C = RNG.normal(size=(5, 3))
    tr = OptimizedTrajectory([0, 0.5, -0.5], [1, -1, 0.3], 0.0, 2.0, C)
    for t, q in ((0.0, [0, 0.5, -0.5]), (2.0, [1, -1, 0.3])):
        qq, qd, qdd = tr.evaluate(t)
        assert np.allclose(qq, q) and np.allclose(qd, 0, atol=1e-12) and np.allclose(qdd, 0, atol=1e-12)
    # derivatives consistent with finite differences
    t = np.linspace(0.2, 1.8, 9)
    q1, _, _ = tr.evaluate(t + 1e-6)
    q0, _, _ = tr.evaluate(t - 1e-6)
    assert np.allclose((q1 - q0) / 2e-6, tr.evaluate(t)[1], atol=1e-6)


def test_min_energy_plan_beats_quintic_and_respects_limits(robot):
    qA, qB = robot.inverse_kinematics((1.6, 0.6)), robot.inverse_kinematics((-1.2, 0.9))
    lim = PlanningLimits()
    res = plan_trajectory(robot, qA, qB, 3.0, objective="energy", n_basis=3, maxiter=400)
    rq = trajectory_report(robot, QuinticTrajectory(qA, qB, 0, 3.0))
    assert res.success
    assert res.report["energy_tau2"] < 0.8 * rq["energy_tau2"]
    assert res.report["torque_limit_ratio"] <= 1.0 + 1e-3
    assert res.report["peak_qd"] <= lim.qd_max + 0.05
    assert res.report["min_height_m"] >= -0.01


def test_min_time_plan_is_faster_and_torque_limited(robot):
    qA, qB = robot.inverse_kinematics((1.6, 0.6)), robot.inverse_kinematics((-1.2, 0.9))
    res = plan_trajectory(robot, qA, qB, 3.0, objective="time", n_basis=3)
    assert res.success and res.report["duration_s"] < 1.5
    assert 0.95 < max(res.report["torque_limit_ratio"], res.report["peak_qd"] / 3.0) < 1.02  # a limit is active


# ----------------------------------------------------------------- controllers
@pytest.fixture(scope="module")
def short_traj(robot):
    qA, qB = robot.inverse_kinematics((1.6, 0.6)), robot.inverse_kinematics((0.4, 1.6))
    return QuinticTrajectory(qA, qB, 0.0, 1.5)


@pytest.mark.parametrize("kind", ["ctc", "smc"])
def test_model_based_controllers_track_exactly_with_perfect_model(robot, short_traj, kind):
    res = Simulator(robot, make_controller(kind, robot), short_traj).run(2.0)
    assert res.metrics()["rmse_ee"] < 1e-6


def test_ctc_error_dynamics_are_linear(robot, short_traj):
    wn = 20.0
    e0 = np.array([0.02, -0.03, 0.04])
    hold = QuinticTrajectory(short_traj.q0, short_traj.q0, 0, 1)
    res = Simulator(robot, make_controller("ctc", robot, wn=wn), hold).run(1.0, q0=hold.q0 - e0)
    expected = np.outer((1 + wn * res.t) * np.exp(-wn * res.t), e0)
    assert np.allclose(res.e, expected, atol=1e-6)


def test_smc_beats_ctc_and_pid_under_payload_and_disturbance(robot, short_traj):
    plant = RobotArm3R(RobotParams3R(payload=1.0))
    d = default_disturbance()
    rmse = {k: Simulator(plant, make_controller(k, robot), short_traj, disturbance=d).run(2.0).metrics()["rmse_ee"]
            for k in ("pid", "ctc", "smc")}
    assert rmse["smc"] < 0.5 * min(rmse["pid"], rmse["ctc"])


def test_sign_switching_chatters_more_than_boundary_layer(robot):
    hold_q = robot.inverse_kinematics((1.2, 0.8))
    tr = QuinticTrajectory(hold_q, hold_q, 0.0, 0.1)
    plant = RobotArm3R(RobotParams3R(payload=0.5))
    tv = {}
    for sw in ("sign", "sat"):
        r = Simulator(plant, make_controller("smc", robot, switching=sw), tr, control_period=1e-3).run(0.3)
        tv[sw] = r.metrics()["tv_rate"]
    assert tv["sign"] > 20 * tv["sat"]


def test_pid_integral_removes_static_error(robot):
    q = robot.inverse_kinematics((1.5, 0.8))
    tr = QuinticTrajectory(q, q, 0, 0.1)
    res = Simulator(robot, make_controller("pid", robot), tr).run(10.0)
    assert np.max(np.abs(res.e[-1])) < 2e-3
