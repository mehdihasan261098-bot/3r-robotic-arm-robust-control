"""
robot3r — Robust control and energy-efficient trajectory planning of a 3R planar arm.

Modules
-------
robot          RobotArm3R: kinematics, Jacobian, redundancy-resolving IK, dynamics with friction
trajectory     Quintic / piecewise-quintic trajectories, Cartesian via-point helper
optimization   Minimum-energy and minimum-time trajectory optimisation (SLSQP)
controllers    PID, computed-torque (with friction compensation), sliding-mode control
disturbances   External torque disturbance models
simulation     solve_ivp simulator (continuous and sampled-data) and metrics
scenarios      Robustness scenarios and batch runner
visualization  Plots and animation
symbolic       SymPy derivation and verification of M, C, G
"""
from .controllers import (ComputedTorqueController, Controller, PIDController, Reference,
                          SlidingModeController, ZeroTorqueController, make_controller)
from .disturbances import Disturbance, Sinusoid, SmoothPulse, default_disturbance
from .optimization import OptimizedTrajectory, PlanningLimits, plan_trajectory, trajectory_report
from .robot import RobotArm3R, RobotParams3R, SingularityWarning, WorkspaceError
from .scenarios import Scenario, robustness_summary, run_study, standard_scenarios
from .simulation import SimulationResult, Simulator, format_metrics_table
from .trajectory import (PiecewiseQuinticTrajectory, QuinticTrajectory, check_trajectory_singularities,
                         joint_waypoints_from_cartesian)

__version__ = "1.0.0"
