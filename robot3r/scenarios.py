"""
Robustness study: scenario definitions and batch runner (docs/THEORY.md §8.1).

Every controller is built on the *nominal* model; each scenario changes only the
simulated plant and/or adds an external disturbance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .controllers import make_controller
from .disturbances import Disturbance, default_disturbance
from .robot import RobotArm3R, RobotParams3R
from .simulation import SimulationResult, Simulator


@dataclass
class Scenario:
    """A plant variant plus an optional disturbance."""

    name: str
    params: RobotParams3R
    disturbance: Optional[Disturbance] = None
    description: str = ""

    def plant(self) -> RobotArm3R:
        return RobotArm3R(self.params)


def standard_scenarios(nominal: Optional[RobotParams3R] = None) -> List[Scenario]:
    """The five scenarios required by the project brief."""
    p = nominal or RobotParams3R()
    d = default_disturbance()
    return [
        Scenario("Nominal", p, None, "plant = model"),
        Scenario("Payload +0.5 kg", p.with_payload(0.5), None, "0.5 kg point mass at the tool"),
        Scenario("Payload +1.0 kg", p.with_payload(1.0), None, "1.0 kg point mass at the tool"),
        Scenario("Disturbance", p, d, "sinusoids + torque pulses"),
        Scenario("Combined", p.scaled(mass_scale=1.10, friction_scale=1.5).with_payload(1.0), d,
                 "1 kg payload, masses +10 %, friction +50 %, disturbance"),
    ]


def run_study(trajectory, controllers: Sequence[str], scenarios: Sequence[Scenario], *,
              nominal: Optional[RobotArm3R] = None, t_final: Optional[float] = None,
              control_period: Optional[float] = None, controller_overrides: Optional[Dict[str, dict]] = None,
              progress: Optional[Callable[[str], None]] = None) -> Dict[str, Dict[str, SimulationResult]]:
    """Simulate every (controller, scenario) pair.

    Returns
    -------
    dict
        ``results[controller_name][scenario_name] -> SimulationResult``.
    """
    nominal = nominal or RobotArm3R()
    overrides = controller_overrides or {}
    out: Dict[str, Dict[str, SimulationResult]] = {}
    for kind in controllers:
        for sc in scenarios:
            ctrl = make_controller(kind, nominal, **overrides.get(kind, {}))
            sim = Simulator(sc.plant(), ctrl, trajectory, disturbance=sc.disturbance,
                            control_period=control_period, scenario=sc.name)
            res = sim.run(t_final)
            out.setdefault(ctrl.name, {})[sc.name] = res
            if progress:
                m = res.metrics()
                progress(f"  {ctrl.name:<5s} | {sc.name:<16s} | EE RMSE {m['rmse_ee'] * 1e3:9.3f} mm"
                         f" | peak tau {np.array2string(m['peak_torque'], precision=1)}")
    return out


def robustness_summary(results: Dict[str, Dict[str, SimulationResult]]) -> Dict[str, Dict[str, float]]:
    """Worst-case / mean tool RMSE over scenarios and growth from nominal to combined."""
    summary = {}
    for ctrl, by_sc in results.items():
        rmse = {sc: r.metrics()["rmse_ee"] for sc, r in by_sc.items()}
        vals = np.array(list(rmse.values()))
        summary[ctrl] = {
            "worst_rmse_ee": float(vals.max()),
            "mean_rmse_ee": float(vals.mean()),
            "nominal_rmse_ee": rmse.get("Nominal", np.nan),
            "combined_rmse_ee": rmse.get("Combined", np.nan),
            "worst_scenario": max(rmse, key=rmse.get),
        }
    return summary
