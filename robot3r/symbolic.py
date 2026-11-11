"""
Independent SymPy derivation of the 3R dynamics (docs/THEORY.md Part 5).

Builds the Lagrangian from COM positions, applies the Euler-Lagrange equations,
extracts M (acceleration coefficients), G (dV/dq) and C (Christoffel symbols),
checks that M q'' + C q' + G reproduces the Euler-Lagrange expression, and
compares everything with the closed forms in :mod:`robot3r.robot`.

Run:  python -m robot3r.symbolic
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import sympy as sp

from .robot import RobotArm3R


@dataclass
class SymbolicModel:
    M: sp.Matrix
    C: sp.Matrix
    G: sp.Matrix
    params: tuple
    M_func: Callable
    C_func: Callable
    G_func: Callable


def derive_dynamics(check_residual: bool = True) -> SymbolicModel:
    t = sp.symbols("t", real=True)
    l1, l2, l3, lc1, lc2, lc3 = sp.symbols("l1 l2 l3 lc1 lc2 lc3", positive=True)
    m1, m2, m3, I1, I2, I3, g = sp.symbols("m1 m2 m3 I1 I2 I3 g", positive=True)
    th = [sp.Function(f"theta{i}")(t) for i in (1, 2, 3)]
    a1, a12, a123 = th[0], th[0] + th[1], th[0] + th[1] + th[2]

    def e(a):
        return sp.Matrix([sp.cos(a), sp.sin(a)])

    o1 = l1 * e(a1)
    o2 = o1 + l2 * e(a12)
    pc = [lc1 * e(a1), o1 + lc2 * e(a12), o2 + lc3 * e(a123)]
    w = [sp.diff(a1, t), sp.diff(a12, t), sp.diff(a123, t)]
    m, I = [m1, m2, m3], [I1, I2, I3]
    T = sum(sp.Rational(1, 2) * (m[i] * pc[i].diff(t).dot(pc[i].diff(t)) + I[i] * w[i] ** 2) for i in range(3))
    V = g * sum(m[i] * pc[i][1] for i in range(3))
    L = T - V

    q = sp.symbols("q1:4", real=True)
    dq = sp.symbols("dq1:4", real=True)
    ddq = sp.symbols("ddq1:4", real=True)
    subs = {}
    for i in range(3):
        subs[sp.diff(th[i], t, 2)] = ddq[i]
    for i in range(3):
        subs[sp.diff(th[i], t)] = dq[i]
    for i in range(3):
        subs[th[i]] = q[i]

    eom = sp.Matrix([sp.diff(sp.diff(L, sp.diff(th[i], t)), t) - sp.diff(L, th[i]) for i in range(3)])
    eom = eom.subs(subs)
    Vq = V.subs(subs)

    M = sp.Matrix(3, 3, lambda i, j: sp.simplify(sp.diff(eom[i], ddq[j])))
    G = sp.Matrix([sp.simplify(sp.diff(Vq, qi)) for qi in q])
    C = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            C[i, j] = sp.simplify(sum(
                sp.Rational(1, 2) * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j]) - sp.diff(M[j, k], q[i])) * dq[k]
                for k in range(3)))
    if check_residual:
        res = sp.simplify(sp.expand(eom - (M * sp.Matrix(ddq) + C * sp.Matrix(dq) + G)))
        assert res == sp.zeros(3, 1), "Euler-Lagrange equations disagree with M, C, G"

    params = (l1, l2, l3, lc1, lc2, lc3, m1, m2, m3, I1, I2, I3, g)
    return SymbolicModel(
        M=M, C=C, G=G, params=params,
        M_func=sp.lambdify((q, params), M, "numpy"),
        C_func=sp.lambdify((q, dq, params), C, "numpy"),
        G_func=sp.lambdify((q, params), G, "numpy"),
    )


def verify_against_numeric(robot: RobotArm3R, model: SymbolicModel | None = None,
                           n_samples: int = 100, seed: int = 0) -> Dict[str, float]:
    """Max |symbolic - closed form| for M, C, G over random states."""
    model = model or derive_dynamics()
    p = robot.params
    pv = (*p.l, *p.lc, *p.m_eff, *p.I, p.g)
    rng = np.random.default_rng(seed)
    err = {"M": 0.0, "C": 0.0, "G": 0.0}
    for _ in range(n_samples):
        q, qd = rng.uniform(-np.pi, np.pi, 3), rng.uniform(-4, 4, 3)
        err["M"] = max(err["M"], np.abs(np.array(model.M_func(q, pv), float) - robot.mass_matrix(q)).max())
        err["C"] = max(err["C"], np.abs(np.array(model.C_func(q, qd, pv), float) - robot.coriolis_matrix(q, qd)).max())
        err["G"] = max(err["G"], np.abs(np.array(model.G_func(q, pv), float).ravel() - robot.gravity_vector(q)).max())
    return err


if __name__ == "__main__":  # pragma: no cover
    sm = derive_dynamics()
    for name in ("M", "C", "G"):
        print(f"\n{name} =")
        sp.pprint(getattr(sm, name))
    print("\nmax |symbolic - closed form| (nominal):", verify_against_numeric(RobotArm3R()))
    from .robot import RobotParams3R
    print("max |symbolic - closed form| (1 kg payload):",
          verify_against_numeric(RobotArm3R(RobotParams3R(payload=1.0)), sm))
