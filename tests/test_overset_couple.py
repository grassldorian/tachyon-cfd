"""Overset Phase 4: the two-grid coupled time loop preserves exact steady Euler
solutions — free-stream (uniform) and a parallel shear u(r) that varies across
the body/background overlap (so the wall-normal donor/receptor exchange is
genuinely exercised)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd import overset
from rocketcfd.overset_connect import cell_centers, HOLE, FRINGE
from rocketcfd.overset_couple import OversetCoupler

gam, rho0, p0 = 1.4, 1.2, 1.0e5


def run_case(amp, nsteps=120):
    xw = np.linspace(0.0, 1.0, 140)
    rw = np.full_like(xw, 0.2)
    g = overset.wall_normal_grid(xw, rw, n_eta=10, first_cell=8e-4,
                                 growth=1.14, n_xi=90)
    Xn, Rn = g["X"], g["R"]
    x0, y0, dx = 0.0, 0.11, 0.004
    ncx, ncy = int(1.0 / dx), int((0.204 - 0.11) / dx)
    cp = OversetCoupler(Xn, Rn, x0, y0, dx, ncx, ncy, gamma=gam,
                        blank_frac=0.55, fringe_layers=2)
    ng = cp.ng

    def shear(r):
        return 200.0 + amp * (0.2 - r) / 0.1

    def prim(U, s0, s1, u):
        U[0, s0, s1] = rho0
        U[1, s0, s1] = rho0 * u
        U[2, s0, s1] = 0.0
        U[3, s0, s1] = p0 / (gam - 1) + 0.5 * rho0 * u ** 2

    bcx, bcy = cell_centers(Xn, Rn)
    Nbi, Nbj = bcx.shape
    prim(cp.body.U, slice(ng, ng + Nbi), slice(ng, ng + Nbj), shear(bcy))
    ccy = y0 + (np.arange(ncy) + 0.5) * dx
    prim(cp.back.U, slice(ng, ng + ncx), slice(ng, ng + ncy),
         shear(ccy)[None, :])
    cp.body.set_bc(imin="periodic", jmin="slipwall", jmax="extrap")
    cp.back.set_bc(imin="periodic", jmin="extrap", jmax="extrap")

    rb, rk = cp.body.U.copy(), cp.back.U.copy()
    dt = min(cp.body.max_wave_dt(0.4), cp.back.max_wave_dt(0.4))
    for _ in range(nsteps):
        cp.step(dt)
    solve = cp.conn.blank == 0
    ks = (slice(ng, ng + ncx), slice(ng, ng + ncy))
    dk = np.max(np.abs(cp.back.U[1][ks] / cp.back.U[0][ks]
                       - rk[1][ks] / rk[0][ks])[solve])
    bs = (slice(ng, ng + Nbi), slice(ng, ng + Nbj - 2))
    db = np.max(np.abs(cp.body.U[1][bs] / cp.body.U[0][bs]
                       - rb[1][bs] / rb[0][bs]))
    return float(dk), float(db), cp


dk, db, _ = run_case(0.0)
assert dk == 0.0 and db == 0.0, f"free-stream not preserved: {dk}, {db}"
print(f"coupled free-stream: max |du| back {dk:.1e} body {db:.1e} OK")

dk, db, cp = run_case(100.0)
rel = max(dk, db) / 300.0
assert rel < 1e-4, f"steady shear drifted: {rel:.2e}"
print(f"coupled steady shear u(r): max |du| back {dk:.2e} body {db:.2e} m/s "
      f"(rel {rel:.2e}) OK")
print(f"overset: holes {(cp.conn.blank==HOLE).sum()}, "
      f"fringe {(cp.conn.blank==FRINGE).sum()}, receptors {len(cp.conn.br_idx)}")
print("overset-couple test OK")
