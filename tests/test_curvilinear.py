"""Overset Phase 2: the curvilinear finite-volume Euler solver is correct —
free-stream preservation (geometric conservation law) on the curved body-fitted
grid, and a rotated-grid Sod shock tube matching the exact Riemann solution."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd import engine_design as ed, overset
from rocketcfd.curvilinear import CurvilinearEuler

# ---- A: free-stream preservation on the curved near-body grid ---------------
x_mm, r_mm, _ = ed.build_contour(
    dict(chamber_l=0.09, chamber_d=0.11, throat_d=0.04, exit_d=0.09,
         nozzle_l=0.11), "Bell (Rao 80%)", fillet_mm=5, throat_r_mm=6)
g = overset.wall_normal_grid(x_mm * 1e-3, r_mm * 1e-3, n_eta=18,
                             first_cell=3e-5, growth=1.18, n_xi=80)
sim = CurvilinearEuler(g["X"], g["R"], gamma=1.4)
sim.set_uniform(1.2, 300.0, -50.0, 1.0e5)
ref = (1.2, 300.0, -50.0, 1.0e5)
for _ in range(150):
    sim.step(sim.max_wave_dt(0.4))
rho, u, v, p = sim.primitives()
i, j = sim.ii, sim.jj
dev = max(np.max(np.abs(rho[i, j] - ref[0])) / ref[0],
          np.max(np.abs(u[i, j] - ref[1])) / abs(ref[1]),
          np.max(np.abs(v[i, j] - ref[2])) / abs(ref[2]),
          np.max(np.abs(p[i, j] - ref[3])) / ref[3])
assert dev < 1e-11, f"free-stream not preserved: {dev:.2e}"
print(f"free-stream on curved grid: max rel dev {dev:.2e} OK")


# ---- B: rotated-grid Sod shock tube vs the exact Riemann solution -----------
def exact_sod(xi, G=1.4, WL=(1.0, 0.0, 1.0), WR=(0.125, 0.0, 0.1)):
    rL, uL, pL = WL
    rR, uR, pR = WR
    aL, aR = np.sqrt(G * pL / rL), np.sqrt(G * pR / rR)

    def fk(P, rk, pk, ak):
        if P > pk:
            A, B = 2.0 / ((G + 1) * rk), (G - 1) / (G + 1) * pk
            return (P - pk) * np.sqrt(A / (P + B))
        return 2 * ak / (G - 1) * ((P / pk) ** ((G - 1) / (2 * G)) - 1)

    def fp(P):
        return fk(P, rL, pL, aL) + fk(P, rR, pR, aR) + (uR - uL)
    ps = 0.5 * (pL + pR)
    for _ in range(80):
        e = 1e-6 * ps
        ps = max(1e-9, ps - fp(ps) / ((fp(ps + e) - fp(ps - e)) / (2 * e)))
    us = 0.5 * (uL + uR) + 0.5 * (fk(ps, rR, pR, aR) - fk(ps, rL, pL, aL))
    out = np.zeros((len(xi), 3))
    for m, s in enumerate(xi):
        if s < us:
            if ps > pL:
                rs = rL * ((ps / pL + (G - 1) / (G + 1))
                           / ((G - 1) / (G + 1) * ps / pL + 1))
                Sh = uL - aL * np.sqrt((G + 1) / (2 * G) * ps / pL
                                       + (G - 1) / (2 * G))
                out[m] = (rL, uL, pL) if s < Sh else (rs, us, ps)
            else:
                aLs = aL * (ps / pL) ** ((G - 1) / (2 * G))
                if s < uL - aL:
                    out[m] = rL, uL, pL
                elif s > us - aLs:
                    out[m] = rL * (ps / pL) ** (1 / G), us, ps
                else:
                    a = 2 / (G + 1) * (aL + (G - 1) / 2 * (uL - s))
                    out[m] = (rL * (a / aL) ** (2 / (G - 1)),
                              2 / (G + 1) * (aL + (G - 1) / 2 * uL + s),
                              pL * (a / aL) ** (2 * G / (G - 1)))
        else:
            if ps > pR:
                rs = rR * ((ps / pR + (G - 1) / (G + 1))
                           / ((G - 1) / (G + 1) * ps / pR + 1))
                Sh = uR + aR * np.sqrt((G + 1) / (2 * G) * ps / pR
                                       + (G - 1) / (2 * G))
                out[m] = (rR, uR, pR) if s > Sh else (rs, us, ps)
            else:
                aRs = aR * (ps / pR) ** ((G - 1) / (2 * G))
                if s > uR + aR:
                    out[m] = rR, uR, pR
                elif s < us + aRs:
                    out[m] = rR * (ps / pR) ** (1 / G), us, ps
                else:
                    a = 2 / (G + 1) * (aR - (G - 1) / 2 * (uR - s))
                    out[m] = (rR * (a / aR) ** (2 / (G - 1)),
                              2 / (G + 1) * (-aR + (G - 1) / 2 * uR + s),
                              pR * (a / aR) ** (2 * G / (G - 1)))
    return out


Nx, Ny, dx = 160, 6, 1.0 / 160
th = np.radians(30.0)
I, J = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), indexing="ij")
Xn = I * dx * np.cos(th) - J * dx * np.sin(th)
Rn = I * dx * np.sin(th) + J * dx * np.cos(th)
sod = CurvilinearEuler(Xn, Rn, gamma=1.4)
ng = sod.ng
xc = (np.arange(sod.ncx) - ng + 0.5) * dx
left = (xc < 0.5)[:, None]
sod.U[0] = np.where(left, 1.0, 0.125)
sod.U[1] = 0.0
sod.U[2] = 0.0
sod.U[3] = np.where(left, 1.0, 0.1) / (1.4 - 1)
sod.set_bc(imin="fixed", imax="fixed", jmin="periodic", jmax="periodic")
sod.set_bc(imin_state=(1.0, 0.0, 0.0, 1.0), imax_state=(0.125, 0.0, 0.0, 0.1))
sod.run(0.2, cfl=0.4)
rho, u, v, p = sod.primitives()
jc = sod.ncy // 2
xs = xc[ng:sod.ncx - ng]
ex = exact_sod((xs - 0.5) / 0.2)
l1_rho = float(np.mean(np.abs(rho[ng:sod.ncx - ng, jc] - ex[:, 0])))
l1_p = float(np.mean(np.abs(p[ng:sod.ncx - ng, jc] - ex[:, 2])))
assert l1_rho < 0.03 and l1_p < 0.03, (l1_rho, l1_p)
print(f"rotated-grid Sod vs exact: L1 rho {l1_rho:.4f}, L1 p {l1_p:.4f} OK")
print("curvilinear test OK")
