"""Menter k-omega SST on the curvilinear solver.

Validated component-wise against exact results:
  * decay of homogeneous turbulence (no production) has the analytic solution
    w = w0/(1+beta w0 t), k = k0 (1+beta w0 t)^(-betastar/beta);
  * far from walls F1 -> 0 (k-epsilon branch) and mu_t reduces to rho k / w;
  * the strain invariant of a pure shear u=S*y is S^2 (the production P_k =
    mu_t S^2 is built from it);
  * the smooth-wall omega BC is Menter's 60 nu / (beta1 y1^2).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.curvilinear import CurvilinearEuler, _BSTAR, _BET1, _BET2, _A1

rho0, p0, gam, R = 1.2, 1.0e5, 1.4, 287.0
mu = 1.8e-5


def uniform_box(nx=6, ny=6, dx=0.01):
    I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    s = CurvilinearEuler(I * dx, J * dx, gamma=gam)
    s.set_viscous(mu, Rgas=R)
    s.U[0] = rho0
    s.U[1:3] = 0.0
    s.U[3] = p0 / (gam - 1)
    return s


# ---- 1) decay of homogeneous turbulence vs the analytic ODE -----------------
s = uniform_box()
ng = s.ng
s.set_bc(imin="periodic", jmin="periodic")
s.set_turbulence(np.full((s.ncx, s.ncy), 1.0e6), k0=1.0, w0=100.0)  # no walls
k0, w0, dt, nstep = 1.0, 100.0, 1e-4, 1000
for _ in range(nstep):
    s.step(dt)
t = nstep * dt
_, k, w = s._turb_prim()
b = _BET2                                        # F1 -> 0 far from walls
w_ex = w0 / (1 + b * w0 * t)
k_ex = k0 * (1 + b * w0 * t) ** (-_BSTAR / b)
ew = abs(float(w[ng, ng]) - w_ex) / w_ex
ek = abs(float(k[ng, ng]) - k_ex) / k_ex
assert ew < 5e-3 and ek < 5e-3, (ew, ek)
assert float(s._F1[ng, ng]) < 1e-6, "F1 should vanish far from walls"
print(f"SST decay vs exact: omega {ew*100:.3f}%, k {ek*100:.3f}%, F1~0 OK")

# mu_t must reduce to rho k / w when the strain vanishes
mut_ex = rho0 * float(k[ng, ng]) / float(w[ng, ng])
assert abs(float(s.mu_t[ng, ng]) - mut_ex) / mut_ex < 1e-6
print(f"mu_t = rho k / w at zero strain OK ({s.mu_t[ng, ng]:.5g})")

# ---- 2) strain invariant of a pure shear u = S*y is S^2 ---------------------
S = 250.0
s2 = uniform_box(nx=6, ny=8)
ng = s2.ng
yc = (np.arange(s2.ncy) - ng + 0.5) * 0.01
u = (S * yc)[None, :] * np.ones((s2.ncx, 1))
s2.U[1] = rho0 * u
s2.U[3] = p0 / (gam - 1) + 0.5 * rho0 * u ** 2
s2.set_bc(imin="periodic", jmin="extrap", jmax="extrap")
s2.set_turbulence(np.full((s2.ncx, s2.ncy), 1.0e6), k0=0.5, w0=200.0)
s2._apply_bc()
s2._update_mut()
# sample mid-domain: cells touching the zeroth-order extrap ghosts see a
# flattened profile there, which is a boundary artefact, not the strain formula
mid = ng + 4
err = abs(float(s2._S2[ng, mid]) - S * S) / (S * S)
assert err < 1e-6, f"strain invariant wrong: {s2._S2[ng, mid]:.4g} vs {S*S:.4g}"
# with F2 -> 0 far from walls mu_t = rho a1 k / max(a1 w, S F2) = rho k / w
mut_ex = rho0 * 0.5 / 200.0
assert abs(float(s2.mu_t[ng, mid]) - mut_ex) / mut_ex < 1e-3
print(f"pure-shear strain S^2 exact ({err*100:.4f}%); production P_k = mu_t S^2 "
      f"= {float(s2.mu_t[ng, mid]) * S * S:.4g}")

# ---- 3) smooth-wall omega BC = Menter's 60 nu / (beta1 y1^2) ----------------
nx, ny, dy = 4, 12, 0.001
I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
s3 = CurvilinearEuler(I * dy, J * dy, gamma=gam)
s3.set_viscous(mu, Rgas=R)
ng = s3.ng
s3.U[0] = rho0
s3.U[1:3] = 0.0
s3.U[3] = p0 / (gam - 1)
wd = np.abs((np.arange(s3.ncy) - ng + 0.5) * dy)[None, :] * np.ones((s3.ncx, 1))
wd = np.maximum(wd, 0.5 * dy)
s3.set_bc(imin="periodic", jmin="noslip", jmax="extrap")
s3.set_turbulence(wd, k0=1e-6, w0=10.0)
s3._turb_bc()
d1 = wd[ng, ng]
w_wall_ex = 60.0 * (mu / rho0) / (_BET1 * d1 * d1)
w_ghost = float(s3.Ut[1][ng, ng - 1] / rho0)
assert abs(w_ghost - w_wall_ex) / w_wall_ex < 1e-6, (w_ghost, w_wall_ex)
print(f"wall omega = 60 nu/(beta1 y1^2) = {w_ghost:.4g} OK")
print("sst test OK")
