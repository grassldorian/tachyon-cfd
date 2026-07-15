"""Curvilinear viscous (laminar Navier-Stokes): Couette flow. The exact linear
profile is a steady solution and stays steady; the wall shear extracted from the
viscous flux equals mu*U/h. (Full relaxation from rest also reaches the linear
profile to 0.01% -- see scratch val; too many steps for a unit test.)"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.curvilinear import CurvilinearEuler

nx, ny = 4, 16
dx = dy = 0.00625
h = ny * dy
gam, R, rho0, p0 = 1.4, 287.0, 1.2, 1.0e5
mu, U = 0.05, 10.0

I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
s = CurvilinearEuler(I * dx, J * dy, gamma=gam)
s.set_viscous(mu, Pr=0.72, Rgas=R)
ng = s.ng
yc = (np.arange(ny) + 0.5) * dy
u0 = (U * yc / h)[None, :] * np.ones((nx, 1))
s.U[0, ng:ng + nx, ng:ng + ny] = rho0
s.U[1, ng:ng + nx, ng:ng + ny] = rho0 * u0
s.U[2, ng:ng + nx, ng:ng + ny] = 0.0
s.U[3, ng:ng + nx, ng:ng + ny] = p0 / (gam - 1) + 0.5 * rho0 * u0 ** 2
s.set_bc(imin="periodic", jmin="noslip", jmax="noslip")
s.set_wall_velocity("jmin", 0.0)
s.set_wall_velocity("jmax", U)

# 1) the exact linear Couette profile is steady
ref = s.U.copy()
dt = s.max_wave_dt(0.4)
for _ in range(200):
    s.step(dt)
u = s.U[1, ng:ng + nx, ng:ng + ny] / s.U[0, ng:ng + nx, ng:ng + ny]
u_ref = ref[1, ng:ng + nx, ng:ng + ny] / ref[0, ng:ng + nx, ng:ng + ny]
drift = float(np.max(np.abs(u - u_ref))) / U
assert drift < 1e-3, f"linear Couette not steady: {drift:.2e}"

# 2) wall shear from the viscous flux equals mu*U/h
s.U[...] = ref
s._apply_bc()
rho, uu, vv, pp = s.primitives()
_, Fvj = s._viscous_fluxes(rho, uu, vv, pp)
face = ng - 1                                   # jmin wall j-interface
Sy = s.Syj[ng, ng]                              # wall face length (~dx)
tau_num = abs(float(np.mean(Fvj[1, ng:ng + nx, face])) / Sy)
tau_exact = mu * U / h
rel = abs(tau_num - tau_exact) / tau_exact
assert rel < 0.02, f"wall shear off: {tau_num:.4f} vs {tau_exact:.4f}"
print(f"Couette: linear profile steady (drift {drift:.1e}); wall shear "
      f"{tau_num:.4f} vs mu*U/h {tau_exact:.4f} ({rel*100:.1f}%)")
print("viscous test OK")
