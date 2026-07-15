"""Overset Phase 3: Chimera connectivity — a sensible hole/fringe map and
donor/receptor interpolation that is a partition of unity (free-stream exact),
exact for linear fields on the affine Cartesian side, and convergent on the
curvilinear side."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd import engine_design as ed, overset
from rocketcfd.overset_connect import (OversetConnectivity, cell_centers,
                                       HOLE, FRINGE)

x_mm, r_mm, _ = ed.build_contour(
    dict(chamber_l=0.09, chamber_d=0.11, throat_d=0.04, exit_d=0.09,
         nozzle_l=0.11), "Bell (Rao 80%)", fillet_mm=5, throat_r_mm=6)


def make(nxi, neta):
    g = overset.wall_normal_grid(x_mm * 1e-3, r_mm * 1e-3, n_eta=neta,
                                 first_cell=6e-5, growth=1.2, n_xi=nxi)
    b = cell_centers(g["X"], g["R"])
    x0, y0 = b[0].min() - 0.004, b[1].min() - 0.004
    x1, y1 = b[0].max() + 0.004, b[1].max() + 0.004
    dx = 0.001
    c = OversetConnectivity(g["X"], g["R"], x0, y0, dx,
                            int((x1 - x0) / dx), int((y1 - y0) / dx),
                            blank_frac=0.55, fringe_layers=2)
    return b, c


b, conn = make(60, 16)

# hole / fringe / receptor map is populated
assert (conn.blank == HOLE).sum() > 0
assert (conn.blank == FRINGE).sum() > 0 and len(conn.fr_don) == len(conn.fr_idx)
assert len(conn.br_idx) > 0
print(f"connectivity: holes {(conn.blank==HOLE).sum()}, "
      f"fringe {(conn.blank==FRINGE).sum()}, receptors {len(conn.br_idx)}")

# partition of unity both ways -> uniform transfers exactly (conservation)
u_c = float(np.max(np.abs(conn.interp_to_cart_fringe(np.ones_like(b[0])) - 1.0)))
u_b = float(np.max(np.abs(
    conn.interp_to_body_outer(np.ones((conn.ncx, conn.ncy))) - 1.0)))
assert u_c < 1e-12 and u_b < 1e-12, (u_c, u_b)

# linear field: exact on the affine Cartesian donor side
def L(x, y):
    return 3.0 + 1.5 * x - 0.7 * y
cf = L(conn.ccx[:, None], conn.ccy[None, :])
got = conn.interp_to_body_outer(cf)
want = np.array([L(b[0][i, j], b[1][i, j]) for i, j in conn.br_idx])
assert np.max(np.abs(got - want)) < 1e-11, "cart->body not exact for linear"
print("partition-of-unity + cart->body linear-exact OK")

# curvilinear side (body->cart): 2nd-ish-order convergence of a smooth field
def S(x, y):
    return np.sin(40 * x) * np.cos(35 * y)

def rms(bb, cc):
    g_ = cc.interp_to_cart_fringe(S(*bb))
    w_ = S(cc.ccx[cc.fr_idx[:, 0]], cc.ccy[cc.fr_idx[:, 1]])
    return float(np.sqrt(np.mean((g_ - w_) ** 2)))

e0 = rms(b, conn)
b2, conn2 = make(120, 32)
e1 = rms(b2, conn2)
assert e1 < e0 and e0 / e1 > 2.5, f"weak convergence: {e0:.2e}->{e1:.2e}"
print(f"body->cart smooth RMS {e0:.2e}->{e1:.2e} (ratio {e0/e1:.2f}) OK")
print("overset-connect test OK")
