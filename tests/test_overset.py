"""Overset Phase 1: the body-fitted boundary-layer grid generator is valid
(no folded cells, near-orthogonal, correct wall clustering)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd import engine_design as ed
from rocketcfd import overset

geom = dict(chamber_l=0.090, chamber_d=0.110, throat_d=0.040,
            exit_d=0.090, nozzle_l=0.110)
x_mm, r_mm, key = ed.build_contour(geom, "Bell (Rao 80%)",
                                   fillet_mm=5.0, throat_r_mm=6.0)
x, r = x_mm * 1e-3, r_mm * 1e-3

g = overset.wall_normal_grid(x, r, n_eta=30, first_cell=8e-6, growth=1.12,
                             n_xi=260, smooth_normals=4)
X, R = g["X"], g["R"]

# shape + surface anchoring
assert X.shape == (260, 31) and R.shape == (260, 31), X.shape
assert np.allclose(X[:, 0], g["xw"]) and np.allclose(R[:, 0], g["rw"])

# a valid mesh: no folded (sign-flipped) cells, strictly positive min area
assert g["folded_cells"] == 0, g["folded_cells"]
assert g["min_cell_area"] > 0.0, g["min_cell_area"]
assert g["min_orthogonality_deg"] > 60.0, g["min_orthogonality_deg"]

# wall clustering: first cell = requested, geometric growth, marches into fluid
d = g["eta"]
assert abs(d[1] - 8e-6) < 1e-12, d[1]
ratios = np.diff(d)[1:] / np.diff(d)[:-1]
assert np.allclose(ratios, 1.12, atol=1e-4), ratios[:3]
assert np.mean(R[:, -1]) < np.mean(R[:, 0]), "grid should march toward the axis"
assert g["thickness"] > d[1] * g["n_eta"], "geometric growth increases spacing"

# y+ helper
yp1 = overset.first_cell_for_yplus(1.0, 95.0, 1e-4 / 5.0)
assert 0.1e-6 < yp1 < 1.0e-6, yp1

print(f"overset grid {g['n_xi']}x{g['n_eta']+1}: folded {g['folded_cells']}, "
      f"min area {g['min_cell_area']:.2e}, orth {g['min_orthogonality_deg']:.0f} deg, "
      f"first cell {d[1]*1e6:.1f} um, thickness {g['thickness']*1e3:.2f} mm")
print("overset test OK")
