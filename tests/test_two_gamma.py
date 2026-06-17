"""Two-gamma plume mixing: the transported exhaust mass fraction stays
bounded [0,1], the engine core fills with exhaust (Z->1) while ambient stays
air (Z->0) with a mixing layer between, and thrust is unchanged vs the
single-gas run (the engine core is pure exhaust either way).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.cuda_kernels import axis_j
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def run(two_gamma, steps=9000):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.inlet_p0 = 4.0e6
    cfg.gamma, cfg.R_gas = 1.20, 380.0      # low-gamma exhaust...
    cfg.ambient_gamma, cfg.ambient_R = 1.40, 287.0   # ...mixing into air
    cfg.two_gamma = two_gamma
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    sol = GPUSolver(mask, cfg)
    sol.step(steps)
    return cfg, sol, sol.snapshot()


cfg0, sol0, s0 = run(False)
cfg1, sol1, s1 = run(True)

# fields present + finite + bounded
Z = s1["fields"]["Mixture fraction [-]"]
g = s1["fields"]["Local gamma [-]"]
zf = Z[np.isfinite(Z)]
assert zf.min() >= -1e-4 and zf.max() <= 1.0 + 1e-4, (zf.min(), zf.max())
print(f"mixture fraction range [{zf.min():.3f}, {zf.max():.3f}]  (bounded)")

# engine core (near the axis at the throat region) should be ~pure exhaust
jc = int(round(axis_j(cfg1, sol1.ny) - 2.0))
mach = s1["fields"]["Mach"]
sup = np.where(np.isfinite(mach[jc]) & (mach[jc] > 1.5))[0]
core_Z = float(np.nanmean(Z[jc, sup[:10]])) if sup.size else 0.0
print(f"supersonic core Z = {core_Z:.3f} (exhaust-dominated; <1 from the "
      f"1st-order scalar's numerical diffusion)")
assert core_Z > 0.7, core_Z

# ambient far corner should be air (Z ~ 0)
corner_Z = float(np.nanmean(Z[:20, -40:]))
print(f"ambient corner Z = {corner_Z:.3f} (expect near 0 = air)")
assert corner_Z < 0.15, corner_Z

# a real mixing layer exists (intermediate Z somewhere)
frac_mixed = float(np.mean((zf > 0.2) & (zf < 0.8)))
print(f"mixing-layer cells (0.2<Z<0.8): {frac_mixed*100:.1f}%")
assert frac_mixed > 0.01, "no mixing layer formed"

# local gamma spans exhaust->air
gf = g[np.isfinite(g)]
print(f"local gamma range [{gf.min():.3f}, {gf.max():.3f}]")
assert gf.min() < cfg1.gamma + 0.02 and gf.max() > 1.38

# thrust unchanged (engine core is pure exhaust in both)
F0, F1 = s0["meta"]["performance"]["F"], s1["meta"]["performance"]["F"]
dF = abs(F1 - F0) / abs(F0)
print(f"thrust off {F0:.5g} N, two-gamma {F1:.5g} N ({dF*100:+.2f}%)")
assert dF < 0.03, f"thrust moved {dF*100:.1f}% (engine core should be unchanged)"
print("two-gamma test OK")
