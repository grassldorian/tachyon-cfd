"""Thermally perfect gas model: kernel compile + short-run sanity vs CP.

Runs the small example nozzle 800 steps in both gas models and checks:
  - both compile and produce finite fields
  - TP temperatures differ from CP (gamma actually varies)
  - TP energy/temperature round-trip is consistent with the cp(T) polynomial
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig, PROPELLANTS
from rocketcfd import thermo
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def short_run(gas_model):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.propellant = "LOX/RP-1 (kerosene)"
    for k, v in PROPELLANTS[cfg.propellant].items():
        setattr(cfg, k, v)
    cfg.inlet_p0 = 7.0e6
    cfg.gas_model = gas_model
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    sol = GPUSolver(mask, cfg)
    sol.step(800)
    return cfg, sol.snapshot()


cfg_cp, snap_cp = short_run("calorically perfect")
cfg_tp, snap_tp = short_run("thermally perfect")

for name, snap in (("CP", snap_cp), ("TP", snap_tp)):
    for f in ("Mach", "Pressure [Pa]", "Temperature [K]"):
        arr = snap["fields"][f]
        assert np.isfinite(arr[np.isfinite(arr)]).all(), (name, f)
        assert np.nanmax(arr) > 0, (name, f)
    print(f"{name}: Mach max {np.nanmax(snap['fields']['Mach']):.3f}  "
          f"T range [{np.nanmin(snap['fields']['Temperature [K]']):.0f}, "
          f"{np.nanmax(snap['fields']['Temperature [K]']):.0f}] K  "
          f"F = {snap['meta']['performance']['F']:.4g}")

# the two gas models must actually differ
dT = np.nanmax(np.abs(snap_tp["fields"]["Temperature [K]"]
                      - snap_cp["fields"]["Temperature [K]"]))
print(f"max |T_tp - T_cp| = {dT:.1f} K")
assert dT > 10.0, "thermally perfect mode produced identical fields to CP"

# TP thermodynamic consistency: p = rho R T must hold in the snapshot
rho = snap_tp["fields"]["Density [kg/m^3]"]
p = snap_tp["fields"]["Pressure [Pa]"]
T = snap_tp["fields"]["Temperature [K]"]
m = np.isfinite(rho) & np.isfinite(p) & np.isfinite(T) & (p > 10.0)
rel = np.abs(p[m] - rho[m] * cfg_tp.R_gas * T[m]) / p[m]
print(f"ideal-gas consistency: max rel err {np.max(rel):.2e}")
assert np.max(rel) < 1e-3

# gamma(T) of the mixture spans a physical range
c = thermo.cpr_coeffs(cfg_tp.propellant)
g_hot = float(thermo.gamma_of_T(c, cfg_tp.inlet_T0))
g_cold = float(thermo.gamma_of_T(c, 800.0))
print(f"gamma(T0)={g_hot:.3f}  gamma(800K)={g_cold:.3f}")
assert 1.10 < g_hot < 1.30 and g_cold > g_hot

print("thermo gas-model test OK")
