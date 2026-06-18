"""Mass-flow / Isp robustness: once the flow is developed, the reported mdot,
Isp and c_eff must not randomly drop to zero (regression for the injector-face
acoustic dropout) nor blow up (regression for ambient/plume contamination).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig, PROPELLANTS
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")

# the exact config the dropout was first reported with
cfg = SimConfig()
cfg.axisymmetric = True
cfg.axis_location = "center"
cfg.wall_type = "slip"
cfg.flux_scheme = "hllc"
cfg.muscl_order = 2
cfg.limiter = "minmod"
cfg.gas_model = "calorically perfect"
p = PROPELLANTS["LOX/Ethanol (75%)"]
cfg.gamma = p["gamma"]; cfg.R_gas = p["R_gas"]; cfg.inlet_T0 = p["inlet_T0"]
cfg.inlet_p0 = 20e5

mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                 smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                 axisym_center=True)
sol = GPUSolver(mask, cfg)

dropouts = 0
developed = 0
mdots = []
for _ in range(24):
    sol.step(500)
    perf = sol.snapshot()["meta"]["performance"]
    md, isp, ce = perf["mdot"], perf["Isp"], perf["c_eff"]
    # once the chamber has filled (past the soft-start ramp) the flow is
    # developed and the performance numbers must be finite and non-zero
    if sol.step_count > cfg.inlet_ramp_steps + 1500:
        developed += 1
        mdots.append(md)
        if md <= 1e-9 or isp == 0.0 or ce == 0.0:
            dropouts += 1
        # c_eff must stay physical (no divide-by-tiny-mdot blowups)
        assert ce < 1.5e4, f"c_eff blew up: {ce:.1f} m/s at step {sol.step_count}"

assert developed >= 5, "not enough developed-flow snapshots to judge"
assert dropouts == 0, f"{dropouts}/{developed} developed snapshots had mdot/Isp = 0"
print(f"developed snapshots: {developed}, dropouts: {dropouts}")
print(f"mdot range: {min(mdots):.3f} .. {max(mdots):.3f} kg/s (stable, non-zero)")
print("mass-flow robustness test OK")
