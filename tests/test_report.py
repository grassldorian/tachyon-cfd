"""End-to-end check: short GPU run on the example nozzle -> PDF report."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.cuda_kernels import axis_j
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver
from rocketcfd.report import generate_report

cfg = SimConfig()
cfg.axisymmetric = True
cfg.axis_location = "center"

png = str(ROOT / "examples" / "nozzle_small.png")
mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                 smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                 mesh_scale=cfg.mesh_scale)
sol = GPUSolver(mask, cfg)
for _ in range(6):
    sol.step(100)
    sol.snapshot()                    # populate thrust_history like the GUI
print("steps done:", sol.step_count)

snap = sol.snapshot()
snap["thrust_history"] = sol.thrust_history
axis_row = axis_j(cfg, sol.ny) - 2.0
fake_sweep = [
    dict(alt_km=0.0, F=1200.0, Isp=210.0, force_unit="N"),
    dict(alt_km=10.0, F=1350.0, Isp=235.0, force_unit="N"),
    dict(alt_km=20.0, F=1410.0, Isp=246.0, force_unit="N"),
]
out_path = os.path.join(tempfile.gettempdir(), "tachyon_test_report.pdf")
out = generate_report(
    out_path, snap, cfg,
    mask_ct=mask.cell_type[2:-2, 2:-2], dx=mask.dx,
    axis_row=axis_row, y_off=(axis_row + 0.5) * mask.dx,
    thrust_history=snap["thrust_history"],
    mask_lam=mask.lam[2:-2, 2:-2] if mask.smooth else None,
    engine_name="nozzle_small (e2e test)",
    sweep_results=fake_sweep)
sz = os.path.getsize(out)
print(f"report written: {out} ({sz/1024:.0f} KiB)")
assert sz > 20_000, "PDF suspiciously small"
print("report e2e OK")
