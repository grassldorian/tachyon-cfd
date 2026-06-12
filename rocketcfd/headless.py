"""Headless CLI runner: run a simulation from a PNG without the GUI.

Usage:
    python -m rocketcfd.headless nozzle.png --steps 2000 --out results
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import SimConfig
from .mask import load_mask
from .solver import GPUSolver


def main():
    ap = argparse.ArgumentParser(description="RocketCFD headless runner")
    ap.add_argument("png", help="input nozzle PNG/SVG (black=wall, white=fluid, blue=inlet)")
    ap.add_argument("--config", help="JSON config file", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--report", type=int, default=100)
    ap.add_argument("--axi", action="store_true", help="force axisymmetric mode")
    ap.add_argument("--scheme", default=None, choices=["hllc", "hll", "roe", "ausm"],
                    help="override Riemann solver")
    ap.add_argument("--no-smooth", action="store_true",
                    help="disable cut-cell smooth walls (pixel boundaries)")
    ap.add_argument("--mesh", type=float, default=None,
                    help="mesh density multiplier (>1 finer, <1 coarser)")
    ap.add_argument("--out", default=None, help="output directory for NPZ + plots")
    args = ap.parse_args()

    cfg = SimConfig.load(args.config) if args.config else SimConfig()
    if args.axi:
        cfg.axisymmetric = True
    if args.scheme:
        cfg.flux_scheme = args.scheme
    if args.no_smooth:
        cfg.smooth_boundary = False
    if args.mesh:
        cfg.mesh_scale = args.mesh
    mask = load_mask(args.png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    print(f"grid {mask.nx}x{mask.ny}: {mask.n_fluid} fluid cells, "
          f"{mask.n_inlet} inlet cells, dx={mask.dx*1000:.3f} mm, "
          f"{'axisymmetric (' + cfg.axis_location + ')' if cfg.axisymmetric else 'planar 2D'}, "
          f"{'smooth cut-cell walls' if mask.smooth else 'pixel walls'}")

    solver = GPUSolver(mask, cfg)
    t0 = time.perf_counter()
    done = 0
    while done < args.steps:
        n = min(args.report, args.steps - done)
        solver.step(n)
        done += n
        snap = solver.snapshot()
        f = snap["fields"]
        pf = snap["meta"]["performance"]
        print(f"step {solver.step_count:6d}  res {solver.residual:.3e}  "
              f"Mach max {np.nanmax(f['Mach']):6.3f}  "
              f"T [{np.nanmin(f['Temperature [K]']):7.1f},{np.nanmax(f['Temperature [K]']):8.1f}] K  "
              f"p [{np.nanmin(f['Pressure [Pa]']):10.1f},{np.nanmax(f['Pressure [Pa]']):12.1f}] Pa  "
              f"F {pf['F']:10.4g}  "
              f"{solver._steps_per_sec:6.1f} steps/s")
        if not np.isfinite(np.nanmax(f["Mach"])):
            print("!! non-finite field detected, aborting")
            break
        if solver.residual < cfg.residual_target:
            print("converged")
            break
    print(f"wall time {time.perf_counter()-t0:.1f} s")

    perf = solver.snapshot()["meta"]["performance"]
    print(f"thrust  F = {perf['F']:.4g} {perf['force_unit']}  "
          f"(Fx={perf['Fx']:.4g}, Fy={perf['Fy']:.4g})")
    print(f"massflow mdot = {perf['mdot']:.4g} {perf['mdot_unit']}")
    print(f"Isp = {perf['Isp']:.1f} s   c_eff = {perf['c_eff']:.1f} m/s")

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        solver.save_npz(outdir / "fields.npz")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            snap = solver.snapshot()
            for name in ("Mach", "Temperature [K]", "Density [kg/m^3]",
                         "Velocity |V| [m/s]", "Pressure [Pa]"):
                fig, ax = plt.subplots(figsize=(9, 7), dpi=110)
                im = ax.imshow(snap["fields"][name], cmap="turbo", origin="upper")
                fig.colorbar(im, ax=ax, label=name)
                ax.set_title(f"{name} — step {solver.step_count}")
                safe = name.split(" [")[0].replace(" ", "_").replace("|", "").lower()
                fig.savefig(outdir / f"{safe}.png", bbox_inches="tight")
                plt.close(fig)
            print(f"results written to {outdir}")
        except Exception as e:                       # plotting is best-effort
            print(f"plotting failed: {e}")


if __name__ == "__main__":
    main()
