"""PDF report generator for a Tachyon CFD run.

Assembles a multi-page PDF (via matplotlib ``PdfPages`` — no extra dependency)
covering geometry & mesh, the solved fields, the thrust history, centerline and
wall-pressure profiles, a Bartz wall heat-flux estimate, and a performance
table.  An optional altitude-sweep result adds a thrust/Isp-vs-altitude page.

Call ``generate_report(...)`` with a solver snapshot and the geometry context;
everything is computed on the CPU from the snapshot, so it works from the GUI
(current run) or headless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SimConfig, PROPELLANT_MIX
from .mask import WALL, INLET, OUTLET, FLUID
from . import probe, heatflux

ACCENT = "#D97757"
TEAL = "#0E7490"
FIELD_PAGES = ["Mach", "Pressure [Pa]", "Temperature [K]",
               "Density [kg/m^3]", "Velocity |V| [m/s]"]


def _extent_mm(nx, ny, dx, y_off):
    return [0.0, nx * dx * 1e3, (ny * dx - y_off) * 1e3, -y_off * 1e3]


def _imshow_field(ax, arr, extent, title, cmap="turbo"):
    im = ax.imshow(arr, origin="upper", extent=extent, cmap=cmap, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    return im


def _geometry_rgba(ct):
    h, w = ct.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    rgba[ct == FLUID] = (0.96, 0.95, 0.91, 1.0)
    rgba[ct == WALL] = (0.10, 0.10, 0.10, 1.0)
    rgba[ct == INLET] = (0.12, 0.43, 1.0, 1.0)
    rgba[ct == OUTLET] = (0.90, 0.24, 0.20, 1.0)
    return rgba


def generate_report(path: str, snap: dict, cfg: SimConfig, *,
                    mask_ct: np.ndarray, dx: float, axis_row: float,
                    y_off: float = 0.0, thrust_history=None,
                    mask_lam: np.ndarray | None = None,
                    engine_name: str = "engine",
                    sweep_results: list | None = None,
                    T_wall: float = 800.0) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fields = snap["fields"]
    meta = snap.get("meta", {})
    perf = meta.get("performance", {})
    ny, nx = mask_ct.shape
    extent = _extent_mm(nx, ny, dx, y_off)

    path = str(Path(path).with_suffix(".pdf"))
    with PdfPages(path) as pdf:
        # ---------------- Page 1: title + performance + conditions ----------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=120)        # A4 portrait
        fig.suptitle(f"Tachyon CFD — {engine_name}", fontsize=20,
                     fontweight="bold", color=ACCENT, y=0.96)
        ax = fig.add_axes([0.08, 0.55, 0.84, 0.34]); ax.axis("off")
        unit = perf.get("force_unit", "N")
        mdu = perf.get("mdot_unit", "kg/s")
        rows = [
            ("Thrust F", f"{perf.get('F', 0):.4g} {unit}"),
            ("  Fx / Fy", f"{perf.get('Fx', 0):.4g} / {perf.get('Fy', 0):.4g} {unit}"),
            ("Mass flow", f"{perf.get('mdot', 0):.4g} {mdu}"),
            ("Specific impulse Isp", f"{perf.get('Isp', 0):.1f} s"),
            ("Eff. exhaust velocity", f"{perf.get('c_eff', 0):.1f} m/s"),
            ("Step", f"{meta.get('step', 0):,}"),
            ("Residual", f"{meta.get('residual', float('nan')):.2e}"),
        ]
        mix = PROPELLANT_MIX.get(cfg.propellant)
        if mix and perf.get("mdot", 0) > 1e-12:
            fuel, ox, of = mix
            mf = perf["mdot"] / (1.0 + of)
            rows.append((f"  {fuel} / {ox}",
                         f"{mf:.3g} / {perf['mdot'] - mf:.3g} {mdu}"))
        txt = "Performance\n" + "\n".join(f"{k:<26}{v}" for k, v in rows)
        ax.text(0.0, 1.0, txt, family="monospace", fontsize=11, va="top")

        # convergence stamp: only quote steady-state numbers when the thrust
        # history is actually flat (last 10 % varies < 0.5 % peak-to-peak)
        from .postproc import thrust_convergence
        conv, rel = thrust_convergence(thrust_history or [])
        if conv:
            stamp, color = f"CONVERGED  (thrust varies {rel*100:.2f}% " \
                           "over last 10% of run)", "#2E7D32"
        elif rel == rel:                       # not NaN
            stamp, color = f"NOT CONVERGED  (thrust varies {rel*100:.1f}% " \
                           "over last 10% of run — numbers are provisional)", \
                           "#C62828"
        else:
            stamp, color = "CONVERGENCE UNKNOWN  (thrust history too short)", \
                           "#C62828"
        ax.text(0.0, 0.02, stamp, fontsize=11, fontweight="bold", color=color,
                va="bottom")

        ax2 = fig.add_axes([0.08, 0.10, 0.84, 0.38]); ax2.axis("off")
        cond = [
            ("Propellant", cfg.propellant),
            ("Gas model", getattr(cfg, "gas_model", "calorically perfect")),
            ("Chamber p0", f"{cfg.inlet_p0:.4g} Pa"),
            ("Chamber T0", f"{cfg.inlet_T0:.1f} K"),
            ("Combustion eff.", f"{getattr(cfg, 'eta_cstar', 1.0):.3f}"
             + ("  (T0_eff = "
                f"{cfg.inlet_T0 * getattr(cfg, 'eta_cstar', 1.0) ** 2:.0f} K)"
                if getattr(cfg, "eta_cstar", 1.0) < 1.0 else "")),
            ("gamma / R", f"{cfg.gamma:.3f} / {cfg.R_gas:.1f} J/(kg K)"),
            ("Ambient pressure", f"{cfg.farfield_p:.4g} Pa"),
            ("Mode", "axisymmetric" if cfg.axisymmetric else "planar 2D"),
            ("Axis location", cfg.axis_location),
            ("Grid", f"{nx} x {ny} cells, dx = {dx*1e3:.4g} mm"),
            ("Mesh density", f"{cfg.mesh_scale:g} x"),
            ("Riemann solver", cfg.flux_scheme.upper()),
            ("Spatial order", f"{cfg.muscl_order} ({cfg.limiter})"),
            ("Turbulence", "k-omega SST" if cfg.turbulence else "off"),
            ("Smooth walls", "cut-cell" if cfg.smooth_boundary else "pixel"),
        ]
        ctxt = "Run conditions\n" + "\n".join(f"{k:<22}{v}" for k, v in cond)
        ax2.text(0.0, 1.0, ctxt, family="monospace", fontsize=10, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Page 2: geometry + Mach + Pressure ----------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
        fig.suptitle("Geometry & key fields", fontsize=14, y=0.97)
        axg = fig.add_subplot(3, 1, 1)
        axg.imshow(_geometry_rgba(mask_ct), origin="upper", extent=extent,
                   aspect="equal")
        axg.set_title("Geometry (wall / inlet / outlet)", fontsize=10)
        axg.set_xlabel("x [mm]", fontsize=8); axg.set_ylabel("y [mm]", fontsize=8)
        axg.tick_params(labelsize=7)
        if cfg.axisymmetric:
            axg.axhline(0.0, color=TEAL, ls="--", lw=1)
        for k, name in enumerate(["Mach", "Pressure [Pa]"]):
            if name in fields:
                ax = fig.add_subplot(3, 1, k + 2)
                _imshow_field(ax, fields[name], extent, name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Page 3: T, density, velocity ----------------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
        fig.suptitle("Thermodynamic & velocity fields", fontsize=14, y=0.97)
        for k, name in enumerate(["Temperature [K]", "Density [kg/m^3]",
                                  "Velocity |V| [m/s]"]):
            if name in fields:
                ax = fig.add_subplot(3, 1, k + 1)
                _imshow_field(ax, fields[name], extent, name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Page 4: thrust history + centerline ---------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
        fig.suptitle("Convergence & centerline", fontsize=14, y=0.97)
        ax = fig.add_subplot(3, 1, 1)
        if thrust_history:
            h = np.asarray(thrust_history, dtype=float)
            ax.plot(h[:, 0], h[:, 1], color=ACCENT, lw=1.5)
        ax.set_title("Thrust history", fontsize=10)
        ax.set_xlabel("step", fontsize=8); ax.set_ylabel(f"F [{unit}]", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        try:
            cl_m = probe.centerline(fields["Mach"], dx, axis_row, y_off=y_off)
            cl_p = probe.centerline(fields["Pressure [Pa]"], dx, axis_row, y_off=y_off)
            axm = fig.add_subplot(3, 1, 2)
            axm.plot(cl_m["x"] * 1e3, cl_m["values"], color=ACCENT)
            axm.set_title("Centerline Mach", fontsize=10)
            axm.set_xlabel("x [mm]", fontsize=8); axm.set_ylabel("M", fontsize=8)
            axm.grid(alpha=0.3); axm.tick_params(labelsize=7)
            axp = fig.add_subplot(3, 1, 3)
            axp.plot(cl_p["x"] * 1e3, cl_p["values"], color=TEAL)
            axp.set_title("Centerline pressure", fontsize=10)
            axp.set_xlabel("x [mm]", fontsize=8); axp.set_ylabel("p [Pa]", fontsize=8)
            axp.grid(alpha=0.3); axp.tick_params(labelsize=7)
        except Exception:
            pass
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Page 5: wall pressure + Bartz heat flux -----------
        if cfg.axisymmetric:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
            fig.suptitle("Wall loads", fontsize=14, y=0.97)
            try:
                wp = probe.wall_pressure(fields["Pressure [Pa]"], mask_ct, dx,
                                         axis_row)
                ax = fig.add_subplot(3, 1, 1)
                ax.plot(wp["x"] * 1e3, wp["values"], color=TEAL)
                ax.set_title("Wall pressure distribution", fontsize=10)
                ax.set_xlabel("x [mm]", fontsize=8)
                ax.set_ylabel("p_wall [Pa]", fontsize=8)
                ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            except Exception:
                pass
            try:
                hf = heatflux.bartz_heat_flux(
                    mask_ct, fields["Mach"], dx, axis_row, cfg,
                    T_wall=T_wall, mdot=perf.get("mdot"))
                if hf.get("valid"):
                    axq = fig.add_subplot(3, 1, 2)
                    axq.plot(hf["x"] * 1e3, hf["q"] / 1e6, color=ACCENT,
                             label="Bartz correlation")
                    # overlay the in-solver wall-function heat flux when the
                    # run used an isothermal no-slip wall (wall_T > 0)
                    qwf = fields.get("Wall heat flux [W/m^2]")
                    if qwf is not None and np.nanmax(np.abs(qwf)) > 1.0e3:
                        wq = probe.wall_pressure(qwf, mask_ct, dx, axis_row)
                        axq.plot(wq["x"] * 1e3, wq["values"] / 1e6,
                                 color=TEAL, lw=1.2,
                                 label="CFD wall function")
                        axq.legend(fontsize=7)
                    axq.axvline(hf["x_throat"] * 1e3, color="#888", ls=":")
                    axq.set_title(
                        f"Bartz wall heat flux  (q_throat = "
                        f"{hf['q_throat']/1e6:.2f} MW/m², T_wall = {T_wall:.0f} K)",
                        fontsize=10)
                    axq.set_xlabel("x [mm]", fontsize=8)
                    axq.set_ylabel("q [MW/m²]", fontsize=8)
                    axq.grid(alpha=0.3); axq.tick_params(labelsize=7)
                    axh = fig.add_subplot(3, 1, 3)
                    axh.plot(hf["x"] * 1e3, hf["h_g"], color=TEAL)
                    axh.set_title("Gas-side heat-transfer coefficient h_g",
                                  fontsize=10)
                    axh.set_xlabel("x [mm]", fontsize=8)
                    axh.set_ylabel("h_g [W/m²/K]", fontsize=8)
                    axh.grid(alpha=0.3); axh.tick_params(labelsize=7)
            except Exception:
                pass
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)

        # ---------------- Page 6: altitude sweep ----------------------------
        if sweep_results:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=120)
            fig.suptitle("Altitude sweep", fontsize=14, y=0.97)
            alt = [r["alt_km"] for r in sweep_results]
            F = [r["F"] for r in sweep_results]
            Isp = [r["Isp"] for r in sweep_results]
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(alt, F, "o-", color=ACCENT)
            ax1.set_title("Thrust vs altitude", fontsize=10)
            ax1.set_xlabel("altitude [km]", fontsize=8)
            ax1.set_ylabel(f"F [{unit}]", fontsize=8)
            ax1.grid(alpha=0.3); ax1.tick_params(labelsize=7)
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(alt, Isp, "s--", color=TEAL)
            ax2.set_title("Specific impulse vs altitude", fontsize=10)
            ax2.set_xlabel("altitude [km]", fontsize=8)
            ax2.set_ylabel("Isp [s]", fontsize=8)
            ax2.grid(alpha=0.3); ax2.tick_params(labelsize=7)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = f"Tachyon CFD report — {engine_name}"
        d["Creator"] = "Tachyon CFD"
    return path
