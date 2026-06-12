"""Altitude sweep: run one engine across a range of ambient back-pressures.

For a fixed nozzle the thrust and specific impulse depend strongly on the
ambient pressure: a bell nozzle is optimal at one altitude and loses
performance elsewhere (overexpanded low, underexpanded high), while an
aerospike compensates and keeps Isp high over a wide altitude band.  This
module batch-runs the solver at a list of back-pressures (derived from
altitudes via the US Standard Atmosphere) and records F, Isp and mdot so the
altitude-compensation behaviour can be plotted.

Runs are **warm-started**: results are computed from sea level upward, each
back-pressure continuing from the previous converged flow (see
``GPUSolver.reconfigure``), which is much faster than restarting from
quiescent gas every time.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import SimConfig
from .mask import load_mask
from .solver import GPUSolver

G0 = 9.80665


# --------------------------------------------------------- standard atmosphere
def isa_pressure(alt_m: float) -> float:
    """US Standard Atmosphere static pressure [Pa] vs geometric altitude [m].

    Piecewise-analytic through the stratosphere (0..32 km); an exponential
    extrapolation above that keeps very high altitudes monotonic and small.
    """
    h = float(alt_m)
    if h <= 11000.0:                       # troposphere, lapse 6.5 K/km
        T = 288.15 - 0.0065 * h
        return 101325.0 * (T / 288.15) ** 5.255877
    if h <= 20000.0:                       # tropopause, isothermal 216.65 K
        return 22632.06 * np.exp(-(h - 11000.0) / 6341.62)
    if h <= 32000.0:                       # stratosphere 1, lapse +1.0 K/km
        T = 216.65 + 0.001 * (h - 20000.0)
        return 5474.89 * (T / 216.65) ** (-34.1632)
    # above 32 km: exponential tail (scale height ~7 km)
    return 868.02 * np.exp(-(h - 32000.0) / 7000.0)


def isa_temperature(alt_m: float) -> float:
    """US Standard Atmosphere static temperature [K] vs altitude [m]."""
    h = float(alt_m)
    if h <= 11000.0:
        return 288.15 - 0.0065 * h
    if h <= 20000.0:
        return 216.65
    if h <= 32000.0:
        return 216.65 + 0.001 * (h - 20000.0)
    return 228.65


def altitude_for_pressure(pa: float) -> float:
    """Approximate geometric altitude [m] for a static pressure [Pa] (inverse ISA)."""
    pa = max(float(pa), 1.0)
    if pa >= 22632.06:                     # troposphere
        return (288.15 / 0.0065) * (1.0 - (pa / 101325.0) ** (1.0 / 5.255877))
    if pa >= 5474.89:                      # isothermal
        return 11000.0 - 6341.62 * np.log(pa / 22632.06)
    if pa >= 868.02:
        return 20000.0 + (216.65 / 0.001) * ((pa / 5474.89) ** (-1.0 / 34.1632) - 1.0)
    return 32000.0 - 7000.0 * np.log(pa / 868.02)


# ------------------------------------------------------------------- sweep
def sweep(png_path: str, cfg: SimConfig, pressures, *, steps: int = 4000,
          avg_frac: float = 0.25, warm_start: bool = True,
          progress=None, ambient_temperature: bool = True) -> list[dict]:
    """Run ``png_path`` at each ambient pressure and return performance points.

    Parameters
    ----------
    pressures : iterable of ambient static pressures [Pa].
    steps : solver steps per point.
    avg_frac : fraction of the run (trailing) over which thrust/Isp are averaged,
        which smooths the residual startup/transient wobble.
    warm_start : continue each point from the previous converged flow.
    progress : optional callback ``(done, total, p_amb)``.
    ambient_temperature : also set farfield_T from the ISA temperature.

    Returns a list of dicts (one per pressure, sorted by ascending pressure):
      p_amb [Pa], alt_km, F, Isp, mdot, c_eff, F_std, residual.
    """
    pressures = [float(p) for p in pressures]
    mask = load_mask(png_path, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    if mask.n_fluid == 0:
        raise RuntimeError("No fluid cells in the engine image.")

    # sea level first, then upward: each lower pressure is a gentle continuation
    order = sorted(range(len(pressures)), key=lambda i: -pressures[i])
    solver = None
    out_by_idx: dict[int, dict] = {}
    chunk = max(50, steps // 20)
    n_avg = max(1, int(steps * avg_frac))

    for k, idx in enumerate(order):
        pa = pressures[idx]
        kw = {"farfield_p": pa}
        if ambient_temperature:
            alt = altitude_for_pressure(pa)
            kw["farfield_T"] = isa_temperature(alt)
        c = replace(cfg, **kw)
        if solver is None or not warm_start:
            solver = GPUSolver(mask, c)
        else:
            solver.reconfigure(c)
        Fs, perf = [], None
        done = 0
        while done < steps:
            n = min(chunk, steps - done)
            solver.step(n)
            done += n
            perf = solver.snapshot()["meta"]["performance"]
            if done > steps - n_avg:
                Fs.append((perf["F"], perf["mdot"]))
        Favg = float(np.mean([f for f, _ in Fs]))
        Fstd = float(np.std([f for f, _ in Fs]))
        mdot = float(np.mean([m for _, m in Fs]))
        out_by_idx[idx] = {
            "p_amb": pa,
            "alt_km": altitude_for_pressure(pa) / 1000.0,
            "F": Favg, "F_std": Fstd, "mdot": mdot,
            "Isp": Favg / (mdot * G0) if mdot > 1e-12 else 0.0,
            "c_eff": Favg / mdot if mdot > 1e-12 else 0.0,
            "residual": float(solver.residual),
            "force_unit": perf["force_unit"],
        }
        if progress is not None:
            progress(k + 1, len(order), pa)
    return [out_by_idx[i] for i in sorted(out_by_idx,
                                          key=lambda i: pressures[i])]


def default_altitudes_km() -> list[float]:
    return [0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0]


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Tachyon CFD altitude sweep")
    ap.add_argument("png", help="engine PNG/SVG")
    ap.add_argument("--config", default=None, help="JSON config")
    ap.add_argument("--alts", type=float, nargs="+", default=None,
                    help="altitudes [km] (default 0..30)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--axi", action="store_true")
    ap.add_argument("--mesh", type=float, default=None)
    ap.add_argument("--no-warm", action="store_true", help="cold-start each point")
    ap.add_argument("--out", default=None, help="save plot PNG + CSV here")
    args = ap.parse_args()

    cfg = SimConfig.load(args.config) if args.config else SimConfig()
    if args.axi:
        cfg.axisymmetric = True
    if args.mesh:
        cfg.mesh_scale = args.mesh
    alts = args.alts if args.alts else default_altitudes_km()
    pressures = [isa_pressure(a * 1000.0) for a in alts]

    def prog(done, total, pa):
        print(f"  [{done}/{total}] p_amb = {pa:9.1f} Pa "
              f"(alt {altitude_for_pressure(pa)/1000:5.1f} km)")

    print(f"sweeping {len(pressures)} altitudes, {args.steps} steps each...")
    res = sweep(args.png, cfg, pressures, steps=args.steps,
                warm_start=not args.no_warm, progress=prog)
    unit = res[0]["force_unit"]
    print(f"\n{'alt[km]':>8} {'p_amb[Pa]':>11} {'F['+unit+']':>12} "
          f"{'Isp[s]':>8} {'mdot':>10}")
    for r in res:
        print(f"{r['alt_km']:8.1f} {r['p_amb']:11.1f} {r['F']:12.4g} "
              f"{r['Isp']:8.1f} {r['mdot']:10.4g}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        _save_outputs(res, outp, unit)
        print(f"\nwrote {outp.with_suffix('.png')} and {outp.with_suffix('.csv')}")


def _save_outputs(res: list[dict], outp: Path, unit: str):
    import csv
    with open(outp.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["alt_km", "p_amb_Pa", "F", "F_std", "Isp_s", "mdot", "c_eff"])
        for r in res:
            w.writerow([r["alt_km"], r["p_amb"], r["F"], r["F_std"],
                        r["Isp"], r["mdot"], r["c_eff"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        alt = [r["alt_km"] for r in res]
        F = [r["F"] for r in res]
        Isp = [r["Isp"] for r in res]
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=120)
        ax1.plot(alt, F, "o-", color="#D97757", label="Thrust")
        ax1.set_xlabel("altitude [km]")
        ax1.set_ylabel(f"thrust F [{unit}]", color="#D97757")
        ax2 = ax1.twinx()
        ax2.plot(alt, Isp, "s--", color="#0E7490", label="Isp")
        ax2.set_ylabel("specific impulse Isp [s]", color="#0E7490")
        ax1.set_title("Altitude sweep — thrust & Isp vs altitude")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outp.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
