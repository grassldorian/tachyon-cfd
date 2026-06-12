"""Wall heat-flux estimate via the Bartz correlation.

The Bartz equation gives the gas-side convective heat-transfer coefficient
along a rocket-nozzle contour from the throat geometry and the chamber
(stagnation) conditions (Bartz, 1957; see Sutton & Biblarz, *Rocket
Propulsion Elements*, and Huzel & Huang, NASA SP-125):

    h_g = (0.026 / D_t^0.2) * (mu^0.2 * c_p / Pr^0.6)
          * (p_c / c*)^0.8 * (D_t / R_c)^0.1 * (A_t / A)^0.9 * sigma

with the property-variation correction

    sigma = 1 / ( [0.5*(T_wg/T_c)*(1 + (g-1)/2 M^2) + 0.5]^0.68
                  * [1 + (g-1)/2 M^2]^0.12 )

All quantities are SI, so ``h_g`` comes out in W/(m^2 K).  The transport
properties (mu, c_p, Pr) are evaluated at the chamber stagnation temperature
using the same Sutherland law the solver uses.  The local convective flux is

    q(x) = h_g(x) * (T_aw(x) - T_wg)

where the adiabatic-wall (recovery) temperature uses a turbulent recovery
factor r = Pr^(1/3):

    T_aw = T_c * (1 + r*(g-1)/2 M^2) / (1 + (g-1)/2 M^2)

Assumptions / limitations
-------------------------
* Frozen ideal gas with a single gamma (consistent with the solver).
* Gas-side wall temperature ``T_wall`` is an input (default 800 K); a real
  design couples this to the coolant.  Heat flux scales weakly with it.
* The characteristic velocity ``c*`` defaults to the ideal frozen value from
  the chamber conditions; pass the measured mass flow to use ``c* = p_c A_t / mdot``.
* The throat radius of curvature ``R_c`` is fit from the contour; if the fit
  is unreliable it falls back to ``R_c = 1.5 * r_t`` (a common bell-nozzle value).
This is an engineering estimate, not a conjugate heat-transfer solution.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from . import probe


def ideal_cstar(cfg: SimConfig) -> float:
    """Ideal (frozen) characteristic velocity c* [m/s] from chamber conditions."""
    g = cfg.gamma
    R = cfg.R_gas
    eta = getattr(cfg, "eta_cstar", 1.0)
    T0 = cfg.inlet_T0 * eta * eta
    gp = (g + 1.0) / (g - 1.0)
    gamma_fac = np.sqrt(g) * (2.0 / (g + 1.0)) ** (0.5 * gp)
    return float(np.sqrt(R * T0) / gamma_fac)


def _throat_curvature_radius(x: np.ndarray, r: np.ndarray, i_t: int,
                             r_t: float) -> float:
    """Radius of curvature of the contour at the throat, from a local fit.

    Fits a parabola r(x) ~ a (x-x_t)^2 + ... over a small window around the
    throat; R_c = 1 / (2a).  Falls back to 1.5*r_t when the fit is degenerate.
    """
    n = len(x)
    half = max(3, n // 40)
    lo = max(0, i_t - half)
    hi = min(n, i_t + half + 1)
    xs = x[lo:hi]
    rs = r[lo:hi]
    good = np.isfinite(rs)
    if good.sum() >= 3 and np.ptp(xs[good]) > 0:
        try:
            a = np.polyfit(xs[good] - x[i_t], rs[good], 2)[0]
            if a > 1e-9:
                Rc = 1.0 / (2.0 * a)
                if np.isfinite(Rc) and Rc > 0:
                    return float(np.clip(Rc, 0.2 * r_t, 50.0 * r_t))
        except Exception:
            pass
    return 1.5 * r_t


def bartz_heat_flux(ct: np.ndarray, mach: np.ndarray, dx: float,
                    axis_row: float, cfg: SimConfig, *,
                    T_wall: float = 800.0, mdot: float | None = None,
                    side: str = "upper") -> dict:
    """Bartz heat-transfer coefficient and flux along the nozzle wall.

    Parameters
    ----------
    ct : interior cell-type map (ny, nx).
    mach : interior Mach field (ny, nx), NaN in walls.
    axis_row : axis row position (interior, half-integer).
    cfg : SimConfig (chamber p0/T0, gamma, R, Pr, Sutherland mu).
    T_wall : assumed gas-side wall temperature [K].
    mdot : measured mass flow [kg/s]; if given, c* = p_c A_t / mdot.

    Returns dict with arrays along the contour (only where a wall exists):
      x [m], r [m], M [-], h_g [W/m^2/K], q [W/m^2], T_aw [K],
    and scalars: throat_radius, throat_area, D_t, R_c, c_star, q_throat, q_max.
    Returns an empty result (``valid=False``) if no usable throat is found.
    """
    g = cfg.gamma
    Pr = cfg.Pr
    eta = getattr(cfg, "eta_cstar", 1.0)
    T0 = cfg.inlet_T0 * eta * eta            # effective chamber temperature
    p_c = cfg.inlet_p0
    cp = cfg.cp
    mu = cfg.sutherland_mu(T0)

    con = probe.wall_contour(ct, dx, axis_row, side=side)
    x = con["x"]
    r = con["r"]
    fr = con["fluid_row"]
    have = np.isfinite(r) & (fr >= 0)
    if have.sum() < 5:
        return {"valid": False}

    cols = np.flatnonzero(have)
    xv = x[cols]
    rv = r[cols]
    Mv = np.array([mach[fr[i], i] for i in cols], dtype=np.float64)
    Mv = np.nan_to_num(Mv, nan=0.0)

    # throat = minimum radius
    k_t = int(np.argmin(rv))
    r_t = float(rv[k_t])
    if r_t <= 0:
        return {"valid": False}
    A_t = np.pi * r_t * r_t
    D_t = 2.0 * r_t
    i_t = int(cols[k_t])
    R_c = _throat_curvature_radius(x, r, i_t, r_t)

    c_star = (p_c * A_t / mdot) if (mdot and mdot > 1e-9) else ideal_cstar(cfg)

    A = np.pi * rv * rv                       # local flow area (axisymmetric)
    area_ratio = A_t / np.maximum(A, 1e-30)

    half_gm1 = 0.5 * (g - 1.0)
    M2 = Mv * Mv
    stag = 1.0 + half_gm1 * M2
    sigma = 1.0 / (
        np.power(0.5 * (T_wall / T0) * stag + 0.5, 0.68)
        * np.power(stag, 0.12)
    )

    h_g = (0.026 / D_t ** 0.2
           * (mu ** 0.2 * cp / Pr ** 0.6)
           * (p_c / c_star) ** 0.8
           * (D_t / R_c) ** 0.1
           * np.power(area_ratio, 0.9)
           * sigma)

    r_rec = Pr ** (1.0 / 3.0)                  # turbulent recovery factor
    T_aw = T0 * (1.0 + r_rec * half_gm1 * M2) / stag
    q = h_g * (T_aw - T_wall)

    return {
        "valid": True,
        "x": xv, "r": rv, "M": Mv,
        "h_g": h_g, "q": q, "T_aw": T_aw,
        "throat_radius": r_t, "throat_area": A_t, "D_t": D_t, "R_c": R_c,
        "c_star": c_star, "x_throat": float(xv[k_t]),
        "q_throat": float(q[k_t]), "q_max": float(np.nanmax(q)),
        "T_wall": T_wall,
    }


if __name__ == "__main__":  # pragma: no cover - sanity check
    import os
    import tempfile

    from .mask import load_mask
    from .sample import make_nozzle_png

    p = os.path.join(tempfile.gettempdir(), "bartz_test.png")
    make_nozzle_png(p, 400, 240)
    m = load_mask(p, 0.0005, smooth=False)
    ct = m.cell_type[2:-2, 2:-2]
    axis_row = ct.shape[0] / 2.0 - 0.5
    # synthetic Mach: 0 in chamber rising to ~3 at exit (rough), for magnitude check
    ny, nx = ct.shape
    xi = np.linspace(0, 1, nx)[None, :]
    mach = np.broadcast_to(0.2 + 3.0 * xi, ct.shape).astype(np.float32).copy()
    mach[ct == 1] = np.nan
    cfg = SimConfig(inlet_p0=5e6, inlet_T0=3300.0, gamma=1.21, R_gas=346.0)
    res = bartz_heat_flux(ct, mach, m.dx, axis_row, cfg, T_wall=800.0)
    if res["valid"]:
        print(f"throat D_t = {res['D_t']*1000:.2f} mm  R_c = {res['R_c']*1000:.2f} mm")
        print(f"c* = {res['c_star']:.1f} m/s")
        print(f"q_throat = {res['q_throat']/1e6:.2f} MW/m^2  "
              f"q_max = {res['q_max']/1e6:.2f} MW/m^2")
        print(f"h_g range = [{np.nanmin(res['h_g']):.0f}, "
              f"{np.nanmax(res['h_g']):.0f}] W/m^2/K")
        print("heatflux.py self-test OK")
    else:
        print("heatflux.py self-test: no throat found (FAIL)")
