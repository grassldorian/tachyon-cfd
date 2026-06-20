"""Parametric rocket-engine geometry + ideal-rocket sizing, and a rasterizer
that turns the geometry into a Tachyon mask PNG (black wall / white flow /
blue pressure inlet / red outlet).

The 1-D isentropic model and the Rao bell contour are ported from the standalone
"Rocket Engine Designer" (tkinter prototype) into pure, GUI-free functions so the
PySide6 designer tab can reuse them and feed the CFD solver.
"""
from __future__ import annotations

import math

import numpy as np

R_UNIVERSAL = 8314.462        # J/(kmol K)
G0 = 9.80665                  # m/s^2
P_SEA_LEVEL = 101325.0        # Pa

# Representative ideal-combustion values near the optimum mixture ratio.
#   Tc [K], gamma [-], M [kg/kmol], of (O/F mass), Lstar [m]
PROPELLANTS = {
    "LOX / Ethanol": dict(Tc=3450.0, gamma=1.21, M=23.5, of=1.6,  Lstar=1.0),
    "LOX / RP-1":    dict(Tc=3670.0, gamma=1.22, M=23.3, of=2.56, Lstar=1.1),
    "LOX / LH2":     dict(Tc=3350.0, gamma=1.22, M=13.0, of=5.5,  Lstar=0.85),
    "LOX / CH4":     dict(Tc=3550.0, gamma=1.17, M=21.4, of=3.6,  Lstar=1.1),
}


# --------------------------------------------------------------------------- #
#  Thermodynamics
# --------------------------------------------------------------------------- #
def ambient_pressure(altitude_km: float) -> float:
    """US-Standard-Atmosphere-ish ambient pressure [Pa] from altitude [km]."""
    h = max(0.0, altitude_km) * 1000.0
    if h <= 11000.0:
        T = 288.15 - 0.0065 * h
        return P_SEA_LEVEL * (T / 288.15) ** 5.2559
    p11 = P_SEA_LEVEL * (216.65 / 288.15) ** 5.2559
    return p11 * math.exp(-G0 * (h - 11000.0) / (287.05 * 216.65))


def gamma_func(g: float) -> float:
    """Vandenkerckhove function Gamma(gamma)."""
    return math.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area_ratio(eps: float, g: float) -> float:
    """Supersonic Mach for a given area ratio Ae/At via bisection."""
    if eps <= 1.0:
        return 1.0

    def area_ratio(M):
        return (1.0 / M) * ((2.0 / (g + 1.0)) *
                            (1.0 + 0.5 * (g - 1.0) * M * M)) ** \
                            ((g + 1.0) / (2.0 * (g - 1.0)))

    lo, hi = 1.0000001, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_ratio(mid) > eps:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  Rao thrust-optimised bell
# --------------------------------------------------------------------------- #
_RAO_EPS = [3.5,  5.0,  10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 100.0]
_RAO_TN  = [20.0, 22.5, 26.0, 27.7, 28.8, 30.3, 31.2, 31.8, 33.5]
_RAO_TE  = [13.5, 12.0, 8.8,  7.7,  7.0,  6.0,  5.4,  5.0,  4.0]
THROAT_ARC = 0.382
BELL_FRACTION = 0.80


def rao_angles(eps: float):
    e = min(max(eps, _RAO_EPS[0]), _RAO_EPS[-1])
    tn = float(np.interp(e, _RAO_EPS, _RAO_TN))
    te = float(np.interp(e, _RAO_EPS, _RAO_TE))
    return math.radians(tn), math.radians(te)


def rao_bell_points(rt: float, re: float, ln: float, eps: float, n: int = 64):
    """Rao bell wall contour, x from the throat plane. Returns (x, r)."""
    thn, the = rao_angles(eps)
    R2 = THROAT_ARC * rt
    a = np.linspace(0.0, thn, 18)
    xa = R2 * np.sin(a)
    ya = rt + R2 * (1.0 - np.cos(a))
    Nx, Ny = xa[-1], ya[-1]
    Ex, Ey = max(ln, Nx * 1.2), re
    m1, m2 = math.tan(thn), math.tan(the)
    Px = ((Ey - m2 * Ex) - (Ny - m1 * Nx)) / (m1 - m2)
    Py = Ny + m1 * (Px - Nx)
    t = np.linspace(0.0, 1.0, n)
    xb = (1 - t) ** 2 * Nx + 2 * (1 - t) * t * Px + t ** 2 * Ex
    yb = (1 - t) ** 2 * Ny + 2 * (1 - t) * t * Py + t ** 2 * Ey
    return np.concatenate([xa, xb[1:]]), np.concatenate([ya, yb[1:]])


def divergence_efficiency(nozzle_type: str, eps: float) -> float:
    if nozzle_type.startswith("Rao") or nozzle_type.startswith("Bell"):
        _, the = rao_angles(eps)
        return 0.5 * (1.0 + math.cos(the))
    return 0.5 * (1.0 + math.cos(math.radians(15.0)))


# --------------------------------------------------------------------------- #
#  Ideal-rocket performance + sizing
# --------------------------------------------------------------------------- #
def solve_engine(geom: dict, prop: dict, pc: float, pa: float,
                 lam: float = 1.0) -> dict:
    """1-D isentropic ideal-rocket model. geom: throat_d, exit_d [m]."""
    g, Tc = prop["gamma"], prop["Tc"]
    R = R_UNIVERSAL / prop["M"]
    dt, de = geom["throat_d"], geom["exit_d"]
    At = math.pi * dt * dt / 4.0
    Ae = math.pi * de * de / 4.0
    eps = Ae / At
    cstar = math.sqrt(R * Tc) / gamma_func(g)
    mdot = pc * At / cstar
    Me = mach_from_area_ratio(eps, g)
    pe = pc * (1.0 + 0.5 * (g - 1.0) * Me * Me) ** (-g / (g - 1.0))
    Te = Tc / (1.0 + 0.5 * (g - 1.0) * Me * Me)
    ve = Me * math.sqrt(g * R * Te)
    term = (2.0 * g * g / (g - 1.0)) * \
           (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)) * \
           (1.0 - (pe / pc) ** ((g - 1.0) / g))
    cf_mom = lam * math.sqrt(max(term, 0.0))
    cf = cf_mom + (pe - pa) / pc * eps
    thrust = cf * pc * At
    isp = thrust / (mdot * G0)
    return dict(At=At, Ae=Ae, eps=eps, cstar=cstar, mdot=mdot, Me=Me,
                pe=pe, Te=Te, ve=ve * lam, cf=cf, thrust=thrust, isp=isp,
                lam=lam)


def optimize_geometry(prop: dict, pc: float, pa: float, target_thrust: float,
                      nozzle_type: str = "Conical (15°)") -> dict:
    """Size a fresh geometry: perfect expansion at this altitude + thrust."""
    g = prop["gamma"]
    pa = max(pa, 500.0)
    Me = math.sqrt((2.0 / (g - 1.0)) * ((pc / pa) ** ((g - 1.0) / g) - 1.0))
    eps = (1.0 / Me) * ((2.0 / (g + 1.0)) *
                        (1.0 + 0.5 * (g - 1.0) * Me * Me)) ** \
                        ((g + 1.0) / (2.0 * (g - 1.0)))
    term = (2.0 * g * g / (g - 1.0)) * \
           (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)) * \
           (1.0 - (pa / pc) ** ((g - 1.0) / g))
    lam = divergence_efficiency(nozzle_type, eps)
    cf = lam * math.sqrt(max(term, 0.0))
    At = target_thrust / (cf * pc)
    dt = math.sqrt(4.0 * At / math.pi)
    de = dt * math.sqrt(eps)
    dt_cm = dt * 100.0
    contraction = min(max(8.0 * dt_cm ** (-0.6) + 1.25, 2.0), 12.0)
    dc = dt * math.sqrt(contraction)
    Vc = prop["Lstar"] * At
    Ac = math.pi * dc * dc / 4.0
    l_conv = (dc - dt) / 2.0 / math.tan(math.radians(30.0))
    rc, rt = dc / 2.0, dt / 2.0
    v_conv = math.pi * l_conv / 3.0 * (rc * rc + rc * rt + rt * rt)
    lc = max((Vc - v_conv) / Ac, 0.25 * dc)
    cone_len = (de - dt) / 2.0 / math.tan(math.radians(15.0))
    ln = (BELL_FRACTION * cone_len
          if (nozzle_type.startswith("Rao") or nozzle_type.startswith("Bell"))
          else cone_len)
    return dict(chamber_l=lc, chamber_d=dc, nozzle_l=ln, exit_d=de, throat_d=dt)


# --------------------------------------------------------------------------- #
#  Wall contour
# --------------------------------------------------------------------------- #
def build_contour(geom: dict, nozzle_type: str = "Conical (15°)"):
    """Upper engine-wall contour (x, r) in mm, plus station keys."""
    lc = geom["chamber_l"] * 1000.0
    dc = geom["chamber_d"] * 1000.0
    ln = geom["nozzle_l"] * 1000.0
    de = geom["exit_d"] * 1000.0
    dt = geom["throat_d"] * 1000.0
    rc, rt, re = dc / 2.0, dt / 2.0, de / 2.0
    l_conv = (rc - rt) / math.tan(math.radians(30.0)) if rc > rt else 0.0
    x0, x1, x2 = 0.0, lc, lc + l_conv
    if nozzle_type.startswith("Rao") or nozzle_type.startswith("Bell"):
        eps = (re / rt) ** 2 if rt > 0 else 1.0
        xb, rb = rao_bell_points(rt, re, ln, eps)
        x = np.concatenate([[x0, x1], x2 + xb])
        r = np.concatenate([[rc, rc], rb])
        x3 = x[-1]
    else:
        x3 = lc + l_conv + ln
        x = np.array([x0, x1, x2, x3])
        r = np.array([rc, rc, rt, re])
    return x, r, dict(x_inj=x0, x_chend=x1, x_throat=x2, x_exit=x3,
                      rc=rc, rt=rt, re=re)


# --------------------------------------------------------------------------- #
#  Rasterizer -> Tachyon mask
# --------------------------------------------------------------------------- #
WALL = (0, 0, 0)
FLOW = (255, 255, 255)
INLET = (0, 0, 255)
OUTLET = (255, 0, 0)


def rasterize_mask(geom: dict, nozzle_type: str = "Conical (15°)", *,
                   engine_px: int = 620, plume_factor: float = 1.6,
                   wall_mm: float | None = None, add_inlet: bool = True,
                   inlet_frac: float = 0.75):
    """Render the engine as a Tachyon mask image (full axisymmetric section).

    Returns (rgb uint8 array (H, W, 3), info dict). ``info`` carries
    ``meters_per_pixel`` so the physical size is preserved in the solver, plus
    the throat resolution and grid size for the UI.
    """
    from PIL import Image, ImageDraw

    x, r, key = build_contour(geom, nozzle_type)
    L = float(key["x_exit"])                       # engine length [mm]
    rc, rt, re = key["rc"], key["rt"], key["re"]
    rmax = max(rc, re)
    if wall_mm is None:
        wall_mm = max(0.05 * rmax, 2.0)            # cosmetic wall thickness
    px_per_mm = engine_px / max(L, 1e-6)

    x_off = wall_mm + max(0.06 * L, 4.0)           # left farfield + injector room
    plume = plume_factor * L
    margin_r = max(0.30 * rmax, 6.0)
    W = int(round((x_off + L + plume) * px_per_mm))
    H = int(round(2.0 * (rmax + wall_mm + margin_r) * px_per_mm))
    H += H % 2                                      # even -> symmetric axis
    axis_y = H / 2.0

    def px(xmm, rmm):
        return (xmm + x_off) * px_per_mm, axis_y - rmm * px_per_mm

    img = Image.new("RGB", (W, H), FLOW)
    d = ImageDraw.Draw(img)

    # upper + lower nozzle walls (band between the bore contour and contour+wall)
    for sgn in (+1, -1):
        inner = [px(xx, sgn * rr) for xx, rr in zip(x, r)]
        outer = [px(xx, sgn * (rr + wall_mm)) for xx, rr in zip(x, r)][::-1]
        d.polygon(inner + outer, fill=WALL)

    # injector face: black bar closing the chamber's left end
    bx0, _ = px(0.0, 0.0)
    bx1, _ = px(wall_mm, 0.0)
    d.rectangle([bx0, axis_y - (rc + wall_mm) * px_per_mm,
                 bx1, axis_y + (rc + wall_mm) * px_per_mm], fill=WALL)

    # pressure inlet: blue opening centred on the injector face
    if add_inlet:
        ir = max(inlet_frac, 0.05) * rc
        d.rectangle([bx0, axis_y - ir * px_per_mm,
                     bx1, axis_y + ir * px_per_mm], fill=INLET)

    # pressure outlet: red strip down the downstream (right) edge
    rw = max(2, int(round(0.004 * W)))
    d.rectangle([W - rw, 0, W - 1, H - 1], fill=OUTLET)

    rgb = np.asarray(img, dtype=np.uint8)
    info = dict(meters_per_pixel=1.0e-3 / px_per_mm,
                px_per_mm=px_per_mm, nx=W, ny=H,
                throat_px=2.0 * rt * px_per_mm,
                exit_px=2.0 * re * px_per_mm)
    return rgb, info
