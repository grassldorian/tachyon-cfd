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
def _round_corners(x, r, idxs, radius: float, n_arc: int = 16):
    """Replace the sharp interior vertices in ``idxs`` with tangent circular
    fillets of radius ``radius`` mm; every other vertex passes through
    untouched. The tangent length is clamped to 45% of each adjoining segment,
    so neighbouring fillets (the chamber corner and the throat) can never
    overlap even at large radii, and the fillet degrades gracefully instead of
    inverting the wall. Returns the resampled (x, r)."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(r, float)])
    if radius <= 0.0:
        return pts[:, 0], pts[:, 1]
    want = {i for i in idxs if 0 < i < len(pts) - 1}
    out = [pts[0]]
    for j in range(1, len(pts) - 1):
        V = pts[j]
        if j not in want:
            out.append(V)
            continue
        A, B = pts[j - 1], pts[j + 1]
        u, w = A - V, B - V
        lu, lw = float(np.hypot(*u)), float(np.hypot(*w))
        if lu < 1e-9 or lw < 1e-9:
            out.append(V)
            continue
        u, w = u / lu, w / lw
        alpha = math.acos(float(np.clip(np.dot(u, w), -1.0, 1.0)))
        if alpha < 1e-3 or alpha > math.pi - 1e-3:   # already straight
            out.append(V)
            continue
        half = 0.5 * alpha
        t = min(radius / math.tan(half), 0.45 * lu, 0.45 * lw)
        reff = t * math.tan(half)                    # radius after clamping
        T1, T2 = V + u * t, V + w * t
        bis = u + w
        bis = bis / float(np.hypot(*bis))
        C = V + bis * (reff / math.sin(half))        # fillet-circle centre
        a1 = math.atan2(T1[1] - C[1], T1[0] - C[0])
        a2 = math.atan2(T2[1] - C[1], T2[0] - C[0])
        da = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi   # short way
        ang = a1 + da * np.linspace(0.0, 1.0, n_arc)
        out.extend(np.column_stack([C[0] + reff * np.cos(ang),
                                    C[1] + reff * np.sin(ang)]))
    out.append(pts[-1])
    P = np.asarray(out, dtype=float)
    return P[:, 0], P[:, 1]


def build_contour(geom: dict, nozzle_type: str = "Conical (15°)",
                  fillet_mm: float = 0.0):
    """Upper engine-wall contour (x, r) in mm, plus station keys.

    ``fillet_mm`` > 0 rounds the two structural corners — the chamber /
    converging-cone junction and (conical only) the throat — with tangent
    circular fillets of that radius. Rao/Bell nozzles already carry a smooth
    tangent throat arc, so only their chamber corner is rounded."""
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
        corner_idx = [1]                       # throat is a tangent arc already
    else:
        x3 = lc + l_conv + ln
        x = np.array([x0, x1, x2, x3])
        r = np.array([rc, rc, rt, re])
        corner_idx = [1, 2]                     # chamber junction + sharp throat
    if fillet_mm > 0.0 and l_conv > 1e-6:
        x, r = _round_corners(x, r, corner_idx, fillet_mm)
    return x, r, dict(x_inj=x0, x_chend=x1, x_throat=x2, x_exit=x3,
                      rc=rc, rt=rt, re=re)


# --------------------------------------------------------------------------- #
#  Rasterizer -> Tachyon mask
# --------------------------------------------------------------------------- #
WALL = (0, 0, 0)
FLOW = (255, 255, 255)
INLET = (0, 0, 255)
OUTLET = (255, 0, 0)


def _polygon_sdf(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance from the points (X, Y) to a closed polygon.

    ``poly`` is (N, 2); the closing edge is implicit. Negative inside.
    Exact distance to the boundary; inside/outside via even-odd ray casting.
    """
    P = np.asarray(poly, dtype=np.float64)
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])
    d2 = np.full(X.shape, np.inf, dtype=np.float64)
    inside = np.zeros(X.shape, dtype=bool)
    for k in range(len(P) - 1):
        ax_, ay_ = P[k]
        bx_, by_ = P[k + 1]
        dx_, dy_ = bx_ - ax_, by_ - ay_
        L2 = dx_ * dx_ + dy_ * dy_
        if L2 < 1e-18:
            continue
        t = np.clip(((X - ax_) * dx_ + (Y - ay_) * dy_) / L2, 0.0, 1.0)
        qx = ax_ + t * dx_
        qy = ay_ + t * dy_
        d2 = np.minimum(d2, (X - qx) ** 2 + (Y - qy) ** 2)
        # even-odd ray cast along +x
        cond = (ay_ > Y) != (by_ > Y)
        if np.any(cond):
            xint = ax_ + (Y - ay_) * dx_ / (dy_ if abs(dy_) > 1e-15 else 1e-15)
            inside ^= cond & (X < xint)
    d = np.sqrt(d2)
    return np.where(inside, -d, d).astype(np.float32)


def rasterize_mask(geom: dict, nozzle_type: str = "Conical (15°)", *,
                   engine_px: int = 620, plume_factor: float = 1.6,
                   margin_factor: float = 0.30,
                   wall_mm: float | None = None, add_inlet: bool = True,
                   inlet_frac: float = 0.75, analytic: bool = False,
                   enclose: bool = True, half: bool = False,
                   expand_deg: float = 0.0, fillet_mm: float = 0.0):
    """Render the engine as a Tachyon mask image (full axisymmetric section).

    Returns (rgb uint8 array (H, W, 3), info dict). ``info`` carries
    ``meters_per_pixel`` so the physical size is preserved in the solver, plus
    the throat resolution and grid size for the UI. With ``analytic=True``
    (used on send-to-solver; costs ~a second) ``info['node_phi']`` holds the
    exact signed distance from the analytic wall contour at every mesh node —
    the solver's cut-cell surface then follows the true curve with zero
    rasterization ripple.
    """
    from PIL import Image, ImageDraw

    x, r, key = build_contour(geom, nozzle_type, fillet_mm=fillet_mm)
    L = float(key["x_exit"])                       # engine length [mm]
    rc, rt, re = key["rc"], key["rt"], key["re"]
    rmax = max(rc, re)
    if wall_mm is None:
        wall_mm = max(0.05 * rmax, 2.0)            # cosmetic wall thickness
    px_per_mm = engine_px / max(L, 1e-6)

    face_mm = 2.0 * wall_mm                        # injector face plate thickness
    x_off = face_mm + max(0.06 * L, 4.0)           # left farfield + face room
    plume = plume_factor * L
    # radial white space above/below the engine, as a multiple of the biggest
    # engine radius — enlarge for high-altitude plumes that balloon radially
    margin_r = max(max(margin_factor, 0.05) * rmax, 6.0)
    W = int(round((x_off + L + plume) * px_per_mm))
    H = int(round(2.0 * (rmax + wall_mm + margin_r) * px_per_mm))
    H += H % 2                                      # even -> symmetric axis
    axis_y = H / 2.0

    # 4x supersampled rendering, box-downsampled: edge pixels then carry the
    # true sub-pixel wall fraction (anti-aliased coverage), which the cut-cell
    # level set in mask.py reads directly -> the reconstructed wall follows
    # the analytic contour instead of rippling around a binary staircase.
    SS = 4
    ss = float(SS)

    def px(xmm, rmm):
        return ((xmm + x_off) * px_per_mm * ss,
                (axis_y - rmm * px_per_mm) * ss)

    img = Image.new("RGB", (W * SS, H * SS), FLOW)
    d = ImageDraw.Draw(img)

    if enclose:
        # solid block everywhere beside/behind the engine, up to the exit
        # plane: the ambient pocket around the engine body is a resonating
        # cavity (startup blasts and acoustics ricochet between the engine
        # and the farfield edges); filling it leaves only bore + plume fluid.
        bx1, _ = px(L, 0.0)
        d.rectangle([0, 0, bx1, H * SS - 1], fill=WALL)
        # carve the bore (flow path) back out of the block
        bore = ([px(xx, rr) for xx, rr in zip(x, r)]
                + [px(xx, -rr) for xx, rr in zip(x, r)][::-1])
        d.polygon(bore, fill=FLOW)
    else:
        # free-standing engine: wall bands around the bore + injector plate
        for sgn in (+1, -1):
            inner = [px(xx, sgn * rr) for xx, rr in zip(x, r)]
            outer = [px(xx, sgn * (rr + wall_mm)) for xx, rr in zip(x, r)][::-1]
            d.polygon(inner + outer, fill=WALL)
        # injector face: solid plate that IS part of the engine — spans the
        # full chamber diameter (+ walls) at x in [-face_mm, 0] so the chamber
        # volume is untouched and the plate merges with the wall bands at x=0.
        fx0, fy0 = px(-face_mm, rc + wall_mm)
        fx1, fy1 = px(0.0, -(rc + wall_mm))
        d.rectangle([fx0, fy0, fx1, fy1], fill=WALL)

    # pressure inlet: blue opening set INTO the face plate, on the chamber
    # side only — the outer half of the plate stays black, so the inlet feeds
    # the chamber and is closed toward the outside (never open on both sides).
    if add_inlet:
        ir = max(inlet_frac, 0.05) * rc
        ix0, iy0 = px(-0.5 * face_mm, ir)
        ix1, iy1 = px(0.0, -ir)
        d.rectangle([ix0, iy0, ix1, iy1], fill=INLET)

    # pressure outlet: red strip down the downstream (right) edge
    rw = max(2, int(round(0.004 * W)))
    d.rectangle([(W - rw) * SS, 0, W * SS - 1, H * SS - 1], fill=OUTLET)

    # expanding wall section (altitude-cell / diffuser): downstream of the exit
    # the plume flows into a diverging duct instead of a parallel channel — the
    # walls flare out at ``expand_deg`` from the nozzle-exit lip. Drawn AFTER
    # the outlet so the solid wedges trim the red strip back to the duct mouth.
    if expand_deg > 0.0:
        slope = math.tan(math.radians(min(expand_deg, 60.0)))
        x_end = W / px_per_mm - x_off              # right edge in engine mm
        r0 = re                                    # flush with the nozzle bore
        Rf = rmax + wall_mm + margin_r + 20.0      # beyond the domain edge
        for sgn in (+1, -1):
            wedge = [px(L, sgn * r0),
                     px(x_end + 20.0, sgn * (r0 + slope * (x_end + 20.0 - L))),
                     px(x_end + 20.0, sgn * Rf),
                     px(L, sgn * Rf)]
            d.polygon(wedge, fill=WALL)

    img = img.resize((W, H), Image.Resampling.BOX)  # -> coverage grays
    rgb = np.asarray(img, dtype=np.uint8)
    if half:
        # upper half only, axis along the bottom image edge (a symmetry plane
        # in the solver): exact mirror symmetry by construction + 2x fewer
        # cells. The full-section render is simply cropped at the axis row.
        rgb = np.ascontiguousarray(rgb[:H // 2])
    info = dict(meters_per_pixel=1.0e-3 / px_per_mm,
                px_per_mm=px_per_mm, nx=W, ny=rgb.shape[0],
                throat_px=2.0 * rt * px_per_mm,
                exit_px=2.0 * re * px_per_mm,
                axis_location="bottom" if half else "center")

    if analytic:
        # ---- exact node level set from the analytic contour ----
        # node (J, I) sits at pixel corner (x=I, y=J) in final-pixel coords
        def fx(xmm):
            return (xmm + x_off) * px_per_mm

        def fy(rmm):
            return axis_y - rmm * px_per_mm

        Xn, Yn = np.meshgrid(np.arange(W + 1, dtype=np.float64),
                             np.arange(H + 1, dtype=np.float64))
        xs_px = fx(x)                              # bore contour in px
        sdf_inlet = None
        if add_inlet:
            ir = max(inlet_frac, 0.05) * rc
            # extend a hair past x=0 so the recess opens cleanly into the
            # chamber (no coincident boundaries)
            inlet = np.array([[fx(-0.5 * face_mm), fy(ir)],
                              [fx(0.0) + 0.75,     fy(ir)],
                              [fx(0.0) + 0.75,     fy(-ir)],
                              [fx(-0.5 * face_mm), fy(-ir)]])
            sdf_inlet = _polygon_sdf(Xn, Yn, inlet)
        if enclose:
            # solid = block (up to the exit plane) minus bore minus inlet
            block = np.array([[-50.0,   -50.0],
                              [fx(L),   -50.0],
                              [fx(L),   H + 50.0],
                              [-50.0,   H + 50.0]])
            bore = np.vstack([np.column_stack([xs_px, fy(r)]),
                              np.column_stack([xs_px, fy(-r)])[::-1]])
            sdf = np.maximum(_polygon_sdf(Xn, Yn, block),
                             -_polygon_sdf(Xn, Yn, bore))
            if sdf_inlet is not None:
                sdf = np.maximum(sdf, -sdf_inlet)
        else:
            band_polys = []
            for sgn in (+1, -1):
                bore = np.column_stack([xs_px, fy(sgn * r)])
                outer = np.column_stack([xs_px, fy(sgn * (r + wall_mm))])[::-1]
                band_polys.append(np.vstack([bore, outer]))
            face = np.array([[fx(-face_mm), fy(rc + wall_mm)],
                             [fx(0.0),      fy(rc + wall_mm)],
                             [fx(0.0),      fy(-(rc + wall_mm))],
                             [fx(-face_mm), fy(-(rc + wall_mm))]])
            sdf = np.minimum(_polygon_sdf(Xn, Yn, band_polys[0]),
                             _polygon_sdf(Xn, Yn, band_polys[1]))
            sdf_face = _polygon_sdf(Xn, Yn, face)
            if sdf_inlet is not None:
                sdf_face = np.maximum(sdf_face, -sdf_inlet)
            sdf = np.minimum(sdf, sdf_face)
        if expand_deg > 0.0:
            # add the two diverging duct walls to the solid (union -> minimum)
            slope = math.tan(math.radians(min(expand_deg, 60.0)))
            x_end = W / px_per_mm - x_off
            r0 = re                                # flush with the nozzle bore
            Rf = rmax + wall_mm + margin_r + 20.0
            re2 = r0 + slope * (x_end + 20.0 - L)
            for sgn in (+1, -1):
                wedge = np.array([[fx(L),          fy(sgn * r0)],
                                  [fx(x_end + 20), fy(sgn * re2)],
                                  [fx(x_end + 20), fy(sgn * Rf)],
                                  [fx(L),          fy(sgn * Rf)]])
                sdf = np.minimum(sdf, _polygon_sdf(Xn, Yn, wedge))
        if half:
            sdf = np.ascontiguousarray(sdf[:H // 2 + 1])
        info["node_phi"] = sdf.astype(np.float32)   # phi > 0 in fluid
    return rgb, info
