"""Engine performance: thrust, mass flow, specific impulse.

Thrust = net gas force on the engine structure, from
  (a) gauge pressure integrated over all fluid/wall faces, and
  (b) gauge pressure + injected momentum over the inlet faces (which stand in
      for the missing chamber back wall / injector face).
Viscous shear on walls is neglected (small for rocket nozzles, and the
boundary layer is wall-modeled anyway).

Planar mode returns per-meter-depth values (N/m, kg/(s*m)); axisymmetric mode
returns true 3D values (N, kg/s) using face areas 2*pi*r*dx. With the axis
through the image center the drawing contains the engine twice (both halves),
so totals are halved.
"""
from __future__ import annotations

import numpy as np

from .mask import FLUID, WALL, INLET

G0 = 9.80665  # standard gravity [m/s^2]


def thrust_convergence(history, tail_frac: float = 0.10,
                       tol: float = 0.005) -> tuple[bool, float]:
    """Is the thrust history flat enough to quote a steady-state number?

    Looks at the last `tail_frac` of the (step, F) history and measures the
    peak-to-peak variation relative to the mean. Returns (converged, rel_var);
    rel_var is NaN when there is not enough history to judge.
    """
    if not history or len(history) < 10:
        return False, float("nan")
    f = np.asarray([h[1] for h in history], dtype=np.float64)
    n_tail = max(5, int(len(f) * tail_frac))
    tail = f[-n_tail:]
    mean = float(np.mean(tail))
    if not np.isfinite(mean) or abs(mean) < 1e-30:
        return False, float("nan")
    rel = float((np.max(tail) - np.min(tail)) / abs(mean))
    return rel < tol, rel


def performance(p: np.ndarray, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                ct: np.ndarray, dx: float, p_amb: float,
                axisymmetric: bool, axis_row: float, axis_center: bool,
                apx: np.ndarray | None = None,
                apy: np.ndarray | None = None) -> dict:
    """All field arrays are interior (ny, nx); ct is the interior cell-type map.

    axis_row: axis position in interior row coordinates (half-integer).
    apx/apy: cut-cell face apertures (ny, nx+1)/(ny+1, nx). When given, the
    wall pressure force uses the smooth embedded surface (aperture
    differences); otherwise pixel faces are used.
    """
    ny, nx = ct.shape
    fl = ct == FLUID
    wa = ct == WALL
    inl = ct == INLET
    pg = np.nan_to_num(p, nan=p_amb) - p_amb
    rho = np.nan_to_num(rho, nan=0.0)
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

    rows = np.arange(ny, dtype=np.float64)
    if axisymmetric:
        ax = 2.0 * np.pi * np.abs(rows - axis_row) * dx * dx          # x-face area, row j
        # y-face between rows j and j+1 (used with index j)
        ay = 2.0 * np.pi * np.abs(rows + 0.5 - axis_row) * dx * dx
    else:
        ax = np.full(ny, dx)                                          # per meter depth
        ay = np.full(ny, dx)
    AX = ax[:, None]
    AY = ay[:, None]

    Fx = Fy = mdot = 0.0

    if apx is not None and apy is not None:
        # ---- smooth embedded surface: S_wall from aperture differences ----
        swx = apx[:, :-1] - apx[:, 1:]          # (ny, nx), + = wall to the east
        swy = apy[:-1, :] - apy[1:, :]
        Fx += float(np.sum((pg * swx * AX)[fl]))
        Fy += float(np.sum((pg * swy * AX)[fl]))   # segment radius ~ cell row
    else:
        # ---- pixel faces (legacy / smooth boundary disabled) ----
        m = fl[:, :-1] & wa[:, 1:]              # wall to the east, n = +x
        Fx += float(np.sum((pg[:, :-1] * AX)[m]))
        m = fl[:, 1:] & wa[:, :-1]              # wall to the west, n = -x
        Fx -= float(np.sum((pg[:, 1:] * AX)[m]))
        m = fl[:-1, :] & wa[1:, :]              # wall below (image), n = +y
        Fy += float(np.sum((pg[:-1, :] * AY[:-1])[m]))
        m = fl[1:, :] & wa[:-1, :]              # wall above, n = -y
        Fy -= float(np.sum((pg[1:, :] * AY[:-1])[m]))

    # ---- inlet faces: pressure + injected momentum (thrust only) ----
    m = inl[:, :-1] & fl[:, 1:]                 # inlet west of fluid, flow +x
    pf, rf = pg[:, 1:][m], rho[:, 1:][m]
    un = u[:, 1:][m]
    A = np.broadcast_to(AX, (ny, nx - 1))[m]
    Fx -= float(np.sum((pf + rf * un * un) * A))

    m = inl[:, 1:] & fl[:, :-1]                 # inlet east of fluid, flow -x
    pf, rf = pg[:, :-1][m], rho[:, :-1][m]
    un = -u[:, :-1][m]
    A = np.broadcast_to(AX, (ny, nx - 1))[m]
    Fx += float(np.sum((pf + rf * un * un) * A))

    m = inl[:-1, :] & fl[1:, :]                 # inlet above fluid, flow +y
    pf, rf = pg[1:, :][m], rho[1:, :][m]
    un = v[1:, :][m]
    A = np.broadcast_to(AY[:-1], (ny - 1, nx))[m]
    Fy -= float(np.sum((pf + rf * un * un) * A))

    m = inl[1:, :] & fl[:-1, :]                 # inlet below fluid, flow -y
    pf, rf = pg[:-1, :][m], rho[:-1, :][m]
    un = -v[:-1, :][m]
    A = np.broadcast_to(AY[:-1], (ny - 1, nx))[m]
    Fy += float(np.sum((pf + rf * un * un) * A))

    # ---- mass flow: median engine-interior axial flux across x-stations ----
    # Integrating rho*u over the injector face alone collapses to zero whenever
    # the chamber's small injector-face velocity momentarily reverses (acoustic
    # / fill transients) -> random Isp/mdot/c_eff dropouts. Instead integrate
    # rho*u only over the fluid contiguous with the axis (out to the first
    # wall), which excludes the surrounding ambient air, and take the median
    # over the walled columns. By conservation this engine-interior flux equals
    # mdot at every station from chamber to exit, so the median is robust to the
    # injector dropouts, pinched slivers and bell separation that corrupt any
    # single plane, while the ambient/plume entrainment is left out entirely.
    k = int(np.floor(axis_row))                 # last interior row index <= axis
    interior = np.zeros((ny, nx), dtype=bool)
    if k >= 0:                                  # side toward row 0
        seg = fl[:k + 1][::-1]
        interior[:k + 1] = np.cumprod(seg, axis=0, dtype=np.uint8).astype(bool)[::-1]
    if k + 1 < ny:                              # side toward row ny-1
        seg = fl[k + 1:]
        interior[k + 1:] = np.cumprod(seg, axis=0, dtype=np.uint8).astype(bool)
    flux_col = np.sum(rho * u * interior * ax[:, None], axis=0)
    walled = wa.any(axis=0) & interior.any(axis=0)
    if walled.any():
        mdot = abs(float(np.median(flux_col[walled])))
    else:                                       # free jet / no walls: inlet plane
        mdot = float(np.sum(rho * np.maximum(u, 0.0) * fl * ax[:, None]))

    if axisymmetric and axis_center:
        Fx *= 0.5; Fy *= 0.5; mdot *= 0.5       # drawing contains both halves

    F = float(np.hypot(Fx, Fy))
    # Isp / c_eff need an established (choked) mass flow. During fill/acoustic
    # transients mdot can be a hair above zero and divide F into a nonphysical
    # value, so gate on a generous chemical-rocket ceiling: c_eff is intensive
    # (size-independent) and never exceeds ~5 km/s for chemical propellants, so
    # > 15 km/s means the flow has not developed yet -> report 0.
    if mdot > 1e-9 and F / mdot < 1.5e4:
        ceff = F / mdot
        isp = ceff / G0
    else:
        ceff = isp = 0.0
    return {
        "Fx": Fx, "Fy": Fy, "F": F, "mdot": mdot, "Isp": isp, "c_eff": ceff,
        "force_unit": "N" if axisymmetric else "N/m",
        "mdot_unit": "kg/s" if axisymmetric else "kg/(s*m)",
    }
