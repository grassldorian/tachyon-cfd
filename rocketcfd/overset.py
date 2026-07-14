"""Overset / Chimera near-body grids (Phase 1: the body-fitted mesh generator).

Tachyon is a Cartesian cut-cell solver (the Cart3D family). To get the
wall-normal-clustered boundary-layer mesh that structured/overset codes like
OVERFLOW use, we generate a *body-fitted* grid that hugs the wall and marches
normal into the fluid with cells packed at the surface (y+ ~ 1) and growing
geometrically outward. In the full overset scheme this near-body grid overlaps
the Cartesian background; here we build and quality-check the grid itself.

Coordinates are the engine (x, r) plane in metres (axisymmetric section), the
same frame the solver and designer use.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _smooth_open(a: np.ndarray, k: int) -> np.ndarray:
    """k passes of a [1 2 1]/4 filter with clamped (non-wrapping) ends."""
    a = np.asarray(a, float).copy()
    for _ in range(max(k, 0)):
        a[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
    return a


def _resample_arclength(x, r, n):
    """Resample the open contour to n points equally spaced in arc length."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(r)))])
    sn = np.linspace(0.0, s[-1], n)
    return np.interp(sn, s, x), np.interp(sn, s, r), float(s[-1])


def eta_geometric(n_eta: int, first_cell: float, growth: float) -> np.ndarray:
    """Wall-normal node distances (length n_eta+1, d[0]=0): a first cell of
    ``first_cell`` then geometric growth by ``growth`` — the classic
    boundary-layer spacing. Total thickness is the emergent d[-1]."""
    d = np.zeros(n_eta + 1)
    step = first_cell
    for j in range(1, n_eta + 1):
        d[j] = d[j - 1] + step
        step *= growth
    return d


def first_cell_for_yplus(y_plus: float, u_tau: float, nu: float) -> float:
    """Wall spacing [m] that lands the first node at a target y+.
    y+ = y * u_tau / nu  ->  y = y+ * nu / u_tau."""
    return y_plus * nu / max(u_tau, 1e-12)


# --------------------------------------------------------------------------- #
#  body-fitted grid
# --------------------------------------------------------------------------- #
def wall_normal_grid(x, r, *, n_eta: int = 48, first_cell: float = 2.0e-5,
                     growth: float = 1.15, n_xi: int = 240,
                     smooth_normals: int = 3):
    """Structured boundary-layer grid marched off the wall contour (x, r).

    Returns a dict with the node arrays ``X``/``R`` of shape (n_xi, n_eta+1)
    (index 0 along the wall = surface, last = outer edge), the surface data,
    and quality metrics. The march is along smoothed inward surface normals
    (toward the fluid, i.e. decreasing r for the upper wall); ``smooth_normals``
    Laplacian passes reduce grid skew where the wall curvature changes.
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)
    xw, rw, arclen = _resample_arclength(x, r, n_xi)

    # unit tangent (central differences on the resampled, uniform-s contour)
    tx = np.gradient(xw)
    tr = np.gradient(rw)
    tl = np.hypot(tx, tr)
    tl[tl < 1e-15] = 1.0
    tx /= tl
    tr /= tl
    # inward normal = rotate tangent -90 deg: (tr, -tx). For a wall contour
    # ordered by increasing x this points toward decreasing r (into the fluid).
    nx = _smooth_open(tr, smooth_normals)
    nr = _smooth_open(-tx, smooth_normals)
    nl = np.hypot(nx, nr)
    nl[nl < 1e-15] = 1.0
    nx /= nl
    nr /= nl

    d = eta_geometric(n_eta, first_cell, growth)
    X = xw[:, None] + nx[:, None] * d[None, :]
    R = rw[:, None] + nr[:, None] * d[None, :]

    metrics = grid_quality(X, R)
    info = dict(
        X=X, R=R, xw=xw, rw=rw, nx=nx, nr=nr, eta=d,
        n_xi=n_xi, n_eta=n_eta, first_cell=first_cell, growth=growth,
        thickness=float(d[-1]), arclen=arclen, **metrics,
    )
    return info


def grid_quality(X: np.ndarray, R: np.ndarray) -> dict:
    """Cell-quality metrics for a curvilinear grid: signed-area (Jacobian)
    positivity — a non-positive cell means grid lines have crossed (folded) —
    plus min orthogonality angle between the xi and eta families."""
    # quad cell (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1); signed area by the shoelace
    x0, r0 = X[:-1, :-1], R[:-1, :-1]
    x1, r1 = X[1:, :-1],  R[1:, :-1]
    x2, r2 = X[1:, 1:],   R[1:, 1:]
    x3, r3 = X[:-1, 1:],  R[:-1, 1:]
    area = 0.5 * ((x0 * r1 - x1 * r0) + (x1 * r2 - x2 * r1)
                  + (x2 * r3 - x3 * r2) + (x3 * r0 - x0 * r3))
    # the (xi, eta) node ordering is clockwise, so a well-formed grid has
    # consistently signed cells; folding is a cell whose sign flips against the
    # majority. Report areas in that consistent (positive) orientation.
    nz = area[area != 0.0]
    sign = float(np.sign(np.median(nz))) if nz.size else 1.0
    if sign == 0.0:
        sign = 1.0
    area *= sign
    min_area = float(area.min())
    folded = int((area <= 0.0).sum())

    # orthogonality: angle between eta-line and xi-line directions at nodes
    ex = np.gradient(X, axis=1)
    er = np.gradient(R, axis=1)
    sx = np.gradient(X, axis=0)
    sr = np.gradient(R, axis=0)
    dot = ex * sx + er * sr
    ne = np.hypot(ex, er)
    ns = np.hypot(sx, sr)
    cos = np.clip(dot / np.maximum(ne * ns, 1e-15), -1.0, 1.0)
    ang = np.degrees(np.arccos(np.abs(cos)))          # 90 deg = orthogonal
    min_orth = float(90.0 - np.abs(90.0 - ang).max())

    return dict(min_cell_area=min_area, folded_cells=folded,
                min_orthogonality_deg=min_orth)
