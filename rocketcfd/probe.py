"""Line probes and 1-D profiles through a 2-D solver field.

All field arrays follow the GUI's interior convention: shape ``(ny, nx)`` with
NaN inside walls.  The world coordinates of the centre of cell ``(j, i)`` are

    x = (i + 0.5) * dx
    y = (j + 0.5) * dx - y_off

where ``y_off`` is the world-y of the symmetry axis, so in axisymmetric mode
``y`` is the radius measured from the axis.  This matches
``MainWindow._update_geometry`` / ``on_mouse_move`` in the GUI, so a point the
user clicks in the field view maps to the same sample here.

Nothing in this module touches the GPU; it operates on the CPU snapshot
fields, so it is cheap to call interactively or from the report generator.
"""
from __future__ import annotations

import numpy as np

from .mask import FLUID, WALL


# ----------------------------------------------------------------- sampling
def _bilinear_nan(field: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """NaN-aware bilinear sample of ``field`` at fractional indices.

    ``fx`` are column (x) indices, ``fy`` row (y) indices, both measured in
    cell-centre units (cell ``(j, i)`` centre is at ``fx = i``, ``fy = j``).
    Samples whose four neighbours are all NaN/out-of-range return NaN; partial
    coverage is renormalised over the finite corners so a probe that grazes a
    wall still returns the fluid value instead of NaN.
    """
    ny, nx = field.shape
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = fx - x0
    wy = fy - y0

    def corner(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        out = np.full(fx.shape, np.nan, dtype=np.float64)
        m = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        if m.any():
            out[m] = field[np.clip(yy, 0, ny - 1)[m], np.clip(xx, 0, nx - 1)[m]]
        return out

    v00 = corner(y0, x0)
    v01 = corner(y0, x0 + 1)
    v10 = corner(y0 + 1, x0)
    v11 = corner(y0 + 1, x0 + 1)
    w00 = (1 - wx) * (1 - wy)
    w01 = wx * (1 - wy)
    w10 = (1 - wx) * wy
    w11 = wx * wy
    vals = np.stack([v00, v01, v10, v11])
    ws = np.stack([w00, w01, w10, w11])
    fin = np.isfinite(vals)
    wsum = np.where(fin, ws, 0.0).sum(axis=0)
    vsum = np.where(fin, ws * np.nan_to_num(vals), 0.0).sum(axis=0)
    return np.where(wsum > 1e-9, vsum / np.maximum(wsum, 1e-12), np.nan)


def sample_line(field: np.ndarray, dx: float, p0, p1, n: int = 400,
                y_off: float = 0.0) -> dict:
    """Sample ``field`` along the world-space segment ``p0 -> p1`` (metres).

    Returns ``dict(s, x, y, values)`` where ``s`` is arc length from ``p0``
    in metres and ``values`` is the NaN-aware bilinear interpolation.
    """
    (x0, y0), (x1, y1) = p0, p1
    t = np.linspace(0.0, 1.0, int(max(n, 2)))
    xw = x0 + (x1 - x0) * t
    yw = y0 + (y1 - y0) * t
    fx = xw / dx - 0.5
    fy = (yw + y_off) / dx - 0.5
    values = _bilinear_nan(field, fx, fy)
    s = np.hypot(xw - x0, yw - y0)
    return {"s": s, "x": xw, "y": yw, "values": values}


def centerline(field: np.ndarray, dx: float, axis_row: float,
               y_off: float = 0.0, n: int | None = None) -> dict:
    """Axial profile of ``field`` along the symmetry axis (radius 0).

    ``axis_row`` is the axis position in interior row coordinates (the same
    half-integer the solver uses).  Returns ``dict(x, values)`` over the full
    axial extent.
    """
    ny, nx = field.shape
    if n is None:
        n = nx
    x0 = 0.5 * dx
    x1 = (nx - 0.5) * dx
    # world-y of the axis is 0 by construction of y_off, but pass it through
    # the same machinery so callers can override.
    out = sample_line(field, dx, (x0, 0.0), (x1, 0.0), n=n, y_off=y_off)
    return {"x": out["x"], "values": out["values"]}


# ----------------------------------------------------------- wall contour
def wall_contour(ct: np.ndarray, dx: float, axis_row: float,
                 side: str = "upper") -> dict:
    """Inner-wall radius of the nozzle versus axial position.

    Scans outward from the axis on the requested half and finds the wall cell
    closest to the axis in each column.  Returns ``dict(x, r, fluid_row)``:
    ``x`` axial centre [m], ``r`` inner-wall radius [m] (NaN where the column
    has no wall, e.g. downstream of the exit lip), and ``fluid_row`` the index
    of the fluid cell just inside the wall (-1 where none), for sampling the
    wall-adjacent flow state.
    """
    ny, nx = ct.shape
    ju = int(np.floor(axis_row))
    x = (np.arange(nx) + 0.5) * dx
    r = np.full(nx, np.nan, dtype=np.float64)
    fluid_row = np.full(nx, -1, dtype=np.int64)
    upper = side == "upper"
    for i in range(nx):
        if upper:
            col = ct[: ju + 1, i]
            walls = np.flatnonzero(col == WALL)
            if walls.size:
                jw = int(walls.max())               # closest to the axis
                r[i] = (axis_row - jw) * dx
                jf = jw + 1
                if jf <= ju and ct[jf, i] == FLUID:
                    fluid_row[i] = jf
        else:
            col = ct[ju + 1:, i]
            walls = np.flatnonzero(col == WALL)
            if walls.size:
                jw = ju + 1 + int(walls.min())
                r[i] = (jw - axis_row) * dx
                jf = jw - 1
                if jf > ju and ct[jf, i] == FLUID:
                    fluid_row[i] = jf
    return {"x": x, "r": r, "fluid_row": fluid_row}


def wall_pressure(field: np.ndarray, ct: np.ndarray, dx: float,
                  axis_row: float, side: str = "upper",
                  contour: dict | None = None) -> dict:
    """Distribution of a field (typically pressure) along the wall contour.

    Samples the fluid cell just inside the wall at every axial station.
    Returns ``dict(x, r, values)`` restricted to columns that actually have a
    wall with adjacent fluid.
    """
    if contour is None:
        contour = wall_contour(ct, dx, axis_row, side=side)
    fr = contour["fluid_row"]
    x = contour["x"]
    r = contour["r"]
    m = fr >= 0
    cols = np.flatnonzero(m)
    vals = np.full(cols.shape, np.nan, dtype=np.float64)
    for k, i in enumerate(cols):
        vals[k] = field[fr[i], i]
    return {"x": x[cols], "r": r[cols], "values": vals}


if __name__ == "__main__":  # pragma: no cover - smoke test
    import os
    import tempfile

    from .mask import load_mask
    from .sample import make_nozzle_png

    p = os.path.join(tempfile.gettempdir(), "probe_test.png")
    make_nozzle_png(p, 300, 200)
    m = load_mask(p, 0.001, smooth=False)
    ct = m.cell_type[2:-2, 2:-2]
    axis_row = ct.shape[0] / 2.0 - 0.5
    con = wall_contour(ct, m.dx, axis_row)
    finite = np.isfinite(con["r"])
    print(f"contour columns with wall: {finite.sum()} / {ct.shape[1]}")
    print(f"throat radius  ~ {np.nanmin(con['r']) * 1000:.2f} mm")
    print(f"chamber radius ~ {np.nanmax(con['r']) * 1000:.2f} mm")
    # fake a linear field to test sampling
    fld = np.fromfunction(lambda j, i: i.astype(float), ct.shape)
    fld[ct == WALL] = np.nan
    line = sample_line(fld, m.dx, (0.0, 0.0), ((ct.shape[1] - 1) * m.dx, 0.0))
    print(f"centerline sample finite frac: "
          f"{np.isfinite(line['values']).mean():.2f}")
    print("probe.py self-test OK")
