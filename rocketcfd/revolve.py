"""Pseudo-3D view: revolve the 2D axisymmetric field around its axis.

The 2D field f(x, r) is treated as an axisymmetric emission volume and
projected onto the screen by averaging along sight chords through the revolved
cylinder, with a slight camera tilt. The result looks like a photograph of a
glowing 3D plume (limb-darkened edges, bright core) while keeping the
original field units, so the colorbar stays meaningful as a
line-of-sight average.
"""
from __future__ import annotations

import numpy as np


def revolve_project(field: np.ndarray, axis_row: float,
                    tilt_deg: float = 18.0, n_samples: int = 48):
    """Project a 2D field (ny, nx, image convention) into a revolved 3D view.

    axis_row: axis position in row coordinates (half-integer, between rows).
    Returns (img, r_max): img has shape (2*r_max, nx); its row 0 sits at
    radial coordinate -r_max (i.e. world y = (axis_row + 0.5 - r_max) * dx).
    Cells outside the revolved volume are NaN.
    """
    ny, nx = field.shape
    ju = int(np.floor(axis_row))            # last row above the axis
    n_up, n_dn = ju + 1, ny - ju - 1
    r_max = max(n_up, n_dn)
    if r_max < 4:
        return field.copy(), None

    # radial emission profile: NaN-aware average of the two halves
    r_idx = np.arange(r_max)
    j_up = ju - r_idx
    j_dn = ju + 1 + r_idx
    up = field[np.clip(j_up, 0, ny - 1), :].astype(np.float32)
    dn = field[np.clip(j_dn, 0, ny - 1), :].astype(np.float32)
    up[j_up < 0] = np.nan
    dn[j_dn > ny - 1] = np.nan
    both = np.isfinite(up) & np.isfinite(dn)
    prof = np.where(both, 0.5 * (up + dn),
                    np.where(np.isfinite(up), up, dn))
    prof = np.nan_to_num(prof, nan=0.0)     # walls/empty -> dark

    # sight chords: for output radial coordinate Y, sample the volume at
    # depth z; with camera tilt the chord drifts in y as it goes deeper.
    tan_a = np.tan(np.radians(tilt_deg))
    K = int(n_samples)
    Y = np.arange(2 * r_max, dtype=np.float32) - (r_max - 0.5)
    z = ((np.arange(K, dtype=np.float32) + 0.5) / K * 2.0 - 1.0) * r_max
    Yk = Y[:, None] + z[None, :] * tan_a            # (2*r_max, K)
    Rk = np.sqrt(Yk * Yk + z[None, :] ** 2)
    w = (Rk < r_max).astype(np.float32)
    idx = np.clip(Rk - 0.5, 0.0, r_max - 1.0).astype(np.int32)

    out = np.zeros((2 * r_max, nx), dtype=np.float32)
    for k in range(K):                               # chunked gather: low memory
        out += prof[idx[:, k], :] * w[:, k][:, None]
    wsum = w.sum(axis=1)
    valid = wsum > 0.5
    out[valid] /= wsum[valid][:, None]
    out[~valid] = np.nan
    return out, r_max
