"""Overset Phase 3: Chimera connectivity between the body-fitted near-body grid
and the Cartesian background — hole-cutting + donor/receptor interpolation.

Two transfers close the overset loop:
  * Cartesian *fringe* cells (the ring just outside the blanked near-wall hole)
    receive their state from the near-body grid (donors found by locating the
    fringe cell in the curvilinear grid; bilinear weights on the 4 surrounding
    cell centres).
  * Near-body *outer* cells (the outermost eta layers) receive their state from
    the Cartesian background (trivial uniform-grid location + bilinear weights).

Interpolation is bilinear on the dual (cell-centre) grid, so it is exact for
linear fields (2nd order) and a partition of unity (uniform -> uniform exactly).
CPU/numpy prototype; consumes the same (x, r) frame as the rest of Tachyon.
"""
from __future__ import annotations

import numpy as np


def cell_centers(Xn, Rn):
    """Cell-centre coordinates of a structured node grid (Mi,Mj) -> (Mi-1,Mj-1)."""
    return (0.25 * (Xn[:-1, :-1] + Xn[1:, :-1] + Xn[1:, 1:] + Xn[:-1, 1:]),
            0.25 * (Rn[:-1, :-1] + Rn[1:, :-1] + Rn[1:, 1:] + Rn[:-1, 1:]))


def _bilinear_st(px, py, cx, cy, iters=15):
    """Invert the bilinear map of a quad (corners cx,cy = [c00,c10,c11,c01]) for
    the (s,t) in [0,1]^2 with B(s,t)=(px,py). Newton from the centre."""
    s = t = 0.5
    for _ in range(iters):
        w00, w10, w11, w01 = (1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t
        Fx = w00 * cx[0] + w10 * cx[1] + w11 * cx[2] + w01 * cx[3] - px
        Fy = w00 * cy[0] + w10 * cy[1] + w11 * cy[2] + w01 * cy[3] - py
        dxs = -(1 - t) * cx[0] + (1 - t) * cx[1] + t * cx[2] - t * cx[3]
        dxt = -(1 - s) * cx[0] - s * cx[1] + s * cx[2] + (1 - s) * cx[3]
        dys = -(1 - t) * cy[0] + (1 - t) * cy[1] + t * cy[2] - t * cy[3]
        dyt = -(1 - s) * cy[0] - s * cy[1] + s * cy[2] + (1 - s) * cy[3]
        det = dxs * dyt - dxt * dys
        if abs(det) < 1e-300:
            break
        s -= (dyt * Fx - dxt * Fy) / det
        t -= (-dys * Fx + dxs * Fy) / det
    return s, t


# codes for the Cartesian blank map
SOLVE, HOLE, FRINGE = 0, 1, 2


class OversetConnectivity:
    def __init__(self, Xn, Rn, x0, y0, dx, ncx, ncy,
                 blank_frac: float = 0.55, fringe_layers: int = 2):
        self.bcx, self.bcy = cell_centers(np.asarray(Xn, float),
                                          np.asarray(Rn, float))
        self.Nbi, self.Nbj = self.bcx.shape
        self.x0, self.y0, self.dx = float(x0), float(y0), float(dx)
        self.ncx, self.ncy = int(ncx), int(ncy)
        self.ccx = x0 + (np.arange(ncx) + 0.5) * dx      # cartesian centre coords
        self.ccy = y0 + (np.arange(ncy) + 0.5) * dx
        self._bflat = np.column_stack([self.bcx.ravel(), self.bcy.ravel()])
        self._build(blank_frac, max(int(fringe_layers), 1))

    # -------------------------------------------------- location in body grid
    def locate_body(self, px, py, tol=1e-9):
        """Return (i, j, s, t) of the dual cell (centres (i,j),(i+1,j),
        (i+1,j+1),(i,j+1)) containing (px,py), or None if outside the grid."""
        k = int(np.argmin((self._bflat[:, 0] - px) ** 2
                          + (self._bflat[:, 1] - py) ** 2))
        i0, j0 = divmod(k, self.Nbj)
        for i in (i0 - 1, i0):
            for j in (j0 - 1, j0):
                if 0 <= i < self.Nbi - 1 and 0 <= j < self.Nbj - 1:
                    cx = (self.bcx[i, j], self.bcx[i + 1, j],
                          self.bcx[i + 1, j + 1], self.bcx[i, j + 1])
                    cy = (self.bcy[i, j], self.bcy[i + 1, j],
                          self.bcy[i + 1, j + 1], self.bcy[i, j + 1])
                    s, t = _bilinear_st(px, py, cx, cy)
                    if -tol <= s <= 1 + tol and -tol <= t <= 1 + tol:
                        return i, j, min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)
        return None

    # ------------------------------------------------------------- build maps
    def _build(self, blank_frac, fringe_layers):
        blank = np.full((self.ncx, self.ncy), SOLVE, np.uint8)
        # eta-fraction (0 at wall .. 1 at outer edge) of each located cartesian
        # cell; below blank_frac the near-body grid is authoritative -> hole
        for ci in range(self.ncx):
            for cj in range(self.ncy):
                loc = self.locate_body(self.ccx[ci], self.ccy[cj])
                if loc is None:
                    continue
                _, j, _, t = loc
                if (j + t) / (self.Nbj - 1) < blank_frac:
                    blank[ci, cj] = HOLE
        # fringe = SOLVE cells 4-adjacent to a hole
        h = blank == HOLE
        adj = np.zeros_like(h)
        adj[1:, :] |= h[:-1, :]; adj[:-1, :] |= h[1:, :]
        adj[:, 1:] |= h[:, :-1]; adj[:, :-1] |= h[:, 1:]
        cand = adj & (blank == SOLVE)

        # ---- donors for cartesian fringe (from body grid) ----
        # a candidate is a real fringe only if it locates in the near-body grid
        # (i.e. lies in the overlap); hole-adjacent cells past the band edge have
        # no donor and stay ordinary background cells.
        fr_idx, self.fr_don = [], []                     # (i,j, w[4]) per fringe
        for ci, cj in np.argwhere(cand):
            loc = self.locate_body(self.ccx[ci], self.ccy[cj], tol=0.02)
            if loc is None:
                continue
            i, j, s, t = loc
            blank[ci, cj] = FRINGE
            fr_idx.append((ci, cj))
            self.fr_don.append((i, j,
                                ((1 - s) * (1 - t), s * (1 - t),
                                 s * t, (1 - s) * t)))
        self.fr_idx = np.array(fr_idx, int).reshape(-1, 2)
        self.blank = blank

        # ---- receptors on the body outer edge (from cartesian) ----
        j_recv = range(max(self.Nbj - fringe_layers, 1), self.Nbj)
        self.br_idx = [(i, j) for j in j_recv for i in range(self.Nbi)]
        self.br_don = []                                 # (ci,cj, w[4]) per recv
        for i, j in self.br_idx:
            fi = (self.bcx[i, j] - self.x0) / self.dx - 0.5
            fj = (self.bcy[i, j] - self.y0) / self.dx - 0.5
            ci = int(np.clip(np.floor(fi), 0, self.ncx - 2))
            cj = int(np.clip(np.floor(fj), 0, self.ncy - 2))
            s = min(max(fi - ci, 0.0), 1.0)
            t = min(max(fj - cj, 0.0), 1.0)
            w = ((1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t)
            self.br_don.append((ci, cj, w))

    # -------------------------------------------------------- transfers
    def interp_to_cart_fringe(self, body_field):
        """Sample a body cell-centred field (Nbi,Nbj) at the cartesian fringe.
        Returns an array aligned with ``self.fr_idx``."""
        f = np.asarray(body_field)
        out = np.empty(len(self.fr_don), f.dtype)
        for n, (i, j, w) in enumerate(self.fr_don):
            out[n] = (w[0] * f[i, j] + w[1] * f[i + 1, j]
                      + w[2] * f[i + 1, j + 1] + w[3] * f[i, j + 1])
        return out

    def interp_to_body_outer(self, cart_field):
        """Sample a cartesian cell-centred field (ncx,ncy) at the body outer
        receptors. Returns an array aligned with ``self.br_idx``."""
        f = np.asarray(cart_field)
        out = np.empty(len(self.br_don), f.dtype)
        for n, (ci, cj, w) in enumerate(self.br_don):
            out[n] = (w[0] * f[ci, cj] + w[1] * f[ci + 1, cj]
                      + w[2] * f[ci + 1, cj + 1] + w[3] * f[ci, cj + 1])
        return out
