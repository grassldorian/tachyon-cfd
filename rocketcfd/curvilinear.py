"""Overset Phase 2: a curvilinear finite-volume compressible Euler solver.

CPU/numpy prototype that runs on the structured near-body (body-fitted) grid
from ``overset.wall_normal_grid``. Cell-centred finite volume with rotated-frame
HLLC fluxes, MUSCL (minmod) reconstruction and SSP-RK2 in time. In the full
overset scheme this solver lives on the wall-hugging grid and exchanges its
outer boundary with the Cartesian background.

Layout: ``ng=2`` ghost cells on every side; the node grid is linearly
extrapolated into the ghosts so every cell has a well-defined volume/faces and
boundary conditions only fill ghost *states*. Face area vectors point in +i/+j
(checked against the cell-centre offset), so the divergence is sign-correct on
either grid handedness. 2-D planar for now; axisymmetric sources, viscous /
turbulent fluxes and the GPU port are later increments.
"""
from __future__ import annotations

import numpy as np


def _minmod(a, b):
    return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


class CurvilinearEuler:
    def __init__(self, Xn, Rn, gamma: float = 1.4, ng: int = 2):
        self.gamma = float(gamma)
        self.ng = int(ng)
        self._extend_grid(np.asarray(Xn, float), np.asarray(Rn, float))
        self._metrics()
        self.ncx, self.ncy = self.Xg.shape[0] - 1, self.Xg.shape[1] - 1
        self.U = np.zeros((4, self.ncx, self.ncy))       # rho, ru, rv, E
        g = self.ng
        self.ii = slice(g, self.ncx - g)
        self.jj = slice(g, self.ncy - g)
        self.bc = dict(imin="extrap", imax="extrap",
                       jmin="extrap", jmax="extrap")
        self.bc_state = {}

    # ------------------------------------------------------------------ grid
    def _extend_grid(self, Xn, Rn):
        X, R = Xn.copy(), Rn.copy()
        for _ in range(self.ng):
            X = np.vstack([2 * X[:1] - X[1:2], X, 2 * X[-1:] - X[-2:-1]])
            R = np.vstack([2 * R[:1] - R[1:2], R, 2 * R[-1:] - R[-2:-1]])
        for _ in range(self.ng):
            X = np.hstack([2 * X[:, :1] - X[:, 1:2], X,
                           2 * X[:, -1:] - X[:, -2:-1]])
            R = np.hstack([2 * R[:, :1] - R[:, 1:2], R,
                           2 * R[:, -1:] - R[:, -2:-1]])
        self.Xg, self.Rg = X, R

    def _metrics(self):
        X, R = self.Xg, self.Rg                          # nodes (ncx+1, ncy+1)
        ax, ay = X[:-1, :-1], R[:-1, :-1]
        bx, by = X[1:, :-1], R[1:, :-1]
        cx_, cy_ = X[1:, 1:], R[1:, 1:]
        dx, dy = X[:-1, 1:], R[:-1, 1:]
        area = 0.5 * ((ax * by - bx * ay) + (bx * cy_ - cx_ * by)
                      + (cx_ * dy - dx * cy_) + (dx * ay - ax * dy))
        # one global handedness sign (a uniform scalar keeps each cell's four
        # face vectors summing to zero -> the geometric conservation law and
        # exact free-stream preservation). Standard curvilinear metrics:
        #   S^xi = (y_eta, -x_eta)   at i-faces (points +i after the sign fix)
        #   S^eta = (-y_xi,  x_xi)   at j-faces (points +j)
        gsign = float(np.sign(np.median(area)))
        if gsign == 0.0:
            gsign = 1.0
        self.vol = area * gsign                          # (ncx, ncy) > 0
        self.cx = 0.25 * (ax + bx + cx_ + dx)
        self.cy = 0.25 * (ay + by + cy_ + dy)
        self.Sxi = (R[:, 1:] - R[:, :-1]) * gsign        # (ncx+1, ncy)
        self.Syi = -(X[:, 1:] - X[:, :-1]) * gsign
        self.Sxj = -(R[1:, :] - R[:-1, :]) * gsign       # (ncx, ncy+1)
        self.Syj = (X[1:, :] - X[:-1, :]) * gsign

    # -------------------------------------------------------------- state I/O
    def set_uniform(self, rho, u, v, p):
        g = self.gamma
        self.U[0] = rho
        self.U[1] = rho * u
        self.U[2] = rho * v
        self.U[3] = p / (g - 1.0) + 0.5 * rho * (u * u + v * v)

    def primitives(self, U=None):
        U = self.U if U is None else U
        rho = np.maximum(U[0], 1e-9)
        u, v = U[1] / rho, U[2] / rho
        p = np.maximum((self.gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v)),
                       1e-9)
        return rho, u, v, p

    def _cons_from_prim(self, rho, u, v, p):
        return np.array([rho, rho * u, rho * v,
                         p / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v)])

    # --------------------------------------------------------- boundary conds
    def set_bc(self, **kw):
        for k, val in kw.items():
            if k in self.bc:
                self.bc[k] = val
            if k.endswith("_state"):
                self.bc_state[k[:-6]] = val

    def _apply_bc(self):
        g, U = self.ng, self.U
        nx, ny = self.ncx, self.ncy
        pairs = (("imin", "imax", 1), ("jmin", "jmax", 2))
        for lo, hi, ax in pairs:
            if self.bc[lo] == "periodic":
                if ax == 1:
                    U[:, :g, :] = U[:, nx - 2 * g:nx - g, :]
                    U[:, nx - g:, :] = U[:, g:2 * g, :]
                else:
                    U[:, :, :g] = U[:, :, ny - 2 * g:ny - g]
                    U[:, :, ny - g:] = U[:, :, g:2 * g]
                continue
            self._fill(lo)
            self._fill(hi)

    def _fill(self, side):
        g, U = self.ng, self.U
        nx, ny = self.ncx, self.ncy
        kind = self.bc[side]
        if kind == "slipwall":
            self._slipwall(side)
            return
        fixed = kind == "fixed"
        val = (self._cons_from_prim(*self.bc_state[side])[:, None, None]
               if fixed else None)
        if side == "imin":
            U[:, :g, :] = val if fixed else U[:, g:g + 1, :]
        elif side == "imax":
            U[:, nx - g:, :] = val if fixed else U[:, nx - g - 1:nx - g, :]
        elif side == "jmin":
            U[:, :, :g] = val if fixed else U[:, :, g:g + 1]
        elif side == "jmax":
            U[:, :, ny - g:] = val if fixed else U[:, :, ny - g - 1:ny - g]

    def _slipwall(self, side):
        g, U = self.ng, self.U
        ny = self.ncy
        if side == "jmin":
            face, interior, dst = g, U[:, :, g:2 * g][:, :, ::-1], slice(0, g)
            Sx, Sy = self.Sxj[:, face], self.Syj[:, face]
        elif side == "jmax":
            face = ny - g
            interior, dst = U[:, :, ny - 2 * g:ny - g][:, :, ::-1], slice(ny - g, ny)
            Sx, Sy = self.Sxj[:, face], self.Syj[:, face]
        else:                                            # i-walls: extrap
            self.bc[side] = "extrap"; self._fill(side); return
        nl = np.hypot(Sx, Sy)
        nx = (Sx / np.maximum(nl, 1e-30))[:, None]
        ny_ = (Sy / np.maximum(nl, 1e-30))[:, None]
        rho = interior[0]
        u, v = interior[1] / rho, interior[2] / rho
        un = u * nx + v * ny_
        U[0][:, dst] = rho
        U[1][:, dst] = rho * (u - 2 * un * nx)
        U[2][:, dst] = rho * (v - 2 * un * ny_)
        U[3][:, dst] = interior[3]

    # ------------------------------------------------------------- HLLC flux
    def _hllc(self, WL, WR, nx, ny):
        g = self.gamma
        rL, uL, vL, pL = WL
        rR, uR, vR, pR = WR
        unL, utL = uL * nx + vL * ny, -uL * ny + vL * nx
        unR, utR = uR * nx + vR * ny, -uR * ny + vR * nx
        aL, aR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
        EL = pL / (g - 1.0) + 0.5 * rL * (uL * uL + vL * vL)
        ER = pR / (g - 1.0) + 0.5 * rR * (uR * uR + vR * vR)
        SL = np.minimum(unL - aL, unR - aR)
        SR = np.maximum(unL + aL, unR + aR)
        Ss = ((pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR))
              / (rL * (SL - unL) - rR * (SR - unR)))

        def cons(r, un, ut, E):
            return (r, r * un, r * ut, E)

        def flux(r, un, ut, p, E):
            return (r * un, r * un * un + p, r * un * ut, un * (E + p))

        def star(r, un, ut, p, E, S):
            fac = r * (S - un) / (S - Ss)
            return (fac, fac * Ss, fac * ut,
                    fac * (E / r + (Ss - un) * (Ss + p / (r * (S - un)))))

        UL, UR = cons(rL, unL, utL, EL), cons(rR, unR, utR, ER)
        FL, FR = flux(rL, unL, utL, pL, EL), flux(rR, unR, utR, pR, ER)
        ULs, URs = star(rL, unL, utL, pL, EL, SL), star(rR, unR, utR, pR, ER, SR)
        out = []
        for m in range(4):
            fSL = FL[m] + SL * (ULs[m] - UL[m])
            fSR = FR[m] + SR * (URs[m] - UR[m])
            out.append(np.where(SL >= 0.0, FL[m],
                       np.where(Ss >= 0.0, fSL,
                       np.where(SR > 0.0, fSR, FR[m]))))
        Frn, Frun, Frut, FE = out
        return np.stack([Frn, Frun * nx - Frut * ny, Frun * ny + Frut * nx, FE])

    # --------------------------------------------------------------- stepping
    def _face_states(self, W, axis):
        """MUSCL minmod L/R states at the interfaces between cells along axis
        (0=i,1=j). Returns (WL, WR) of length (N_axis - 1)."""
        ax = axis + 1
        dm = np.diff(W, axis=ax)                          # (…, N-1, …)
        s = np.zeros_like(W)
        sl = [slice(None)] * 3
        a = list(sl); a[ax] = slice(0, -1)
        b = list(sl); b[ax] = slice(1, None)
        inner = list(sl); inner[ax] = slice(1, -1)
        s[tuple(inner)] = _minmod(dm[tuple(a)], dm[tuple(b)])
        WL = W[tuple(a)] + 0.5 * s[tuple(a)]
        WR = W[tuple(b)] - 0.5 * s[tuple(b)]
        return WL, WR

    def _rhs(self, U):
        rho, u, v, p = self.primitives(U)
        W = np.stack([rho, u, v, p])
        # i-interfaces (between cells): metric = interior node-lines Sxi[1:-1]
        WLi, WRi = self._face_states(W, 0)
        Si = self.Sxi[1:-1], self.Syi[1:-1]
        nli = np.hypot(*Si)
        Fi = self._hllc(WLi, WRi, Si[0] / nli, Si[1] / nli) * nli
        # j-interfaces
        WLj, WRj = self._face_states(W, 1)
        Sj = self.Sxj[:, 1:-1], self.Syj[:, 1:-1]
        nlj = np.hypot(*Sj)
        Fj = self._hllc(WLj, WRj, Sj[0] / nlj, Sj[1] / nlj) * nlj
        div = np.zeros_like(U)
        div[:, 1:-1, :] += Fi[:, 1:, :] - Fi[:, :-1, :]
        div[:, :, 1:-1] += Fj[:, :, 1:] - Fj[:, :, :-1]
        return -div / np.maximum(self.vol, 1e-30)

    def max_wave_dt(self, cfl):
        rho, u, v, p = self.primitives()
        a = np.sqrt(self.gamma * p / rho)
        si = 0.5 * (np.hypot(self.Sxi[1:], self.Syi[1:])
                    + np.hypot(self.Sxi[:-1], self.Syi[:-1]))
        sj = 0.5 * (np.hypot(self.Sxj[:, 1:], self.Syj[:, 1:])
                    + np.hypot(self.Sxj[:, :-1], self.Syj[:, :-1]))
        rad = (np.hypot(u, v) + a) * (si + sj)
        return cfl * float(np.min(self.vol / np.maximum(rad, 1e-30)))

    def step(self, dt):
        U0 = self.U.copy()
        self._apply_bc()
        k1 = self._rhs(self.U)
        self.U[:, self.ii, self.jj] = U0[:, self.ii, self.jj] + dt * k1[:, self.ii, self.jj]
        self._apply_bc()
        k2 = self._rhs(self.U)
        self.U[:, self.ii, self.jj] = (0.5 * U0[:, self.ii, self.jj]
                                       + 0.5 * (self.U[:, self.ii, self.jj]
                                                + dt * k2[:, self.ii, self.jj]))

    def run(self, t_end, cfl=0.4, max_steps=200000):
        t = 0.0
        for _ in range(max_steps):
            dt = min(self.max_wave_dt(cfl), t_end - t)
            self.step(dt)
            t += dt
            if t >= t_end - 1e-14:
                break
        return t
