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


# --------------------------------------------------------------------------- #
#  Menter k-omega SST constants (same set as the GPU Cartesian solver)
# --------------------------------------------------------------------------- #
_BSTAR = 0.09
_SIGK1, _SIGK2 = 0.85, 1.0
_SIGW1, _SIGW2 = 0.5, 0.856
_BET1, _BET2 = 0.075, 0.0828
_GAM1, _GAM2 = 0.5532, 0.4404
_A1 = 0.31


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
        # viscous / turbulent (laminar Navier-Stokes; mu_t is an externally
        # supplied eddy viscosity so a turbulence model plugs in as mu+mu_t)
        self.viscous = False
        self.mu = 0.0
        self.Pr = 0.72
        self.Rgas = 287.0
        self.mu_t = None
        self.wall_vel = {}
        # k-omega SST
        self.turb = False
        self.Ut = None                                   # (2,ncx,ncy) rho*k,rho*w
        self.wd = None                                   # wall distance field
        self.Prt = 0.9
        self.body_force = None                           # (fx, fy) per volume

    def set_body_force(self, fx, fy=0.0):
        """Constant volumetric momentum source (e.g. the mean pressure gradient
        that drives a periodic channel: fx = rho u_tau^2 / h)."""
        self.body_force = (float(fx), float(fy))
        return self

    def set_viscous(self, mu, Pr=0.72, Rgas=287.0, mu_t=None):
        self.viscous = True
        self.mu = float(mu)
        self.Pr = float(Pr)
        self.Rgas = float(Rgas)
        self.mu_t = mu_t                                 # (ncx,ncy) cell field
        return self

    # ------------------------------------------------------------ k-omega SST
    def set_turbulence(self, wall_dist, k0=1.0e-6, w0=10.0):
        """Enable Menter k-omega SST. ``wall_dist`` is the cell-centred distance
        to the nearest wall (on the near-body grid this is just the eta
        distance); k0/w0 seed the field."""
        if not self.viscous:
            raise ValueError("enable set_viscous() before set_turbulence()")
        self.turb = True
        self.wd = np.maximum(np.asarray(wall_dist, float), 1e-12)
        rho = np.maximum(self.U[0], 1e-9)
        self.Ut = np.stack([rho * k0, rho * w0])
        self._update_mut()
        return self

    def _turb_prim(self, Ut=None):
        Ut = self.Ut if Ut is None else Ut
        rho = np.maximum(self.U[0], 1e-9)
        k = np.maximum(Ut[0] / rho, 1e-14)
        w = np.maximum(Ut[1] / rho, 1e-3)
        return rho, k, w

    def _strain(self, u, v):
        gux, guy = self._grad(u)
        gvx, gvy = self._grad(v)
        S2 = 2.0 * (gux * gux + gvy * gvy) + (guy + gvx) ** 2
        return S2

    def _update_mut(self, Ut=None):
        """Eddy viscosity mu_t = rho a1 k / max(a1 w, S F2) and the F2 limiter."""
        rho, k, w = self._turb_prim(Ut)
        _, u, v, _ = self.primitives()
        S2 = self._strain(u, v)
        d = self.wd
        nu = self.mu / rho
        t1 = np.sqrt(k) / (_BSTAR * w * d)
        t2 = 500.0 * nu / (d * d * w)
        arg2 = np.minimum(np.maximum(2.0 * t1, t2), 1.0e3)
        F2 = np.tanh(arg2 * arg2)
        self.mu_t = rho * _A1 * k / np.maximum(_A1 * w, np.sqrt(S2) * F2)
        self._S2 = S2
        return self.mu_t

    def _blend_F1(self, rho, k, w):
        gkx, gky = self._grad(k)
        gwx, gwy = self._grad(w)
        d = self.wd
        nu = self.mu / rho
        t1 = np.sqrt(k) / (_BSTAR * w * d)
        t2 = 500.0 * nu / (d * d * w)
        dkdw = gkx * gwx + gky * gwy
        CDkw = np.maximum(2.0 * rho * _SIGW2 / w * dkdw, 1.0e-20)
        arg1 = np.minimum(np.minimum(np.maximum(t1, t2),
                                     4.0 * rho * _SIGW2 * k / (CDkw * d * d)),
                          1.0e3)
        F1 = np.tanh(arg1 ** 4)
        return F1, (gkx, gky), (gwx, gwy), dkdw

    def _turb_rhs(self, Ut, Fim, Fjm):
        """d(rho k, rho w)/dt: convection on the HLLC mass flux, SST diffusion,
        production and cross-diffusion. Destruction is point-implicit in _step."""
        rho, k, w = self._turb_prim(Ut)
        F1, gk, gw, dkdw = self._blend_F1(rho, k, w)
        mut = self.mu_t
        sigk = F1 * _SIGK1 + (1.0 - F1) * _SIGK2
        sigw = F1 * _SIGW1 + (1.0 - F1) * _SIGW2
        gamc = F1 * _GAM1 + (1.0 - F1) * _GAM2

        div = np.zeros_like(Ut)
        for axis, (Sx, Sy), Fm in ((0, (self.Sxi[1:-1], self.Syi[1:-1]), Fim),
                                   (1, (self.Sxj[:, 1:-1], self.Syj[:, 1:-1]), Fjm)):
            fa = (lambda q: 0.5 * (q[:-1, :] + q[1:, :])) if axis == 0 else \
                 (lambda q: 0.5 * (q[:, :-1] + q[:, 1:]))
            up = (lambda q: np.where(Fm > 0.0, q[:-1, :], q[1:, :])) if axis == 0 \
                else (lambda q: np.where(Fm > 0.0, q[:, :-1], q[:, 1:]))
            # convection (mass-flux upwind) minus SST diffusion
            kfx, kfy = self._face_grad(k, gk[0], gk[1], axis)
            wfx, wfy = self._face_grad(w, gw[0], gw[1], axis)
            dk = (self.mu + fa(sigk) * fa(mut)) * (kfx * Sx + kfy * Sy)
            dw = (self.mu + fa(sigw) * fa(mut)) * (wfx * Sx + wfy * Sy)
            Fk = Fm * up(k) - dk
            Fw = Fm * up(w) - dw
            if axis == 0:
                div[0, 1:-1, :] += Fk[1:, :] - Fk[:-1, :]
                div[1, 1:-1, :] += Fw[1:, :] - Fw[:-1, :]
            else:
                div[0, :, 1:-1] += Fk[:, 1:] - Fk[:, :-1]
                div[1, :, 1:-1] += Fw[:, 1:] - Fw[:, :-1]
        out = -div / np.maximum(self.vol, 1e-30)
        # production (limited as in the GPU solver) + cross-diffusion
        S2 = np.minimum(self._S2, 10.0 * _BSTAR * rho * k * w
                        / np.maximum(mut, 1e-12))
        CD = 2.0 * (1.0 - F1) * rho * _SIGW2 / w * dkdw
        out[0] += mut * S2
        out[1] += gamc * rho * S2 + CD
        self._F1 = F1
        return out

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

    def set_wall_velocity(self, side, uw, vw=0.0):
        self.wall_vel[side] = (float(uw), float(vw))
        return self

    def _fill(self, side):
        g, U = self.ng, self.U
        nx, ny = self.ncx, self.ncy
        kind = self.bc[side]
        if kind in ("slipwall", "noslip"):
            self._wall(side, slip=(kind == "slipwall"))
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

    def _wall(self, side, slip):
        """Slip (reflect normal velocity) or no-slip (ghost velocity so the wall
        face carries the wall velocity) j-wall; internal energy is mirrored
        (adiabatic), only the kinetic part is rebuilt. i-walls fall back to
        extrapolation."""
        g, U = self.ng, self.U
        ny = self.ncy
        if side == "jmin":
            face, interior, dst = g, U[:, :, g:2 * g][:, :, ::-1], slice(0, g)
        elif side == "jmax":
            face = ny - g
            interior, dst = U[:, :, ny - 2 * g:ny - g][:, :, ::-1], slice(ny - g, ny)
        else:
            self.bc[side] = "extrap"; self._fill(side); return
        Sx, Sy = self.Sxj[:, face], self.Syj[:, face]
        nl = np.hypot(Sx, Sy)
        nx = (Sx / np.maximum(nl, 1e-30))[:, None]
        ny_ = (Sy / np.maximum(nl, 1e-30))[:, None]
        rho = interior[0]
        u, v = interior[1] / rho, interior[2] / rho
        if slip:
            un = u * nx + v * ny_
            u2, v2 = u - 2 * un * nx, v - 2 * un * ny_
        else:
            uw, vw = self.wall_vel.get(side, (0.0, 0.0))
            u2, v2 = 2 * uw - u, 2 * vw - v       # face velocity -> wall velocity
        ei = interior[3] - 0.5 * rho * (u * u + v * v)
        U[0][:, dst] = rho
        U[1][:, dst] = rho * u2
        U[2][:, dst] = rho * v2
        U[3][:, dst] = ei + 0.5 * rho * (u2 * u2 + v2 * v2)

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

    # ------------------------------------------------------ viscous (laminar)
    def _grad(self, phi):
        """Green-Gauss cell-centred gradient of a cell field (ghosts filled)."""
        gx = np.zeros_like(phi)
        gy = np.zeros_like(phi)
        pfi = 0.5 * (phi[:-1, :] + phi[1:, :])           # i-interfaces
        gxi, gyi = pfi * self.Sxi[1:-1], pfi * self.Syi[1:-1]
        gx[1:-1, :] += gxi[1:, :] - gxi[:-1, :]
        gy[1:-1, :] += gyi[1:, :] - gyi[:-1, :]
        pfj = 0.5 * (phi[:, :-1] + phi[:, 1:])           # j-interfaces
        gxj, gyj = pfj * self.Sxj[:, 1:-1], pfj * self.Syj[:, 1:-1]
        gx[:, 1:-1] += gxj[:, 1:] - gxj[:, :-1]
        gy[:, 1:-1] += gyj[:, 1:] - gyj[:, :-1]
        gx /= self.vol
        gy /= self.vol
        gx[0], gx[-1] = gx[1], gx[-2]
        gy[0], gy[-1] = gy[1], gy[-2]
        gx[:, 0], gx[:, -1] = gx[:, 1], gx[:, -2]
        gy[:, 0], gy[:, -1] = gy[:, 1], gy[:, -2]
        return gx, gy

    def _face_grad(self, phi, gx, gy, axis):
        """Corrected face gradient (averaged cell gradient with its face-normal
        component replaced by the direct difference -> no odd/even decoupling)."""
        if axis == 0:
            pL, pR = phi[:-1, :], phi[1:, :]
            gaX, gaY = 0.5 * (gx[:-1, :] + gx[1:, :]), 0.5 * (gy[:-1, :] + gy[1:, :])
            eX = self.cx[1:, :] - self.cx[:-1, :]
            eY = self.cy[1:, :] - self.cy[:-1, :]
        else:
            pL, pR = phi[:, :-1], phi[:, 1:]
            gaX, gaY = 0.5 * (gx[:, :-1] + gx[:, 1:]), 0.5 * (gy[:, :-1] + gy[:, 1:])
            eX = self.cx[:, 1:] - self.cx[:, :-1]
            eY = self.cy[:, 1:] - self.cy[:, :-1]
        corr = ((pR - pL) - (gaX * eX + gaY * eY)) / np.maximum(eX * eX + eY * eY, 1e-30)
        return gaX + corr * eX, gaY + corr * eY

    def _mu_eff(self, rho):
        m = np.full(rho.shape, self.mu, float)
        if self.mu_t is not None:
            m = m + self.mu_t
        return m

    def _viscous_fluxes(self, rho, u, v, p):
        T = p / (np.maximum(rho, 1e-9) * self.Rgas)
        gux, guy = self._grad(u)
        gvx, gvy = self._grad(v)
        gTx, gTy = self._grad(T)
        mueff = self._mu_eff(rho)
        cp = self.gamma * self.Rgas / (self.gamma - 1.0)
        res = []
        faces = ((0, (self.Sxi[1:-1], self.Syi[1:-1])),
                 (1, (self.Sxj[:, 1:-1], self.Syj[:, 1:-1])))
        for axis, (Sx, Sy) in faces:
            ux, uy = self._face_grad(u, gux, guy, axis)
            vx, vy = self._face_grad(v, gvx, gvy, axis)
            Tx, Ty = self._face_grad(T, gTx, gTy, axis)
            fa = (lambda q: 0.5 * (q[:-1, :] + q[1:, :])) if axis == 0 else \
                 (lambda q: 0.5 * (q[:, :-1] + q[:, 1:]))
            mf, uf, vf = fa(mueff), fa(u), fa(v)
            dvg = ux + vy
            txx = mf * (2.0 * ux - 2.0 / 3.0 * dvg)
            tyy = mf * (2.0 * vy - 2.0 / 3.0 * dvg)
            txy = mf * (uy + vx)
            kap = cp * mf / self.Pr
            Fmx = txx * Sx + txy * Sy
            Fmy = txy * Sx + tyy * Sy
            Fe = ((uf * txx + vf * txy) * Sx + (uf * txy + vf * tyy) * Sy
                  + kap * (Tx * Sx + Ty * Sy))
            res.append(np.stack([np.zeros_like(Fmx), Fmx, Fmy, Fe]))
        return res[0], res[1]

    # --------------------------------------------------------------- residual
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
        if self.viscous:                                 # subtract diffusive flux
            Fvi, Fvj = self._viscous_fluxes(rho, u, v, p)
            Fi = Fi - Fvi
            Fj = Fj - Fvj
        self._Fim, self._Fjm = Fi[0], Fj[0]              # mass flux (for k/omega)
        div = np.zeros_like(U)
        div[:, 1:-1, :] += Fi[:, 1:, :] - Fi[:, :-1, :]
        div[:, :, 1:-1] += Fj[:, :, 1:] - Fj[:, :, :-1]
        out = -div / np.maximum(self.vol, 1e-30)
        if self.body_force is not None:                  # volumetric momentum src
            fx, fy = self.body_force
            out[1] += fx
            out[2] += fy
            out[3] += fx * u + fy * v
        return out

    def dt_field(self, cfl):
        """Per-cell stable dt over the whole array (local time stepping: not
        time accurate, but converges to the same steady state far faster since
        tiny near-wall cells no longer throttle the entire domain)."""
        rho, u, v, p = self.primitives()
        rho = np.maximum(rho, 1e-9)
        a = np.sqrt(self.gamma * np.maximum(p, 1e-9) / rho)
        si = 0.5 * (np.hypot(self.Sxi[1:], self.Syi[1:])
                    + np.hypot(self.Sxi[:-1], self.Syi[:-1]))
        sj = 0.5 * (np.hypot(self.Sxj[:, 1:], self.Syj[:, 1:])
                    + np.hypot(self.Sxj[:, :-1], self.Syj[:, :-1]))
        vol = np.maximum(self.vol, 1e-30)
        rad = (np.hypot(u, v) + a) * (si + sj)
        dt = cfl * vol / np.maximum(rad, 1e-30)
        if self.viscous:
            mut = self.mu_t if self.mu_t is not None else 0.0
            nu = (self.mu + mut) / rho
            dtv = 0.25 * vol ** 2 / np.maximum(nu * (si * si + sj * sj), 1e-30)
            dt = np.minimum(dt, dtv)
        return dt

    def max_wave_dt(self, cfl):
        # only real cells constrain the step; ghost cells may be stale (e.g.
        # uninitialised before the first BC fill), which would poison a global min
        ii, jj = self.ii, self.jj
        rho, u, v, p = self.primitives()
        rho, u, v, p = rho[ii, jj], u[ii, jj], v[ii, jj], p[ii, jj]
        a = np.sqrt(self.gamma * p / rho)
        si = 0.5 * (np.hypot(self.Sxi[1:], self.Syi[1:])
                    + np.hypot(self.Sxi[:-1], self.Syi[:-1]))[ii, jj]
        sj = 0.5 * (np.hypot(self.Sxj[:, 1:], self.Syj[:, 1:])
                    + np.hypot(self.Sxj[:, :-1], self.Syj[:, :-1]))[ii, jj]
        vol = self.vol[ii, jj]
        rad = (np.hypot(u, v) + a) * (si + sj)
        dt = cfl * float(np.min(vol / np.maximum(rad, 1e-30)))
        if self.viscous:                                 # viscous stability
            mut = self.mu_t[ii, jj] if self.mu_t is not None else 0.0
            nu = (self.mu + mut) / rho
            dtv = 0.25 * float(np.min(vol ** 2 / np.maximum(
                nu * (si * si + sj * sj), 1e-30)))
            dt = min(dt, dtv)
        return dt

    # ------------------------------------------------- turbulence BC / update
    def _turb_wall(self, side):
        """Smooth-wall SST: k -> 0, omega -> Menter's 60 nu / (beta1 y1^2)."""
        g, T, ny = self.ng, self.Ut, self.ncy
        rho = np.maximum(self.U[0], 1e-9)
        if side == "jmin":
            dst, first = slice(0, g), g
        elif side == "jmax":
            dst, first = slice(ny - g, ny), ny - g - 1
        else:
            return
        nu = self.mu / rho[:, first]
        d1 = self.wd[:, first]
        wwall = np.minimum(60.0 * nu / (_BET1 * d1 * d1), 1.0e12)
        T[0][:, dst] = (rho[:, first] * 1.0e-14)[:, None]
        T[1][:, dst] = (rho[:, first] * wwall)[:, None]

    def _turb_bc(self):
        g, T = self.ng, self.Ut
        nx, ny = self.ncx, self.ncy
        for lo, hi, ax in (("imin", "imax", 1), ("jmin", "jmax", 2)):
            if self.bc[lo] == "periodic":
                if ax == 1:
                    T[:, :g, :] = T[:, nx - 2 * g:nx - g, :]
                    T[:, nx - g:, :] = T[:, g:2 * g, :]
                else:
                    T[:, :, :g] = T[:, :, ny - 2 * g:ny - g]
                    T[:, :, ny - g:] = T[:, :, g:2 * g]
                continue
            for side in (lo, hi):
                if self.bc[side] in ("noslip", "slipwall"):
                    self._turb_wall(side)
                elif side == "imin":
                    T[:, :g, :] = T[:, g:g + 1, :]
                elif side == "imax":
                    T[:, nx - g:, :] = T[:, nx - g - 1:nx - g, :]
                elif side == "jmin":
                    T[:, :, :g] = T[:, :, g:g + 1]
                elif side == "jmax":
                    T[:, :, ny - g:] = T[:, :, ny - g - 1:ny - g]

    def _turb_apply(self, Ut0, rhs, dt, ca, cb, cc):
        """SSP-RK combine with point-implicit destruction (as the GPU solver)."""
        ii, jj = self.ii, self.jj
        rho, k, w = self._turb_prim()
        bet = self._F1 * _BET1 + (1.0 - self._F1) * _BET2
        num = (ca * Ut0[0][ii, jj] + cb * self.Ut[0][ii, jj]
               + cc * dt * rhs[0][ii, jj])
        self.Ut[0][ii, jj] = num / (1.0 + cc * dt * _BSTAR * w[ii, jj])
        # omega destruction -beta*w^2 point-implicit: w/(1+dt*beta*w) inverts to
        # 1/w_{n+1} = 1/w_n + dt*beta, i.e. exactly the analytic w0/(1+beta w0 t)
        num = (ca * Ut0[1][ii, jj] + cb * self.Ut[1][ii, jj]
               + cc * dt * rhs[1][ii, jj])
        self.Ut[1][ii, jj] = num / (1.0 + cc * dt * bet[ii, jj] * w[ii, jj])
        r = np.maximum(self.U[0], 1e-9)
        self.Ut[0] = np.clip(self.Ut[0], r * 1e-14, r * 1e8)
        self.Ut[1] = np.clip(self.Ut[1], r * 1e-3, r * 1e12)
        self._turb_bc()

    def step(self, dt):
        """dt may be a scalar (time accurate) or a per-cell field from
        dt_field() (local time stepping, steady-state acceleration)."""
        ii, jj = self.ii, self.jj
        dtl = dt[ii, jj] if np.ndim(dt) > 0 else dt
        U0 = self.U.copy()
        Ut0 = self.Ut.copy() if self.turb else None
        self._apply_bc()
        if self.turb:
            self._update_mut()
        k1 = self._rhs(self.U)
        kt1 = self._turb_rhs(self.Ut, self._Fim, self._Fjm) if self.turb else None
        self.U[:, ii, jj] = U0[:, ii, jj] + dtl * k1[:, ii, jj]
        if self.turb:
            self._turb_apply(Ut0, kt1, dtl, 1.0, 0.0, 1.0)
        self._apply_bc()
        if self.turb:
            self._update_mut()
        k2 = self._rhs(self.U)
        kt2 = self._turb_rhs(self.Ut, self._Fim, self._Fjm) if self.turb else None
        self.U[:, ii, jj] = (0.5 * U0[:, ii, jj]
                             + 0.5 * (self.U[:, ii, jj] + dtl * k2[:, ii, jj]))
        if self.turb:
            self._turb_apply(Ut0, kt2, dtl, 0.5, 0.5, 0.5)

    def run(self, t_end, cfl=0.4, max_steps=200000):
        t = 0.0
        for _ in range(max_steps):
            dt = min(self.max_wave_dt(cfl), t_end - t)
            self.step(dt)
            t += dt
            if t >= t_end - 1e-14:
                break
        return t
