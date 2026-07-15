"""Overset Phase 4: two-grid time-loop coupling.

Drives a near-body curvilinear solver and a Cartesian background solver in
lockstep, exchanging donor/receptor states through the Phase-3 connectivity each
step (loose explicit coupling): before every step the receptor cells are
overwritten by interpolation from the other grid, so the interior of each grid
sees a consistent exterior. A uniform Cartesian grid is just a trivial
curvilinear grid, so both sides use the same CurvilinearEuler solver / numerics.
"""
from __future__ import annotations

import numpy as np

from .curvilinear import CurvilinearEuler
from .overset_connect import OversetConnectivity


class OversetCoupler:
    def __init__(self, body_Xn, body_Rn, x0, y0, dx, ncx, ncy, *,
                 gamma: float = 1.4, blank_frac: float = 0.55,
                 fringe_layers: int = 2):
        self.body = CurvilinearEuler(body_Xn, body_Rn, gamma=gamma)
        I, J = np.meshgrid(np.arange(ncx + 1), np.arange(ncy + 1), indexing="ij")
        self.back = CurvilinearEuler(x0 + I * dx, y0 + J * dx, gamma=gamma)
        self.conn = OversetConnectivity(body_Xn, body_Rn, x0, y0, dx, ncx, ncy,
                                        blank_frac=blank_frac,
                                        fringe_layers=fringe_layers)
        self.ng = self.body.ng

    def _body_real(self):
        ng, c = self.ng, self.conn
        return self.body.U[:, ng:ng + c.Nbi, ng:ng + c.Nbj]

    def _back_real(self):
        ng, c = self.ng, self.conn
        return self.back.U[:, ng:ng + c.ncx, ng:ng + c.ncy]

    def exchange(self):
        """Impose receptor states by interpolation (conserved variables)."""
        c = self.conn
        bU, kU = self._body_real(), self._back_real()
        if len(c.recv_idx):
            ri, rj = c.recv_idx[:, 0], c.recv_idx[:, 1]
            for m in range(4):
                kU[m][ri, rj] = c.interp_to_cart_recv(bU[m])
        if len(c.br_idx):
            bi = np.asarray(c.br_idx)
            for m in range(4):
                bU[m][bi[:, 0], bi[:, 1]] = c.interp_to_body_outer(kU[m])

    def step(self, dt):
        self.exchange()
        self.body.step(dt)
        self.back.step(dt)

    def run(self, t_end, cfl=0.4, max_steps=200000):
        t = 0.0
        for _ in range(max_steps):
            dt = min(self.body.max_wave_dt(cfl), self.back.max_wave_dt(cfl),
                     t_end - t)
            self.step(dt)
            t += dt
            if t >= t_end - 1e-14:
                break
        return t
