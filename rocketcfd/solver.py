"""GPU solver: owns device arrays, runs SSP-RK2 steps, produces snapshots."""
from __future__ import annotations

import time

import numpy as np

from .config import SimConfig
from .cuda_kernels import KernelSet, axis_j
from .mask import DomainMask, FLUID
from .postproc import performance


class GPUSolver:
    def __init__(self, mask: DomainMask, cfg: SimConfig):
        import cupy as cp
        self.cp = cp
        self.cfg = cfg
        self.mask = mask
        self.nx, self.ny = mask.nx, mask.ny
        self.sx, self.sy = self.nx + 4, self.ny + 4
        nc = self.sx * self.sy

        self.kern = KernelSet(cfg, self.nx, self.ny)

        f4 = cp.float32
        self.U = cp.zeros((6, self.sy, self.sx), dtype=f4)
        self.U0 = cp.zeros_like(self.U)
        self.P = cp.zeros((10, self.sy, self.sx), dtype=f4)
        self.G = cp.zeros((10, self.sy, self.sx), dtype=f4)
        self.FX = cp.zeros((6, self.sy, self.sx), dtype=f4)
        self.FY = cp.zeros((6, self.sy, self.sx), dtype=f4)
        self.s2 = cp.zeros((self.sy, self.sx), dtype=f4)
        self.sk = cp.zeros((self.sy, self.sx), dtype=f4)
        self.sw = cp.zeros((self.sy, self.sx), dtype=f4)
        self.dtl = cp.full((self.sy, self.sx), 1e30, dtype=f4)
        self.res = cp.zeros((self.sy, self.sx), dtype=f4)
        self.qw = cp.zeros((self.sy, self.sx), dtype=f4)   # wall heat flux

        self.ct = cp.asarray(mask.cell_type)
        self.wd = cp.asarray(mask.wall_dist)
        self.axf = cp.asarray(mask.ax)        # cut-cell face apertures
        self.ayf = cp.asarray(mask.ay)
        self.lam = cp.asarray(mask.lam)       # cut-cell volume fractions
        self.fluid_np = mask.cell_type == FLUID
        self.n_fluid = int(self.fluid_np.sum())

        self.step_count = 0
        self.sim_time = 0.0
        self.residual = 1.0
        self.res0 = None
        self.res_history: list[tuple[int, float]] = []
        self.thrust_history: list[tuple[int, float]] = []
        self._steps_per_sec = 0.0

        self._set_initial_state()

    # ------------------------------------------------------------------
    def _thermo_coeffs(self):
        """cp/R cubic for the thermally perfect gas model (None if CP)."""
        if getattr(self.cfg, "gas_model",
                   "calorically perfect").lower().startswith("thermally"):
            from . import thermo
            return thermo.cpr_coeffs(self.cfg.propellant,
                                     fallback_gamma=self.cfg.gamma)
        return None

    def _eq_tables(self):
        """Equilibrium property tables (None unless gas_model=equilibrium)."""
        from .cuda_kernels import gas_mode
        if gas_mode(self.cfg) == 2:
            from . import equilibrium as eqm
            return eqm.build_tables(self.cfg.propellant)
        return None

    def _set_initial_state(self):
        cp, cfg = self.cp, self.cfg
        ke_far = 0.5 * (cfg.farfield_u ** 2 + cfg.farfield_v ** 2)
        tab = self._eq_tables()
        tc = self._thermo_coeffs()
        if tab is not None:
            from . import equilibrium as eqm
            rho, e, _ = eqm.ambient_state(tab, cfg.farfield_p, cfg.farfield_T)
            E = rho * e + rho * ke_far
        elif tc is not None:
            from . import thermo
            rho = cfg.farfield_p / (cfg.R_gas * cfg.farfield_T)
            e = cfg.R_gas * float(thermo.er(tc, cfg.farfield_T))
            E = rho * e + rho * ke_far
        else:
            rho = cfg.farfield_p / (cfg.R_gas * cfg.farfield_T)
            E = cfg.farfield_p / (cfg.gamma - 1.0) + rho * ke_far
        self.U[0].fill(rho)
        self.U[1].fill(rho * cfg.farfield_u)
        self.U[2].fill(rho * cfg.farfield_v)
        self.U[3].fill(E)
        self.U[4].fill(rho * 1.0e-6)
        self.U[5].fill(rho * 10.0)

    # ------------------------------------------------------------------
    def _p0_effective(self) -> np.float32:
        """Soft-start: ramp inlet total pressure over the first N steps."""
        n = max(int(self.cfg.inlet_ramp_steps), 0)
        s = min(self.step_count / n, 1.0) if n > 0 else 1.0
        p0 = self.cfg.farfield_p + (self.cfg.inlet_p0 - self.cfg.farfield_p) * s
        return np.float32(max(p0, 1.01 * self.cfg.farfield_p))

    def _rhs(self, compute_dt: bool):
        k = self.kern
        p0eff = self._p0_effective()
        k.launch(k.cons2prim, self.U, self.P, self.ct, self.dtl, self.lam,
                 np.int32(1 if compute_dt else 0))
        if compute_dt and not self.cfg.local_dt:
            dt_min = float(self.cp.min(self.dtl))
            self.dtl.fill(dt_min)
            self._dt_global = dt_min
        k.launch(k.halo_fill, self.P, self.ct)
        k.launch(k.gradients, self.P, self.G, self.ct, self.wd, p0eff)
        k.launch(k.turb_visc, self.P, self.G, self.wd, self.ct, self.s2)
        k.launch(k.fluxes, self.P, self.G, self.ct, self.wd, self.FX, np.int32(0), p0eff)
        k.launch(k.fluxes, self.P, self.G, self.ct, self.wd, self.FY, np.int32(1), p0eff)
        k.launch(k.sst_source, self.P, self.G, self.s2, self.ct, self.sk, self.sw)

    def _combine(self, ca: float, cb: float, cc: float):
        k = self.kern
        k.launch(k.rk_combine, self.U0, self.U, self.P, self.G, self.FX, self.FY,
                 self.sk, self.sw, self.dtl, self.ct,
                 self.axf, self.ayf, self.lam, self.wd, self.res, self.qw,
                 np.float32(ca), np.float32(cb), np.float32(cc))

    def reconfigure(self, cfg: SimConfig):
        """Rebuild the kernels for a new config while keeping the current flow
        state. Used for warm-started parameter sweeps (e.g. an altitude sweep
        that only changes the ambient back-pressure): the converged solution at
        one back-pressure is a good initial guess for the next, so each point
        re-equilibrates in far fewer steps than starting from quiescent gas.
        Compile-time constants such as ``farfield_p`` live in the kernel source,
        so the kernels must be recompiled, but ``U`` is preserved."""
        self.cfg = cfg
        self.kern = KernelSet(cfg, self.nx, self.ny)
        self.res0 = None
        self.residual = 1.0

    def step(self, n: int = 1):
        """Advance n SSP-RK2 steps."""
        cp = self.cp
        t_start = time.perf_counter()
        self._dt_global = 0.0
        for _ in range(n):
            cp.copyto(self.U0, self.U)
            self._rhs(compute_dt=True)
            self._combine(1.0, 0.0, 1.0)
            self._rhs(compute_dt=False)
            self._combine(0.5, 0.5, 0.5)
            self.step_count += 1
            if not self.cfg.local_dt:
                self.sim_time += self._dt_global
        # density residual (L2 of d(rho)/dt over fluid cells), normalized by
        # the largest residual seen so far (robust against soft-start ramps)
        r = float(cp.sqrt(cp.sum(self.res ** 2))) / max(self.n_fluid, 1)
        cp.cuda.runtime.deviceSynchronize()
        if r > 0.0 and (self.res0 is None or r > self.res0):
            self.res0 = r
        self.residual = r / self.res0 if self.res0 else 1.0
        self.res_history.append((self.step_count, self.residual))
        dt_wall = time.perf_counter() - t_start
        self._steps_per_sec = n / dt_wall if dt_wall > 0 else 0.0

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Copy interior primitive fields to CPU. Walls are NaN-masked."""
        cp, cfg = self.cp, self.cfg
        P = cp.asnumpy(self.P[:, 2:-2, 2:-2])
        rho, u, v, p, T, k, w, mul, mut = (P[i] for i in range(9))
        tab = self._eq_tables()
        tc = self._thermo_coeffs()
        if tab is not None:
            from . import equilibrium as eqm
            a = eqm.table_interp(tab, "A", np.maximum(rho, 1e-6),
                                 np.maximum(T, 246.0))
        else:
            if tc is not None:
                from . import thermo
                gam = thermo.gamma_of_T(tc, np.maximum(T, 1.0))
            else:
                gam = cfg.gamma
            a = np.sqrt(gam * cfg.R_gas * np.maximum(T, 1.0))
        vel = np.sqrt(u * u + v * v)
        mach = vel / a
        gy, gx = np.gradient(rho, self.mask.dx)
        schlieren = np.sqrt(gx * gx + gy * gy)

        fluid = self.fluid_np[2:-2, 2:-2]
        fields = {
            "Mach": mach, "Pressure [Pa]": p, "Temperature [K]": T,
            "Density [kg/m^3]": rho, "Velocity |V| [m/s]": vel,
            "Velocity u [m/s]": u, "Velocity v [m/s]": v,
            "Turb. kinetic energy k [m^2/s^2]": k, "Specific dissipation omega [1/s]": w,
            "Eddy viscosity ratio mu_t/mu [-]": mut / np.maximum(mul, 1e-12),
            "Schlieren |grad rho|": schlieren,
        }
        if getattr(cfg, "wall_T", 0.0) > 0.0 and cfg.wall_type == "noslip":
            fields["Wall heat flux [W/m^2]"] = cp.asnumpy(self.qw[2:-2, 2:-2])
        for key in fields:
            arr = fields[key].astype(np.float32).copy()
            arr[~fluid] = np.nan
            fields[key] = arr

        ax_i = ay_i = None
        if self.mask.smooth:
            ax_i = self.mask.ax[2:-2, 2:self.nx + 3]
            ay_i = self.mask.ay[2:self.ny + 3, 2:-2]
        perf = performance(
            p, rho, u, v, self.mask.cell_type[2:-2, 2:-2], self.mask.dx,
            self.cfg.farfield_p, self.cfg.axisymmetric,
            axis_j(self.cfg, self.ny) - 2.0,
            self.cfg.axis_location == "center", apx=ax_i, apy=ay_i)
        if not self.thrust_history or self.thrust_history[-1][0] != self.step_count:
            self.thrust_history.append((self.step_count, perf["F"]))
        meta = {
            "step": self.step_count, "residual": self.residual,
            "sim_time": self.sim_time, "steps_per_sec": self._steps_per_sec,
            "performance": perf,
        }
        return {"fields": fields, "meta": meta}

    def save_npz(self, path: str):
        snap = self.snapshot()
        np.savez_compressed(path, **{k: v for k, v in snap["fields"].items()},
                            step=snap["meta"]["step"])
