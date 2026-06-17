"""Simulation configuration. All values in SI units."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


# Propellant presets: ideal-gas properties of the combustion products for the
# single-gamma (calorically perfect) solver. Values are chemical-equilibrium
# chamber results at Pc ~= 70 bar and the listed O/F ratio, from NASA CEA /
# Sutton & Biblarz, "Rocket Propulsion Elements", Table 5-5:
#   inlet_T0 = adiabatic flame (chamber) temperature
#   R_gas    = 8314 / mean molar mass of the products [J/(kg K)]
#   gamma    = effective expansion gamma (between the chamber equilibrium
#              value ~1.13-1.15 and the cold-exit frozen value ~1.24; the
#              published "average k" intended for ideal-nozzle analysis)
# The chamber pressure (inlet_p0) is engine-specific and is NOT set here.
PROPELLANTS = {
    "Air (cold gas)": dict(gamma=1.40, R_gas=287.0, inlet_T0=300.0),
    "H2O (steam)":    dict(gamma=1.33, R_gas=461.5, inlet_T0=1200.0),
    "LOX/RP-1 (kerosene)":    dict(gamma=1.22, R_gas=380.0, inlet_T0=3676.0),
    "LOX/CH4 (methalox)":     dict(gamma=1.17, R_gas=389.0, inlet_T0=3550.0),
    "LOX/LH2":                dict(gamma=1.20, R_gas=625.0, inlet_T0=3528.0),
    "LOX/Ethanol (75%)":      dict(gamma=1.21, R_gas=348.0, inlet_T0=3140.0),
    "MMH/NTO (hypergolic)":   dict(gamma=1.25, R_gas=372.0, inlet_T0=3400.0),
    "UDMH/N2O4 (hypergolic)": dict(gamma=1.22, R_gas=352.0, inlet_T0=3415.0),
    "N2O/HTPB (hybrid)":      dict(gamma=1.24, R_gas=323.0, inlet_T0=3200.0),
    "H2O2/RP-1":              dict(gamma=1.21, R_gas=375.0, inlet_T0=2900.0),
}

# (fuel name, oxidizer name, typical O/F mass ratio) for the mass-flow split
PROPELLANT_MIX = {
    "LOX/RP-1 (kerosene)":    ("RP-1", "LOX", 2.27),
    "LOX/CH4 (methalox)":     ("CH4", "LOX", 3.6),
    "LOX/LH2":                ("LH2", "LOX", 5.5),
    "LOX/Ethanol (75%)":      ("Ethanol (75%)", "LOX", 1.4),
    "MMH/NTO (hypergolic)":   ("MMH", "NTO", 2.16),
    "UDMH/N2O4 (hypergolic)": ("UDMH", "N2O4", 2.6),
    "N2O/HTPB (hybrid)":      ("HTPB", "N2O", 7.0),
    "H2O2/RP-1":              ("RP-1", "H2O2 (98%)", 7.0),
}


@dataclass
class SimConfig:
    # --- Geometry ---
    meters_per_pixel: float = 0.001          # 1 mm per pixel -> 1000px = 1 m
    axisymmetric: bool = False               # 2D planar vs axisymmetric
    axis_location: str = "center"            # "center" | "top" | "bottom" (image edge)
    svg_raster_px: int = 1000                # rasterization size for SVG input
    smooth_boundary: bool = True             # cut-cell sub-pixel smooth walls
    boundary_sigma: float = 1.2              # surface smoothing radius [px]
    mesh_scale: float = 1.0                  # mesh density: resample geometry to
                                             #   mesh_scale x the drawn resolution
                                             #   (>1 finer/more cells, <1 coarser).
                                             #   Effective cell size = meters_per_pixel
                                             #   / mesh_scale; physical size is fixed.
    plume_stretch: float = 1.0               # downstream x-mesh stretch ratio
                                             #   per column past the nozzle exit
                                             #   (1.0 = uniform/off; e.g. 1.03 =
                                             #   each plume column 3% wider than
                                             #   the last). Extends the plume
                                             #   domain + keeps near-exit fine so
                                             #   shock diamonds survive. Walls
                                             #   stay on the uniform grid, so
                                             #   engine performance is unchanged.

    # --- Gas properties (default: air / combustion-gas-like ideal gas) ---
    propellant: str = "Custom"               # preset name or "Custom" (record only)
    gas_model: str = "calorically perfect"   # "calorically perfect" (constant
                                             #   gamma) or "thermally perfect"
                                             #   (cp(T) of the frozen chamber
                                             #   mixture; needs a propellant
                                             #   preset with a composition,
                                             #   else falls back to constant)
    gamma: float = 1.4                       # ratio of specific heats [-]
    R_gas: float = 287.0                     # specific gas constant [J/(kg K)]
    mu_ref: float = 1.716e-5                 # Sutherland reference viscosity [Pa s]
    T_ref_sutherland: float = 273.15         # Sutherland reference temperature [K]
    S_sutherland: float = 110.4              # Sutherland constant [K]
    Pr: float = 0.72                         # laminar Prandtl number [-]
    Pr_t: float = 0.90                       # turbulent Prandtl number [-]

    # --- Chamber / pressure inlet (blue pixels) ---
    inlet_p0: float = 2.0e6                  # total (stagnation) pressure [Pa]
    inlet_T0: float = 2800.0                 # ideal total temperature [K]
    eta_cstar: float = 1.0                   # combustion (c*) efficiency [-]:
                                             #   incomplete combustion releases
                                             #   less energy, so the effective
                                             #   chamber temperature is
                                             #   eta^2 * inlet_T0 (c* ~ sqrt(T0)).
                                             #   1.0 = ideal; real engines
                                             #   typically 0.90-0.98 (F-1: 0.93)
    inlet_turb_intensity: float = 0.05       # turbulence intensity [-]
    inlet_mut_ratio: float = 50.0            # mu_t / mu_lam at inlet [-]
    inlet_ramp_steps: int = 1500             # soft-start: ramp p0 over N steps

    # --- Two-gamma plume mixing (exhaust mixing into ambient air) ---
    two_gamma: bool = False                  # transport an exhaust mass
                                             #   fraction and blend gas
                                             #   properties between the exhaust
                                             #   and the ambient gas across the
                                             #   plume mixing layer (CP model)
    ambient_gamma: float = 1.4               # ambient (farfield) gas gamma
    ambient_R: float = 287.0                 # ambient (farfield) gas constant

    # --- Farfield / pressure outlet (domain edges) ---
    farfield_p: float = 101325.0             # static pressure [Pa]
    farfield_T: float = 288.15               # static temperature [K]
    farfield_u: float = 0.0                  # x-velocity [m/s]
    farfield_v: float = 0.0                  # y-velocity [m/s]

    # --- Numerics ---
    wall_emissivity: float = 0.0             # gray-gas radiative emissivity at
                                             #   the wall [-]; 0 = no radiation.
                                             #   Adds q_rad = eps*sigma*(T_gas^4
                                             #   - T_wall^4) to the wall heat
                                             #   flux (needs no-slip + wall_T>0)
    wall_T: float = 0.0                      # isothermal wall temperature [K];
                                             #   0 = adiabatic. Only acts with
                                             #   no-slip walls (wall-function
                                             #   heat flux via Kader's T+)

    carbuncle_fix: bool = True               # blend HLLC->HLL at strong shocks
                                             #   (Ducros-gated) to cure the
                                             #   Mach-disk carbuncle; no effect
                                             #   away from strong shocks
    compressibility_correction: bool = False # Wilcox dilatational-dissipation
                                             #   correction to k-omega SST for
                                             #   high-Mach shear layers (slows
                                             #   plume spreading; opt-in)

    flux_scheme: str = "hllc"                # "hllc" | "hll" | "roe" | "ausm"
    muscl_order: int = 2                     # 1 = first order, 2 = MUSCL
    limiter: str = "minmod"                  # "minmod" or "vanalbada"
    cfl: float = 0.40
    viscous: bool = True
    turbulence: bool = True                  # k-omega SST on/off
    wall_type: str = "slip"                  # "slip" or "noslip"
    local_dt: bool = True                    # local time stepping (steady-state acceleration)
    max_steps: int = 20000
    residual_target: float = 1e-6            # stop when density residual drops below this

    # --- Visualization / IO ---
    viz_interval: int = 25                   # GUI snapshot every N steps

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    # Derived quantities -------------------------------------------------
    @property
    def cp(self) -> float:
        return self.gamma * self.R_gas / (self.gamma - 1.0)

    def sutherland_mu(self, T: float) -> float:
        return (self.mu_ref * (T / self.T_ref_sutherland) ** 1.5
                * (self.T_ref_sutherland + self.S_sutherland) / (T + self.S_sutherland))
