# Making Tachyon CFD more realistic

What the solver models today, what that misses physically, and a ranked
roadmap of improvements. Each item notes the expected accuracy gain and the
implementation cost, so rounds of work can be planned against it.

## What is modelled now

- **Equations** — 2D planar or axisymmetric compressible RANS
  (Favre-averaged Navier–Stokes), finite volume, SSP-RK2 time integration,
  local time stepping for steady-state convergence.
- **Reconstruction** — 1st order, MUSCL 2nd order, or **WENO5** (low
  dissipation; preserves shock-cell trains and shear layers far downstream,
  falls back to MUSCL next to cut-cell walls). Limiters: minmod, van Albada,
  van Leer, superbee.
- **Inviscid fluxes** — HLLC / HLL / Roe / AUSM, MUSCL second order with
  minmod or van Albada limiting.
- **Turbulence** — k-ω SST two-equation model with eddy-viscosity closure,
  turbulent Prandtl number 0.90.
- **Gas model** — single calorically perfect gas: constant γ and R,
  Sutherland viscosity, constant laminar Prandtl number.
- **Walls** — immersed-boundary cut cells with sub-pixel face apertures and
  volume fractions; slip or no-slip.
- **Heat transfer** — post-hoc Bartz correlation along the wall contour
  (`rocketcfd/heatflux.py`); the wall itself is adiabatic in-solver.

## Gaps, ranked by impact on rocket-nozzle answers

### 1. Variable gas properties (γ(T), cp(T)) — high impact, moderate cost

A combustion gas at 2800–3600 K is not γ = 1.4 air. Real exhaust has
γ ≈ 1.13–1.25 and it *varies* through the nozzle as T drops. Since exit
velocity and thrust scale directly with γ and R, this is the single largest
error in absolute Isp today (easily 5–15 %).

- **DONE (Tier A, 2026-06):** the propellant presets in `config.py` now
  carry CEA/Sutton equilibrium chamber values (γ, R from molar mass, T0)
  at Pc ≈ 70 bar, including a LOX/RP-1 preset validated against the F-1
  replica in `examples/`.
- **DONE (Tier B, 2026-06):** thermally perfect gas — selectable "Gas
  model" in the GUI. cp(T) of the frozen chamber mixture (JANAF species
  data, CEA chamber compositions in `rocketcfd/thermo.py`), cubic-fitted
  and baked into the CUDA kernels; T recovered from energy by warm-started
  Newton. The calorically perfect path is compile-time guarded and
  regression-verified bit-identical. F-1 at 9 km: Isp +1.9 % vs real
  (CP: +3.0 %). Known limitation: frozen γ over-expands high-ε nozzles at
  sea level into separation (see docs/VALIDATION.md) — Tier C fixes this.
- Beyond single-point presets: a small O/F → (γ, R, T0) lookup table per
  propellant covers off-nominal mixture ratios without a full CEA port.

### 2. Frozen vs equilibrium chemistry — high impact, high cost

**DONE (Tier C, 2026-06):** shifting-equilibrium gas model — third entry
in the GUI "Gas model" combo. A mini-CEA (`rocketcfd/equilibrium.py`,
Gibbs minimization over C-H-O-N products, JANAF data, chemistry frozen
below 900 K) precomputes (log ρ, log T) property tables that the kernels
sample with warm-started Newton inversion; the inlet uses a baked
chamber-isentrope table. Validated on the F-1: sea-level thrust within
0.2 % of real, exit pressure ≈ 50 kPa (fixes the frozen-gas separation
artifact), Isp = ideal ceiling ~+16 % over the real engine (combustion
efficiency and kinetics freezing are deliberately not modeled — see
docs/VALIDATION.md for how the three gas models bracket reality).
Remaining beyond this tier: finite-rate kinetics (out of scope).
**DONE (2026-06): combustion-efficiency knob** — `eta_cstar` config
field / GUI entry, T0_eff = η²T0 in all gas models + Bartz. F-1 at 9 km
with the documented η=0.93: mass flow −1.6 %, Isp −3.4 % vs the real
engine (see docs/VALIDATION.md, including the sea-level caveat).

### 3. In-solver wall thermal boundary condition — medium impact, low cost

**DONE (2026-06, bundled with item 4):** `wall_T` config field / GUI entry
("Wall temperature [K], 0=adiab.", Numerics group). With no-slip walls,
the energy equation loses heat to the wall via q_w = ρ cp u_τ (T−T_w)/T⁺
(Kader's law) on the embedded cut-cell segments. Verified to cool the
chamber boundary layer in `tests/test_wall_functions.py`. Future: export
the q_w distribution as a field for direct comparison with the Bartz page
in the PDF report.

### 4. Wall functions or y⁺-aware near-wall treatment — medium impact, medium cost

**DONE (2026-06):** no-slip walls now use Reichardt's law of the wall for
the embedded-segment shear (valid from the viscous sublayer through the
log layer, so τ_w no longer depends on where y⁺ lands on the cut-cell
mesh), Menter's automatic blended ω wall treatment, and zero-gradient k.
Companion feature (REALISM item 3): `wall_T` config field — isothermal
walls with Kader's thermal wall law give an in-solver wall heat flux
(0 = adiabatic, the default). Slip mode is untouched and remains the
default for performance numbers. Validated on the F-1 at 9 km: no-slip
matches slip within convergence noise (friction is sub-1 % on an engine
this large — physically correct); at over-expanded sea level the no-slip
boundary layer separates earlier than slip (real BL physics responding
to the adverse pressure gradient). See docs/VALIDATION.md.

**Measured 2026-06 (see docs/VALIDATION.md):** the smeared wall layer also
lets ambient pressure creep upstream and recompress the supersonic
expansion — exit Mach reads −2.5 % at 5 kPa ambient but −19 % at ~1 atm on
a 30-px-throat test nozzle. This is currently the largest *flow-field*
error at sea-level conditions and is the concrete payoff case for this
item (integral thrust/Isp are much less affected).

### 5. Turbulence model limitations — medium impact, varies

- SST is calibrated for attached boundary layers; the free shear layer of
  the plume and over-expanded separation are weak spots.
- Compressibility correction (Sarkar/Zeman dilatational dissipation) for
  high-Mach shear layers: cheap, improves plume spreading rates.
- A realizability / production limiter at shocks (Menter's 10·β*ρωk clip is
  present in most SST variants — verify it's in the kernel) prevents
  spurious turbulence generation in the shock train of over-expanded flow.

### 6. Multi-species plume mixing — lower impact for thrust, high cost

Thrust hardly cares, but plume structure does: exhaust (low γ, hot, light)
mixing into ambient air is currently one gas. A passive scalar with distinct
γ/R blending (two-gamma model) gets a more realistic shear layer and
afterburning-free plume shape for a moderate kernel change.

### 7. Time accuracy for unsteady phenomena — niche

Local time stepping targets steady state. Side-load studies (flow separation
transients during startup, screech) need global Δt, dual time stepping, and
a DES/LES-grade turbulence treatment — a different tool tier. Document as
out of scope unless requested.

## Numerics quick wins (cheap, do alongside)

- **Convergence quality**: report iterative convergence of *thrust* (already
  tracked in `thrust_history`) and warn when reporting from an unconverged
  field — e.g. require the last 10 % of the history to vary < 0.5 % before
  the PDF report calls a number "converged".
- **Mesh-independence helper**: with `mesh_scale` in place, an automated
  2-point Richardson check (run at 1× and 1.5×, report the thrust delta)
  quantifies discretization error per engine.
- **Reconstruction / limiters (DONE 2026-06):** added **WENO5** (5th-order,
  far lower dissipation — preserves ~24 % more downstream shock-cell
  structure than MUSCL-minmod on the test nozzle) plus **van Leer** and
  **superbee** limiters. Compressiveness ordering verified (Mach max:
  minmod < van Albada < van Leer < superbee). WENO5 runs in the interior
  fluid and falls back to MUSCL at cut-cell walls/edges; default
  MUSCL/minmod path is numerically unchanged. Note: van Albada still
  destabilizes transonic startup — start on minmod/van Leer if it stalls.
  Reducing `inlet_mut_ratio` from 50 to 5–10 is a complementary dissipation
  win (SST inlet eddy viscosity costs ~5–8 % of choked mass flow).
- **Downstream mesh stretching (NEXT ROUND):** the grid is uniform
  (scalar `DX`), so the plume gets throat-resolution everywhere and wastes
  cells on far ambient. True stretching means replacing the compile-time
  `DX` with per-cell/face metric arrays across every kernel + the postproc
  and GUI coordinate mappings — a dedicated architectural round. Biggest
  remaining lever for resolving long diamond trains cheaply.
- **Axisymmetric source terms**: verify the geometric source terms at the
  axis are well-behaved as r → 0 with cut cells (diagnostics exist in
  `tests/diag_axi.py`).

## Validation cases worth wiring into `tests/`

1. **DONE (2026-06):** 1D converging–diverging nozzle vs isentropic theory
   — `tests/test_isentropic.py`: choked mass flow −1.3 %, plane-to-plane
   conservation 0.75 %, centerline exit Mach −11 % (known wall-layer bias,
   bounded and documented in docs/VALIDATION.md).
2. **NASA SP-8120 / Rao nozzle thrust coefficients** — compare C_F against
   published curves at several pressure ratios.
3. **Back-pressure sweep on a known bell** — separation onset vs Summerfield
   criterion (p_wall ≈ 0.35–0.4 p_amb).
4. **Bartz vs published throat heat flux** — e.g. SSME-class numbers,
   order-of-magnitude agreement check (already a self-test in
   `heatflux.py`; promote to a pytest).
