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
  replica in `examples/`. Library expanded to 10 presets — LOX/RP-1,
  LOX/CH₄ (methalox), LOX/LH₂, LOX/Ethanol, MMH/NTO, UDMH/N₂O₄, N₂O/HTPB
  (hybrid), H₂O₂/RP-1, plus air and steam — each with matching equilibrium
  compositions (`thermo.py`) and CEA reactant data (`equilibrium.py`);
  cross-checked flame temperatures (methalox Tc≈3564 K, MMH/NTO≈3412 K,
  N₂O/HTPB≈3473 K, H₂O₂/RP-1≈2991 K).
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
chamber boundary layer in `tests/test_wall_functions.py`. The q_w
distribution is exported as a field ("Wall heat flux [W/m^2]") and overlaid
against the Bartz page in the PDF report.

**DONE (2026-06): gray-gas wall radiation.** `wall_emissivity` config / GUI
entry ("Wall emissivity [-], 0=off", Numerics). When ε>0 (with no-slip +
wall_T>0) the near-wall gas adds a radiative load
q_rad = ε σ (T_gas⁴ − T_wall⁴) on top of the Kader convective flux, in both
the kernel (rk_combine) and the Bartz post-processor (`heatflux.py` splits
`q_conv`/`q_rad`). Verified in `tests/test_radiation.py`: at T_wall=800 K,
ε=0.3 raises peak wall flux ~13.7 → ~14.9 MW/m². Default ε=0 is
bit-identical to before.

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
- **DONE (2026-06):** compressibility correction (Wilcox dilatational
  dissipation) for high-Mach shear layers — `compressibility_correction`
  config / GUI toggle (default off). Lowers eddy viscosity / slows plume
  spreading at high turbulent Mach number; verified in docs/VALIDATION.md.
- **DONE (2026-06):** Ducros-gated **carbuncle cure** for the Mach disk
  (`carbuncle_fix`, default on) — blends HLLC→HLL only at strong shocks.
- Menter's 10·β*ρωk production limiter **is present in the kernel** (verified
  2026-06: `rk_combine` clips S² against 10·β*·ρ·k·ω/μt).
- **DONE (2026-06): configurable eddy-viscosity cap** (`mut_max_ratio`,
  default 1e5 = the classic clamp, bit-identical). Measured on a 20-bar
  LOX/Ethanol bell: SST builds μt/μ ≈ 50 000 (p99) in the plume shear layer,
  which (a) over-mixes the jet — the supersonic core reached 485 mm at 12k
  steps vs **1028 mm laminar** — and (b) dominates the viscous local-dt limit
  in ~55 % of plume cells, so the plume front develops very slowly. Capping
  μt at ~500–2000× laminar restores long shock-diamond plumes and fast
  development while keeping boundary-layer turbulence intact; engine
  thrust/Isp (nozzle interior) are essentially unaffected. RANS jet
  over-spreading is a documented SST weakness; the honest fix (SARC /
  jet-calibrated corrections) remains future work.

### 6. Multi-species plume mixing — lower impact for thrust, high cost

**DONE (2026-06): two-gamma plume mixing.** `two_gamma` config / GUI
checkbox transports an exhaust mass fraction Z (1 = exhaust, 0 = ambient
air) as a conserved scalar riding the density mass flux, with turbulent
diffusion (Sc_t = 0.7) across the shear layer. Implemented as a fully
decoupled `scalar_transport` kernel (double-buffered, launched only when
on), so it changes nothing in the validated 6-equation solver — thrust is
bit-for-bit unchanged. Adds "Mixture fraction" and "Local gamma" fields
(γ blends exhaust→air by mass fraction). Verified: Z bounded [0,1],
exhaust-dominated core, air ambient, mixing layer between, local γ spans
1.20→1.40. Known limitation: the scalar uses 1st-order upwind advection,
so the core reads ~0.85–0.91 (numerically diffused) rather than 1.0;
a MUSCL/WENO reconstruction of Z would sharpen it. Currently a passive
field (γ shown but not fed back into the momentum/energy); dynamic
two-gamma feedback is the natural next extension.

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
- **TENO5 / WENO-Z / WENO9 — evaluated and rejected (2026-06):** three
  higher/sharper reconstructions were fully implemented and measured on the
  standard 40-bar nozzle:
  - *TENO5* (Fu-Hu-Adams, sharp stencil cut-off): blew up at every cut-off
    CT = 1e-5 … 5e-3 (Mach → 1e3). Root cause: TENO recovers the exact linear
    5th-order scheme in smooth regions — zero background dissipation.
  - *WENO-Z* (Borges, q=1): stable, but measured **identical** far-plume
    gradient retention to the existing WENO-JS (274.9 vs 275.1) — an empty
    menu item. (q=6 blows up.)
  - *WENO9* (Balsara-Shu, coefficients derived exactly via rational
    arithmetic, k=3 anchor reproduced the classic WENO5 constants): blew up
    on the same case despite keeping JS-type nonlinear dissipation. 9th-order
    upwind eigenvalues sit too close to the imaginary axis for **SSP-RK2**
    (the literature pairs WENO9 with RK3+), and the 5-point smoothness
    indicators suffer float32 cancellation at MPa pressure levels.

  Conclusion: this solver's float32 + component-wise reconstruction + SSP-RK2
  core tops out at WENO5-JS — its residual smooth-region dissipation is
  load-bearing. The path to higher order is an RK3 time integrator plus
  double-precision smoothness indicators (a major architectural round, noted
  as future work). Sharpness levers that DO measure today: WENO5 (+24 % vs
  MUSCL), van Leer/superbee limiter, mesh resolution, the eddy-viscosity cap.
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
- **Downstream mesh stretching (DONE 2026-06):** `plume_stretch` config /
  GUI field (Geometry). Per-column x-width multiplier grows geometrically a
  few cells past the last wall, so all cut-cell walls stay on the uniform
  grid and the wall-free plume extends downstream for the same cell count.
  Implemented as a kernel metric (`SXW` device array) in the dt limit,
  gradients, viscous fluxes and the rk_combine x-divergence; default off is
  numerically bit-identical (isentropic regression unchanged). Verified:
  enabling stretch on the test nozzle moved thrust 0.31 % and mass flow
  0.16 % (engine on the uniform grid → unchanged, as designed) while the
  plume domain grew 1.32× at ratio 1.03. y is never stretched.
  The field view is remapped (nearest-column) to the TRUE physical length,
  so the stretched plume renders longer (near-field exact, far-field shown
  blocky); hover maps back to the computational column. Exported NPZ also
  carries `x_centers`. Remaining minor polish: the cut-cell mesh overlay and
  replay frames still draw on the uniform rect (only the live field view and
  geometry overlay are remapped).
- **Axisymmetric source terms**: verify the geometric source terms at the
  axis are well-behaved as r → 0 with cut cells (diagnostics exist in
  `tests/diag_axi.py`).
  - **DONE (2026-06): centerline symmetry fix.** The kernel uses a hard 1/r
    reciprocal, so the axis must sit on a cell face (half-integer `AXISJ`).
    For an odd interior row count the axis had to be nudged half a cell off
    the true image center, giving mismatched radii on mirror-image wall rows
    (a visibly lopsided mesh) — and `mesh_scale` resampling could flip the
    parity, so it only appeared on some meshes. `load_mask(axisym_center=…)`
    now duplicates the central row when the count is odd, making it even and
    exactly symmetric about the true center face. Measured asymmetry on an
    odd-height symmetric nozzle: 8.3 % → 0.0 %; even-height meshes are
    unchanged (no regression).

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
