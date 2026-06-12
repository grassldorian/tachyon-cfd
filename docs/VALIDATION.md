# Validation record

Measured accuracy of Tachyon CFD against real-engine data and analytic
theory. Update this table whenever the gas model, numerics, or wall
treatment changes.

## F-1 engine replica (examples/f1_engine.png, sea level)

Published Rocketdyne F-1: throat radius 0.4445 m, area ratio 16, Pc ≈ 7.0
MPa, LOX/RP-1 at O/F 2.27. Real sea-level performance: F = 6.77 MN,
Isp = 263 s, mdot ≈ 2578 kg/s.

Runs: 16 000 steps, 1000x1000 grid (dx = 7 mm), k-ω SST, HLLC/MUSCL2,
axisymmetric, `--config examples/f1_config.json`.

| Quantity | old hand-tuned gas (γ=1.22, R=357, T0=3570) | CEA preset (γ=1.22, R=380, T0=3676) | real F-1 |
|---|---|---|---|
| Thrust | 6.21 MN (−8.3 %) | 6.21 MN (−8.3 %) | 6.77 MN |
| Mass flow | 2546 kg/s (−1.2 %) | 2431 kg/s (−5.7 %) | 2578 kg/s |
| Isp | 248.6 s (−5.5 %) | **260.4 s (−1.0 %)** | 263 s |

Reading the gaps honestly:

- **Isp −1 %** is partly fortunate: ideal 1D analysis of this gas/geometry
  gives ≈ 285 s; the real engine loses ~8 % to combustion inefficiency
  (c* efficiency ≈ 0.93) which the CFD does not model, while the CFD loses
  a similar amount to numerical dissipation, the smeared wall layer, and
  the replica's simplified bell contour. The two error families happen to
  nearly cancel at sea level.
- **Mass flow −5.7 %**: the CFD flows the *theoretical* choked mdot for
  the CEA gas; the real engine flows ~6 % more because combustion
  inefficiency lowers its effective T0 (mdot ∝ 1/√T0). Expected, not a bug.
- **Thrust −8.3 %**: dominated by the replica geometry (a `t^0.7` parabola
  vs the real Rao 80 % bell) and the same loss families as Isp.

## Gas-model comparison: F-1 at 9 km (2026-06, thermally perfect round)

Same engine and grid, 24 000 steps, ambient 30.8 kPa / 229.7 K. Real F-1
at 9 km ≈ 7.47 MN / 295 s (interpolated from published SL and vacuum data).
9 km is the thermally-perfect design-matched altitude (exit pressure ≈
ambient), so both models converge cleanly there (residual < 6e-2, thrust
flat to ±0.5 %).

| Quantity | calorically perfect (γ=1.22) | thermally perfect cp(T), frozen | real F-1 |
|---|---|---|---|
| Thrust | 7.19 MN (−3.7 %) | 7.09 MN (−5.0 %) | ≈7.47 MN |
| Isp | 304.4 s (+3.0 %) | **300.9 s (+1.9 %)** | ≈295 s |
| Mass flow | 2409 kg/s | 2404 kg/s | 2578 kg/s |

Notes:

- Both models sit *above* real Isp at altitude, as they should — the CFD
  does not model combustion inefficiency (real c* efficiency ≈ 0.93). The
  thermally perfect gas removes ~1 % of artificial overprediction by
  letting γ rise (1.215 → ~1.30) as the gas cools through the expansion.
- Real mass flow is ~7 % above both models: combustion inefficiency lowers
  the real effective T0 (ṁ ∝ 1/√T0). Expected, not a bug.
- **Sea-level limitation:** with frozen composition the F-1's exit pressure
  drops to ≈30 kPa, below the Summerfield separation limit (≈0.35 atm) —
  the nozzle physically separates and the flow goes unsteady (thrust
  snapshots are meaningless). The real engine avoids this because
  recombination (shifting equilibrium, REALISM Tier C) keeps the effective
  γ — and hence exit pressure — higher. Use the calorically perfect model
  for sea-level runs of high-expansion engines, or the thermally perfect
  model at/above its design altitude.

## Equilibrium (shifting) gas model: F-1 (2026-06, Tier C round)

Same engine and grid, 20 000 steps, `gas_model = "equilibrium"`. The
composition recombines as the gas expands (mini-CEA property tables,
`rocketcfd/equilibrium.py`), which keeps the exit pressure high enough
that **the sea-level case converges — the frozen-gas separation artifact
is gone** (exit pressure ≈ 50 kPa vs ≈ 30 kPa frozen, Summerfield limit
≈ 35 kPa).

| Quantity | equilibrium CFD | ideal 1D equilibrium | real F-1 |
|---|---|---|---|
| Sea level: thrust | **6.78 MN** | 6.92 MN | 6.77 MN |
| Sea level: Isp | 305.8 s | 301.4 s | 263 s |
| 9 km: thrust | 7.31 MN | 7.62 MN | ≈7.47 MN |
| 9 km: Isp | 349.7 s | ≈332 s | ≈295 s |
| Mass flow | 2130–2260 kg/s | 2342 kg/s | 2578 kg/s |

How to read this: the equilibrium model is an **ideal-engine ceiling** —
perfect combustion and infinitely fast chemistry. The real F-1 sits
~14 % below it in Isp: ≈7 % combustion inefficiency (c* efficiency 0.93),
~2-4 % nozzle/divergence losses, and ~2-4 % because real chemistry
freezes mid-expansion (between this model and the frozen one). The three
gas models bracket reality:

    frozen (thermally perfect)  <  real engine  <  shifting equilibrium

and the calorically perfect preset (γ=1.22 "average") happens to sit
closest to real numbers because its tuned γ partially bakes those losses
in. Use CP for engineering numbers, equilibrium for the theoretical
ceiling and for correct exit-pressure/separation behavior, frozen TP for
the conservative bound.

Convergence note: equilibrium runs settle more slowly (thrust still
±3-4 % at 20 k steps; ~30 % lower steps/s than CP). Run 30 k+ steps for
tight numbers.

## Combustion-efficiency knob (eta_cstar, 2026-06)

`eta_cstar` (GUI: "Combustion eff. η_c*", Chamber inlet group) models
incomplete combustion as a reduced effective chamber temperature,
T0_eff = η² · T0 (since c* ∝ √T0). Applied in all three gas models and
the Bartz estimate; η = 1 is bit-identical to before.

F-1 at 9 km, equilibrium model with the documented η_c* = 0.93
(30 000 steps, converged to ±0.2 %):

| Quantity | equilibrium, η=0.93 | equilibrium, η=1 (ideal) | real F-1 |
|---|---|---|---|
| Mass flow | **2536 kg/s (−1.6 %)** | 2133–2260 kg/s | 2578 kg/s |
| Isp | **284.9 s (−3.4 %)** | ~340-350 s | ≈295 s |
| Thrust | 7.09 MN (−5.1 %) | 7.31 MN | ≈7.47 MN |

With one documented engine parameter, the equilibrium model moves from
"ideal ceiling" (+16 %) to −3.4 % on Isp and −1.6 % on mass flow. The
remaining Isp deficit is expected: the η²T0 model slightly over-cools
the expansion (real unburned propellant does not cool the bulk gas
uniformly), stacked on the solver's wall-layer/dissipation losses.
Together with the tuned-CP model (+3.0 %), the suite brackets the real
engine within ±3.5 %.

Caveat: lowering T0 also lowers dissociation, hence less recombination
heat and a lower exit pressure — at sea level the η=0.93 F-1 drops to
pe ≈ 38 kPa and (marginally, like the real engine nearly did) separates
in-model. Use η-corrected equilibrium runs at/above the engine's design
altitude; at sea level prefer η=1 equilibrium or the CP preset.

## Wall functions / no-slip walls (2026-06)

No-slip walls now use Reichardt's law of the wall (τ_w = ρu_τ², valid at
any y⁺), Menter's blended ω wall treatment, and an optional isothermal
wall temperature (`wall_T`, Kader's thermal law). F-1, calorically
perfect gas:

| Case | slip | no-slip + wall functions |
|---|---|---|
| 9 km (attached) | 7.19 MN / 304.4 s | 7.20 MN / 306.7 s |
| Sea level (over-expanded) | 6.21 MN / 260.4 s, attached | 6.77 MN / 287.7 s, partially separated |

Readings:

- **At 9 km the two agree within convergence noise** — physically correct:
  on an engine with a 0.89 m throat the boundary layer costs well under
  1 % of thrust.
- **At sea level the no-slip boundary layer separates earlier than slip**
  (a real boundary layer responds to the adverse pressure gradient of an
  over-expanded bell; slip walls cannot separate). The separated state
  produces *more* thrust than full-flowing — standard over-expanded
  nozzle behavior — but the flow is unsteady, so treat SL no-slip numbers
  as regime-exploration, not performance quotes.
- Recommendation: slip for performance numbers (validated), no-slip for
  boundary-layer/separation studies and wall heat flux (`wall_T` > 0).
- **Bartz cross-check:** with `wall_T` set, the in-solver wall-function
  heat flux is exported as the "Wall heat flux [W/m²]" field and overlaid
  on the Bartz page of the PDF report. On the small test nozzle the two
  independent methods agree at the throat within ~3 %
  (48.5 vs 49.9 MW/m²).

## Isentropic CD nozzle (tests/test_isentropic.py)

Area-ratio-4 axisymmetric nozzle at design pressure ratio, laminar,
24 000 steps. Asserted bounds in parentheses.

| Check | Result |
|---|---|
| Choked mass flow vs theory | −1.3 % (±3 %) |
| Mass conservation across 5 planes | 0.75 % spread (±6 %) |
| Centerline exit Mach vs area-ratio relation | −11.4 % ([−15 %, +3 %]) |

## Known solver biases (measured 2026-06)

1. **Back-pressure creep**: ambient pressure leaks upstream through the
   numerically smeared subsonic wall layer and recompresses the supersonic
   expansion. Measured exit-Mach bias on the CD-nozzle family: −2.5 % at
   5 kPa ambient, −9 % at 30 kPa, −19 % at 98.6 kPa. Mass flow is immune
   (choking happens upstream). Mitigation: finer mesh (`mesh_scale`), and
   eventually wall functions (REALISM.md item 4).
2. **SST inlet eddy viscosity cost**: with `inlet_mut_ratio = 50`, the
   turbulence model alone removes ~5–8 % of choked mass flow via total
   pressure loss through the contraction. Consider lowering
   `inlet_mut_ratio` (5–10) for engines with smooth feed lines.
3. **van Albada limiter destabilizes** transonic nozzle startups in this
   solver (shock transits the bell, residual stalls ~1). Keep minmod for
   production runs despite its dissipation.
4. **Roe + equilibrium gas is invalid** (perfect-gas eigenvectors inject
   energy at contacts with a tabular EOS — measured Mach ~29 garbage).
   The solver auto-falls back to HLLC and prints a note. Roe works
   normally with the calorically/thermally perfect models.
