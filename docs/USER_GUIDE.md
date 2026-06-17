# Tachyon CFD — Control Panel Guide

This guide explains **every setting on the left-hand control panel** (and the
action bars around it), what it physically means, and how to choose a value.
Settings are grouped exactly as they appear in the app.

> **Golden rule:** edits to the panel only take effect when you press
> **⟲ Initialize**. Initialize rebuilds the mesh and recompiles the GPU
> kernels for the current options; **Run** then time-marches the solver.

---

## Top of the panel — load & run

| Button | What it does |
|--------|--------------|
| **Load engine…** | Open a PNG/SVG drawing of your engine (or load one you drew in the **Designer** tab). The status line reports the grid size and the fluid/inlet cell counts. |
| **⟲ Initialize** | Build the mesh from the drawing + current settings and compile the kernels. Press this after *any* panel change. |
| **▶︎ Run / ⏸︎ Pause** | Start/stop time-marching. The fields, the thrust history and the residual plot update live every *N* steps (see *Run control*). |

The drawing uses four colors: **black = wall**, **white = flow**,
**blue = chamber/pressure inlet**, **red = pressure outlet** (absorbs waves at
ambient pressure). Image edges are farfield boundaries.

---

## Geometry

Controls how the drawing becomes a mesh and the physical scale.

- **Meters per pixel [m]** — physical size of one pixel = one base finite-volume
  cell. This is your master length scale. Example: a 600 px-wide nozzle at
  `0.002` m/px is 1.2 m wide. Set it from a known dimension (throat or exit
  diameter) and count pixels.
- **Mesh density ×** — resamples the drawing to this multiple of its drawn
  resolution *without* changing the physical size. `1` = as drawn, `2` = half
  the cell size (4× the cells, ~4× slower, sharper boundary layers and shocks),
  `0.5` = coarser/faster. Use it to do a mesh-independence check: if thrust
  barely moves from `1`→`2`, you are mesh-converged.
- **Plume stretch (1=off)** — geometric growth ratio applied to the columns
  *downstream of the nozzle exit only*. `1.0` = uniform grid (off). `1.03` means
  each plume column is 3 % wider than the last, so the domain reaches much
  further downstream while staying fine near the exit — shock diamonds survive
  longer for the same cost. Walls stay on the uniform grid, so **engine
  performance is unaffected**; this only buys you plume length.
- **SVG raster size [px]** — when the input is an SVG, its long side is
  rasterized to this many pixels before meshing. Higher = smoother curves from
  vector input. Ignored for PNG input.
- **☑ Smooth sub-pixel walls (cut cells)** — convert the staircase of black
  pixels into a smooth embedded surface (level-set apertures + volume
  fractions). **Keep this on** for curved nozzles; turning it off gives raw
  pixel walls (only useful for debugging). Keep walls ≥ 4 px thick.
- **☑ Axisymmetric (rocket engine)** — treat the 2-D drawing as a body of
  revolution about a symmetry axis (adds the axisymmetric source terms). **On**
  for real round engines → thrust/Isp are physical. **Off** = planar 2-D
  (per-metre-of-span, for wedges/2-D test cases).
- **Symmetry axis** — where the rotation axis sits:
  - *image center* — you drew the **full** cross-section; the axis runs through
    the middle. (Tachyon guarantees an exactly centered, symmetric mesh here.)
  - *top edge* / *bottom edge* — you drew **half** the engine and the axis is
    that image edge (a symmetry plane). Saves half the cells.

---

## Gas

Defines the working fluid. Start by picking a **Propellant** and a **Gas
model**; the property fields auto-fill but stay editable.

- **Propellant** — preset combustion-gas properties (γ, R, chamber T₀) from
  NASA-CEA / Sutton & Biblarz at ~70 bar:
  Air (cold gas), H₂O steam, LOX/RP-1, LOX/CH₄ (methalox), LOX/LH₂,
  LOX/Ethanol, MMH/NTO, UDMH/N₂O₄, N₂O/HTPB (hybrid), H₂O₂/RP-1, or **Custom**.
  Selecting one fills γ, R and T₀ for you (chamber pressure is engine-specific,
  set it yourself).
- **Gas model** — how thermodynamics are computed:
  - *Calorically perfect (constant γ)* — fixed γ and cₚ. Fastest, classic
    nozzle theory. Great default.
  - *Thermally perfect (cp(T), frozen mix)* — cₚ and γ vary with temperature
    using the **frozen** chamber composition of the chosen propellant. More
    accurate temperatures; needs a combustion preset.
  - *Equilibrium (shifting, recombination)* — the composition re-equilibrates
    (recombines) as the gas expands and cools, via the built-in mini-CEA. **Most
    realistic Isp and exit pressure**; needs a combustion preset; chemistry is
    frozen below ~900 K.
- **Heat capacity ratio γ [-]** — ratio of specific heats. ~1.4 cold air;
  ~1.13–1.25 for hot combustion products. Drives the area–Mach relation.
- **Gas constant R [J/(kg·K)]** — `8314 / mean molar mass` of the products.
  Lighter exhaust (more H₂) → larger R → higher Isp.
- **Sutherland μ_ref [Pa·s]** — reference dynamic viscosity for the Sutherland
  law (sets the boundary-layer thickness and wall shear). Default `1.716e-5`.
- **Prandtl number [-]** — ratio of momentum to thermal diffusivity; sets how
  wall heat flux relates to wall shear. ~0.72 for combustion gas.
- **Ambient gas γ [-]** and **Ambient gas R [J/(kg·K)]** — properties of the
  *surrounding* gas (the air the plume expands into). Only used by **Two-gamma
  plume mixing**.
- **☑ Two-gamma plume mixing (exhaust + air)** — transport an exhaust
  mass-fraction scalar and blend gas properties between exhaust and ambient
  across the plume shear layer. Adds *Mixture fraction* and *Local gamma* view
  fields. The pure-exhaust engine core is unchanged; this only improves the
  far-plume mixing region.

---

## Chamber inlet (blue)

Conditions imposed on the **blue** pixels — the stagnation (total) state of the
combustion chamber.

- **Total pressure p₀ [Pa]** — chamber stagnation pressure. The single biggest
  thrust knob (thrust ≈ proportional to p₀). E.g. F-1 ≈ `7.0e6` (70 bar).
- **Total temperature T₀ [K]** — *ideal* chamber stagnation temperature
  (adiabatic flame temperature). Auto-filled by the propellant preset.
- **Combustion eff. η_c\* [-]** — characteristic-velocity efficiency for
  incomplete combustion. Since c\* ∝ √T₀, the solver uses an **effective
  T₀ = η²·T₀**. `1.0` = ideal ceiling; real engines ~0.90–0.98 (F-1 ≈ 0.93).
  Use this to turn textbook numbers into real-engine predictions.
- **Turbulence intensity [-]** — inlet turbulence level (fraction of velocity)
  seeding the k-ω SST model. `0.05` (5 %) is a sensible default.
- **Soft-start ramp [steps]** — ramp p₀ from ambient up to the set value over
  this many steps. Prevents a startup shock from blowing up the solve. Increase
  it (e.g. 1500 → 3000) if a high-pressure case diverges in the first hundreds
  of steps.

---

## Farfield / outlet (edges)

The static state of the external environment, applied at domain edges and the
**red** outlet pixels.

- **Static pressure [Pa]** — ambient back-pressure. `101325` = sea level;
  lower it for altitude (e.g. `~30 000` at ~9 km, near-vacuum for upper stages).
  This sets whether the nozzle is over/under-expanded and where shock diamonds
  form.
- **Static temperature [K]** — ambient temperature (mainly affects the
  entrained-air state and plume mixing). `288.15` = sea level standard.

---

## Numerics

How the equations are discretized. Defaults are robust; change these to trade
speed for sharpness or to stabilize a hard case.

**Wall thermal settings**
- **Wall temperature [K], 0=adiab.** — `0` = adiabatic wall (no heat loss).
  Set a value (e.g. `800`) for an **isothermal** wall; the solver then computes
  the gas-side wall heat flux via the Kader thermal wall law (no-slip only).
- **Wall emissivity [-], 0=off** — gray-gas **radiative** load added on top of
  convection: `q_rad = ε·σ·(T_gas⁴ − T_wall⁴)`. `0` = off. Needs a no-slip wall
  and a non-zero wall temperature. Typical 0.2–0.4 for sooty/hot walls.

**Discretization**
- **CFL number [-]** — time-step safety factor. `0.4` default. Lower (0.2–0.3)
  = more stable / slower; higher = faster but risk of divergence.
- **Riemann solver** — the flux at cell faces:
  - *HLLC* — best all-rounder (resolves contact & shear). **Default.**
  - *HLL* — most robust, more diffusive (smears contacts).
  - *Roe* — sharp, perfect-gas only (auto-falls back to HLLC in equilibrium
    mode).
  - *AUSM+* — low-dissipation, good for shocks; can be lively on coarse meshes.
- **Spatial order** — reconstruction:
  - *2nd order (MUSCL)* — standard, good default.
  - *1st order* — very diffusive; only for debugging/first convergence.
  - *5th order (WENO)* — much lower dissipation (shock diamonds & shear layers
    persist far downstream). Slower per step; pairs well with a fine mesh. Runs
    in the interior and falls back to MUSCL next to walls.
- **Limiter** (MUSCL only) — slope limiter, least → most compressive:
  *minmod* (robust, diffusive) · *van Albada* · *van Leer* (sharp, good
  default) · *superbee* (sharpest, can over-steepen). If a case wobbles near
  shocks, step back toward minmod.
- **Wall condition** —
  - *no-slip* — viscous wall with Reichardt wall functions; required for wall
    shear/heat flux and realistic boundary layers.
  - *slip* — inviscid wall (frictionless); fastest, fine for pure
    performance/inviscid-core estimates.

**Physics toggles**
- **☑ Viscous (Navier–Stokes)** — include viscous stresses. Off = Euler
  (inviscid). Keep on for real boundary layers and heat flux.
- **☑ Turbulence model (k-ω SST)** — Menter SST turbulence. On for turbulent
  rocket flows; off = laminar.
- **☑ Local time stepping (steady)** — each cell advances at its own max stable
  step to reach steady state much faster. **Keep on** for steady runs; turn off
  only if you want a time-accurate transient.
- **☑ Carbuncle cure (HLLC shocks)** — blends HLLC→HLL *only* at strong shocks
  (Ducros-gated) to kill the Mach-disk "carbuncle" instability. No effect away
  from strong shocks; safe to leave on.
- **☑ Compressibility correction (SST)** — Wilcox dilatational-dissipation
  correction for high-Mach shear layers; slows plume spreading to better match
  experiment. Opt-in (off by default).

---

## Run control

When to stop and how often to refresh.

- **Residual target [-]** — stop when the density residual drops below this
  (steady-state convergence). `1e-6` is tight; `1e-4` is often enough for
  thrust/Isp. The **convergence indicator** turns green when reached.
- **Max steps** — hard cap on iterations regardless of residual.
- **GUI update every N steps** — snapshot/redraw cadence. Larger = faster
  (less time spent drawing), smaller = smoother live animation.

---

## Bottom action bar

| Button | What it does |
|--------|--------------|
| **Save config… / Load config…** | Write/read all panel settings as JSON. |
| **Export NPZ…** | Save the raw field arrays (density, velocity, p, T, Mach, …) for offline analysis. |
| **Export view PNG…** | Save the current field view as an image. |
| **Altitude sweep…** | Re-run the converged engine across a range of ambient pressures and plot thrust & Isp vs altitude (great for nozzle/aerospike comparisons). |
| **Report PDF…** | Multi-page PDF: geometry, fields, performance, convergence and a Bartz heat-flux cross-check. |
| **Export video MP4…** | Render the recorded run history to an MP4. |

---

## Field view toolbar (above the plot)

- **Field selector** — choose what to color by (Mach, pressure, temperature,
  density, velocity, wall heat flux, and—if two-gamma is on—mixture fraction /
  local gamma).
- **Colormap** + **min / max** — palette and manual color-scale limits (blank =
  autoscale).
- **Show mesh** — overlay the smooth embedded wall surface and, when zoomed,
  the cell edges.
- **Probe** — click two points to plot p, Mach and T along that line; the dialog
  has centerline and wall-pressure presets.
- **◐ (color scheme)** — pick the GUI theme: Mono (B&W), Light, Dark,
  Blueprint, or Midnight.

---

## A typical rocket setup (quickstart)

1. **Load engine…** your nozzle PNG (black walls, blue chamber face, red along
   the downstream/edge borders).
2. **Geometry:** set *Meters per pixel* from a known diameter; tick
   *Axisymmetric*, *Symmetry axis = image center*, keep *Smooth walls* on.
3. **Gas:** pick your *Propellant* (e.g. LOX/RP-1) and a *Gas model*
   (*Equilibrium* for best Isp).
4. **Chamber inlet:** set *Total pressure p₀* (e.g. 7 MPa); leave T₀ from the
   preset; set *η_c\** ≈ 0.93.
5. **Farfield:** set ambient *Static pressure* for your altitude.
6. **Numerics:** HLLC + 2nd order MUSCL (van Leer), no-slip walls, viscous +
   turbulence on.
7. **⟲ Initialize**, then **▶︎ Run**. Watch the residual fall and the thrust
   history flatten; read thrust/Isp from the performance box and export a
   **Report PDF**.
