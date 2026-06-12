# Tachyon CFD

GPU-accelerated 2D / axisymmetric compressible RANS solver for rocket-engine
nozzles. Draw your engine as a PNG or SVG (or in the built-in vector
designer), set chamber conditions, press Run — get thrust, Isp, fields,
wall heat flux and a PDF report.

Validated against the Rocketdyne F-1: thrust within 0.2 %, Isp bracketed
within ±3.5 % by the included gas models (see
[docs/VALIDATION.md](docs/VALIDATION.md)).

## Highlights

- **Runs on the GPU** — custom CUDA kernels via CuPy (float32); ~100
  steps/s on a 1M-cell grid on a GTX 1060.
- **Three gas models**: calorically perfect (constant γ), thermally
  perfect (cp(T) of the frozen chamber mixture), and **shifting
  equilibrium** — a built-in mini-CEA computes chamber equilibrium,
  recombination chemistry and property tables per propellant.
- **Combustion efficiency knob** (η_c*) to go from ideal-ceiling numbers
  to real-engine predictions.
- **Wall functions** (Reichardt + Menter automatic wall treatment) and
  **isothermal walls** with in-solver wall heat flux — cross-validated
  against the Bartz correlation within 3 %.
- **Engine designer** tab: draw nozzles/aerospikes as editable vector
  geometry (lines, splines, freehand) with mm rulers.
- **Analysis tools**: two-click line probe (centerline / wall-pressure
  presets), altitude sweep (thrust & Isp vs altitude — aerospike
  comparisons), convergence gate, multi-page PDF report, MP4 export,
  3D exhaust view.

## Install (from source)

Requires an NVIDIA GPU with a CUDA 12.x driver and Python 3.11+.

```
pip install -r requirements.txt
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
python run_gui.py
```

The two `nvidia-*` wheels provide the runtime CUDA compiler (NVRTC) and
headers — no CUDA Toolkit installation needed. The "CUDA path could not be
detected" warning at startup is benign.

Headless runs: `python -m rocketcfd.headless engine.png --config cfg.json
--steps 20000`. Altitude sweep CLI: `python -m rocketcfd.sweep`.

## Windows EXE

A standalone build (no Python required) is produced with:

```
pyinstaller packaging\RocketCFD.spec --noconfirm
dist\TachyonCFD\TachyonCFD.exe --selftest    # verify GPU pipeline
```

The installer is built from that with [Inno Setup](https://jrsoftware.org/isinfo.php):

```
iscc packaging\tachyon_installer.iss
```

## Input format

A PNG image (any size; 1 pixel = 1 finite-volume cell) or an SVG drawing
(rasterized automatically):

| Color | Meaning |
|-------|---------|
| **black** | wall |
| **white** | flow space |
| **blue**  | pressure inlet (chamber total conditions p₀, T₀) |
| **red**   | pressure outlet (absorbs waves at ambient pressure) |

Image edges act as farfield boundaries; draw red strips along the borders
to absorb startup shocks. Physical scale is set by *meters per pixel*;
*Mesh density* refines the grid without changing the drawing.

The drawing becomes a **cut-cell mesh with a smooth embedded boundary** —
sub-pixel level-set walls with aperture-weighted fluxes, so curved nozzle
contours behave like real surfaces instead of pixel staircases. Keep walls
≥ 4 px thick. Toggle **Axisymmetric** for round engines (axis through the
image center or a top/bottom edge).

## Physics & numerics

- Compressible RANS, finite volume, SSP-RK2 with local time stepping
- Fluxes: HLLC (default), HLL, Roe, AUSM+ — MUSCL 2nd order
  (minmod / van Albada)
- k-ω SST turbulence (Menter), wall-distance based, point-implicit sources
- Gas models: constant γ / cp(T) frozen mixture (JANAF) / shifting
  equilibrium (Gibbs minimization, tabulated for the GPU; chemistry frozen
  below 900 K)
- Propellant presets with NASA-CEA chamber properties: LOX/RP-1, LOX/LH2,
  LOX/Ethanol, UDMH/N2O4 (+ air, steam, custom)
- Walls: slip, or no-slip with Reichardt wall functions; optional
  isothermal wall temperature with Kader heat flux
- Axisymmetric source terms, Sutherland viscosity, inlet soft-start

Known accuracy envelope and measured solver biases are documented honestly
in [docs/VALIDATION.md](docs/VALIDATION.md); the realism roadmap and
completed tiers are in [docs/REALISM.md](docs/REALISM.md).

## Tests

Plain-script tests in `tests/` (GPU required for most):

```
python tests\test_isentropic.py        # choked flow vs theory
python tests\test_thermo_gas.py        # gas models
python tests\test_equilibrium_gas.py   # equilibrium mode
python tests\test_wall_functions.py    # wall functions + heat flux
python tests\test_report.py            # end-to-end PDF report
```

## License

MIT — see [LICENSE](LICENSE).
