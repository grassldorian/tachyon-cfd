"""RocketCFD — GPU-accelerated 2D compressible RANS solver for rocket nozzles.

Grid is derived directly from a PNG image (1 pixel = 1 finite-volume cell):
  black = wall, white = flow space, blue = pressure inlet.
Domain edges act as pressure outlet / farfield.

Numerics: finite volume, MUSCL reconstruction, HLL/HLLC fluxes,
k-omega SST turbulence model, explicit SSP-RK2 time integration.
All quantities in SI units (Pa, m, kg/m^3, K, m/s).
"""

__version__ = "1.0.0"
