"""Diagnose axisymmetric mass conservation: integrate mdot at x-stations,
restricted to the nozzle interior (excludes ambient air around the engine)."""
import numpy as np

d = np.load("results_axi/fields.npz")
rho = d["Density [kg/m^3]"]
u = d["Velocity u [m/s]"]
ny, nx = rho.shape
axis_row = ny / 2 - 0.5
r = np.arange(ny) - axis_row    # px
dx = 0.001

# walk inward from the axis until the first wall (NaN) -> engine interior only
for x in (60, 100, 140, 150, 160, 200, 250):
    col_rho, col_u = rho[:, x], u[:, x]
    mdot = 0.0
    jc = int(axis_row)          # 159 -> rows 159 (r=-0.5) and 160 (r=+0.5)
    for j0, step in ((jc, -1), (jc + 1, +1)):
        j = j0
        while 0 <= j < ny and not np.isnan(col_rho[j]):
            mdot += col_rho[j] * col_u[j] * 2 * np.pi * abs(r[j]) * dx * dx
            j += step
    print(f"x={x:4d} px: mdot = {mdot/2:8.4f} kg/s (one engine)")
