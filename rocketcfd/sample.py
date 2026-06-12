"""Generate an example converging-diverging (de Laval) nozzle PNG.

Black = walls, white = flow space, blue = pressure inlet (chamber back wall).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def make_nozzle_png(path: str, width: int = 1000, height: int = 1000,
                    wall_px: int = 8):
    w, h = width, height
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    cy = h // 2
    sc = w / 1000.0                      # geometry defined on a 1000-wide canvas
    x0 = int(100 * sc)                   # chamber back wall
    x_conv = int(350 * sc)               # start of convergence
    x_throat = int(470 * sc)             # throat
    x_exit = int(800 * sc)               # nozzle exit lip
    r_ch = int(110 * sc)                 # chamber half-height
    r_th = int(26 * sc)                  # throat half-height
    r_ex = int(150 * sc)                 # exit half-height

    xs = np.arange(w)
    half = np.full(w, -1.0)              # -1 -> no nozzle at this x

    m = (xs >= x0) & (xs < x_conv)
    half[m] = r_ch
    m = (xs >= x_conv) & (xs < x_throat)
    t = (xs[m] - x_conv) / max(x_throat - x_conv, 1)
    half[m] = r_ch + (r_th - r_ch) * 0.5 * (1 - np.cos(np.pi * t))   # smooth converge
    m = (xs >= x_throat) & (xs <= x_exit)
    t = (xs[m] - x_throat) / max(x_exit - x_throat, 1)
    half[m] = r_th + (r_ex - r_th) * t ** 0.8                        # bell-ish diverge

    # symmetric about the half-integer row h/2 - 0.5 (between the two middle
    # pixel rows) — this is exactly where the axisymmetric solver puts the axis
    yy = np.abs(np.arange(h)[:, None] - (cy - 0.5))                  # (h, w)
    hh = half[None, :]
    nozzle_x = hh >= 0
    wall = nozzle_x & (yy >= hh) & (yy < hh + wall_px)
    img[wall] = (0, 0, 0)

    # chamber back wall (2 layers thick); the inner layer carries the inlet
    # strip so blue touches chamber fluid while staying walled off from ambient
    back = (xs[None, :] >= x0 - 2 * wall_px) & (xs[None, :] < x0) & (yy < r_ch + wall_px)
    img[np.broadcast_to(back, (h, w))] = (0, 0, 0)
    inlet = (xs[None, :] >= x0 - wall_px) & (xs[None, :] < x0) & (yy < int(r_ch * 0.85))
    img[np.broadcast_to(inlet, (h, w))] = (0, 80, 255)

    Image.fromarray(img).save(path)
    return path


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/nozzle.png"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    make_nozzle_png(out, size, size)
    print(f"wrote {out} ({size}x{size})")
