"""Generate validation engine geometries:

1. Rocketdyne F-1 (Saturn V first stage), published dimensions:
   - throat radius 0.4445 m (35 in dia), area ratio 16 -> exit radius 1.778 m
   - 80% bell, length 0.8*(Re-Rt)/tan(15 deg) ~ 3.98 m
   - chamber pressure ~7.0 MPa, LOX/RP-1 preset (CEA: gamma 1.22, R 380, Tc 3676 K)
   Real performance (sea level): F = 6.77 MN, Isp = 263 s, mdot ~ 2578 kg/s.
   Image: 1000x1000 at 7 mm/px (7x7 m domain), axisymmetric center axis.

2. Aerospike: annular chamber around a tapering central spike (1 mm/px).

Red border strips act as pressure outlets (absorb the startup blast).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rocketcfd.config import SimConfig, PROPELLANTS

W = H = 1000
CY = H / 2 - 0.5            # axisymmetric solver axis row


def blank():
    return np.full((H, W, 3), 255, dtype=np.uint8)


def yy_grid():
    return np.abs(np.arange(H)[:, None] - CY)     # (H, W) radial px


def add_outlet_borders(img, t=8):
    img[:t, :] = (255, 40, 30)        # top
    img[-t:, :] = (255, 40, 30)       # bottom
    img[:, -t:] = (255, 40, 30)       # right
    return img


def gen_f1(path: str):
    img = blank()
    yy = yy_grid()
    xs = np.arange(W)

    r_t, r_e, r_c = 63.5, 254.0, 95.0          # px (dx = 7 mm)
    x0, x_conv, x_th, x_exit = 60, 200, 271, 840
    wall = 9

    half = np.full(W, -1.0)
    m = (xs >= x0) & (xs < x_conv)
    half[m] = r_c
    m = (xs >= x_conv) & (xs < x_th)
    t = (xs[m] - x_conv) / (x_th - x_conv)
    half[m] = r_c + (r_t - r_c) * 0.5 * (1 - np.cos(np.pi * t))
    m = (xs >= x_th) & (xs <= x_exit)
    t = (xs[m] - x_th) / (x_exit - x_th)
    half[m] = r_t + (r_e - r_t) * t ** 0.7      # 80%-bell-like parabola

    hh = half[None, :]
    wallm = (hh >= 0) & (yy >= hh) & (yy < hh + wall)
    img[wallm] = (0, 0, 0)

    back = (xs[None, :] >= x0 - 2 * wall) & (xs[None, :] < x0) & (yy < r_c + wall)
    img[np.broadcast_to(back, (H, W))] = (0, 0, 0)
    inlet = (xs[None, :] >= x0 - wall) & (xs[None, :] < x0) & (yy < r_c * 0.85)
    img[np.broadcast_to(inlet, (H, W))] = (0, 80, 255)

    add_outlet_borders(img)
    Image.fromarray(img).save(path)

    cfg = SimConfig()
    cfg.meters_per_pixel = 0.007
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.propellant = "LOX/RP-1 (kerosene)"
    preset = PROPELLANTS[cfg.propellant]      # CEA equilibrium at ~70 bar
    cfg.gamma = preset["gamma"]
    cfg.R_gas = preset["R_gas"]
    cfg.inlet_T0 = preset["inlet_T0"]
    cfg.inlet_p0 = 7.0e6                      # F-1 chamber pressure ~70 bar
    cfg.max_steps = 30000
    cfg.save(Path(path).with_name("f1_config.json"))
    print(f"wrote {path} (+ f1_config.json)")


def gen_aerospike(path: str):
    img = blank()
    yy = yy_grid()
    xs = np.arange(W)
    wall = 8

    # central spike: cylinder then power-law taper to a point
    x_sp0, x_taper, x_tip = 92, 300, 620
    r_spike = np.zeros(W)
    m = (xs >= x_sp0) & (xs < x_taper)
    r_spike[m] = 60.0
    m = (xs >= x_taper) & (xs <= x_tip)
    t = (xs[m] - x_taper) / (x_tip - x_taper)
    r_spike[m] = 60.0 * (1.0 - t) ** 0.9
    spike = (r_spike[None, :] > 0) & (yy < r_spike[None, :])
    img[spike] = (0, 0, 0)

    # outer cowl: straight chamber wall, then converging to the lip
    x_ch0, x_cowl, x_lip = 100, 300, 380
    r_out = np.full(W, -1.0)
    m = (xs >= x_ch0) & (xs < x_cowl)
    r_out[m] = 150.0
    m = (xs >= x_cowl) & (xs <= x_lip)
    t = (xs[m] - x_cowl) / (x_lip - x_cowl)
    r_out[m] = 150.0 + (80.0 - 150.0) * 0.5 * (1 - np.cos(np.pi * t))
    hh = r_out[None, :]
    cowl = (hh >= 0) & (yy >= hh) & (yy < hh + wall)
    img[cowl] = (0, 0, 0)

    # annular back wall + inlet ring between spike base and outer wall
    back = (xs[None, :] >= x_ch0 - 2 * wall) & (xs[None, :] < x_ch0) \
        & (yy < 150 + wall)
    img[np.broadcast_to(back, (H, W))] = (0, 0, 0)
    # keep the spike solid through the back-wall region
    img[spike] = (0, 0, 0)
    inlet = (xs[None, :] >= x_ch0 - wall) & (xs[None, :] < x_ch0) \
        & (yy > 68) & (yy < 142)
    img[np.broadcast_to(inlet, (H, W))] = (0, 80, 255)

    add_outlet_borders(img)
    Image.fromarray(img).save(path)

    cfg = SimConfig()
    cfg.meters_per_pixel = 0.001
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.max_steps = 30000
    cfg.save(Path(path).with_name("aerospike_config.json"))
    print(f"wrote {path} (+ aerospike_config.json)")


if __name__ == "__main__":
    gen_f1("examples/f1_engine.png")
    gen_aerospike("examples/aerospike.png")
