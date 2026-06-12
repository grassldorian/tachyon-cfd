"""Offscreen GUI construction test (no display needed)."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication
from rocketcfd.gui.main import MainWindow, apply_claude_theme
from rocketcfd.config import SimConfig

app = QApplication([])
apply_claude_theme(app)
win = MainWindow()
win.show()

# round-trip the config form
cfg = win.cfg_panel.get_config()
assert cfg.inlet_p0 == SimConfig().inlet_p0, cfg.inlet_p0
cfg.inlet_p0 = 5e6
cfg.flux_scheme = "roe"
cfg.axisymmetric = True
win.cfg_panel.set_config(cfg)
cfg2 = win.cfg_panel.get_config()
assert cfg2.inlet_p0 == 5e6
assert cfg2.flux_scheme == "roe"
assert cfg2.axisymmetric is True

# all four flux schemes round-trip
for s in ("hllc", "hll", "roe", "ausm"):
    cfg2.flux_scheme = s
    win.cfg_panel.set_config(cfg2)
    assert win.cfg_panel.get_config().flux_scheme == s, s

# image load path (mask preview, rect, scalebar, axis line)
win.load_image_path("examples/nozzle_small.png")
assert win.img_nx == 320 and win.world_rect is not None

# revolve projection sanity
import numpy as np
from rocketcfd.revolve import revolve_project
f = np.random.rand(320, 320).astype(np.float32)
img, r_max = revolve_project(f, 159.5)
assert img.shape == (2 * r_max, 320) and np.isfinite(img[r_max]).all()

print("GUI construction OK")
