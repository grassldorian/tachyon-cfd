"""Offscreen test: parametric engine designer -> mask PNG -> solver hand-off."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from rocketcfd import engine_design as ed
from rocketcfd.mask import load_mask

app = QApplication.instance() or QApplication([])

# 1. rasterizer produces a valid Tachyon mask (wall + flow + inlet + outlet)
geom = dict(chamber_l=0.12, chamber_d=0.08, throat_d=0.03,
            nozzle_l=0.09, exit_d=0.09)
for nozzle in ("Conical (15°)", "Bell (Rao 80%)"):
    rgb, info = ed.rasterize_mask(geom, nozzle, engine_px=400)
    blue = ((rgb[:, :, 2] > 150) & (rgb[:, :, 0] < 80) & (rgb[:, :, 1] < 80)).sum()
    red = ((rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)).sum()
    black = np.all(rgb < 40, axis=2).sum()
    assert blue > 0 and red > 0 and black > 0, (nozzle, blue, red, black)
    assert info["meters_per_pixel"] > 0
    print(f"{nozzle:16s}: {info['nx']}x{info['ny']}  inlet_px={blue}  ok")

# 1b. the inlet is set INTO the injector face: closed toward the outside
#     (wall backing on the left), open only into the chamber. Checked at the
#     classified cell-type level (the AA rendering blends edge pixel colors,
#     so raw-color adjacency would be too strict).
from rocketcfd.mask import classify_pixels, FLUID as CF, INLET as CI
rgb, info = ed.rasterize_mask(geom, "Conical (15°)", engine_px=400)
ct = classify_pixels(rgb)
inl = ct == CI
fl = ct == CF
js, is_ = np.nonzero(inl)
left_open = sum(1 for j, i in zip(js, is_) if i > 0 and fl[j, i - 1])
right_touch = sum(1 for j, i in zip(js, is_)
                  if i + 1 < ct.shape[1] and fl[j, i + 1])
assert left_open == 0, "inlet open to the outside (both sides)!"
assert right_touch > 0, "inlet does not open into the chamber"
print(f"injector face: inlet backed by wall, {right_touch} cells into chamber")

# 1c. analytic level set: the wall surface must lie exactly on the analytic
#     contour (zero rasterization ripple) — straight cone section check
geo_l = dict(chamber_l=0.12, chamber_d=0.08, throat_d=0.03,
             nozzle_l=0.30, exit_d=0.14)
_, info_l = ed.rasterize_mask(geo_l, "Conical (15°)", engine_px=400,
                              plume_factor=0.5, analytic=True)
phi = info_l["node_phi"]
H1, W1 = phi.shape
xs, ys = [], []
for i in range(int(W1 * 0.50), int(W1 * 0.65)):
    c = phi[:H1 // 2, i]
    cr = np.nonzero((c[:-1] < 0) & (c[1:] >= 0))[0]
    if len(cr):
        j = cr[-1]
        ys.append(j + c[j] / (c[j] - c[j + 1]))
        xs.append(i)
A = np.vstack([np.array(xs, float), np.ones(len(xs))]).T
coef, _, _, _ = np.linalg.lstsq(A, np.array(ys), rcond=None)
rms = float(np.sqrt(np.mean((np.array(ys) - A @ coef) ** 2)))
assert rms < 1e-3, f"analytic surface not exact: RMS {rms:.5f} px"
print(f"analytic SDF: cone-wall surface RMS {rms:.6f} px (exact)")

# 2. the mask loads into the solver pipeline with fluid + inlet cells
import tempfile
from PIL import Image
p = str(Path(tempfile.gettempdir()) / "test_designed.png")
rgb, info = ed.rasterize_mask(geom, "Conical (15°)", engine_px=400)
Image.fromarray(rgb).save(p)
m = load_mask(p, info["meters_per_pixel"], axisym_center=True)
assert m.n_fluid > 1000 and m.n_inlet > 0, (m.n_fluid, m.n_inlet)
print(f"load_mask: {m.n_fluid} fluid, {m.n_inlet} inlet cells")

# 3. the designer tab wires into MainWindow and send() applies the meta config
from rocketcfd.gui import main as M
M.apply_claude_theme(app, True)
win = M.MainWindow()
d = win.designer
d.prop_combo.setCurrentText("LOX / Ethanol")
d.pc_edit.setText("20")
d.thrust_edit.setText("10")
d.optimize()
assert d._rgb is not None and d._info is not None
d.send()                                            # -> _design_to_sim(meta)
cp = win.cfg_panel
assert cp.axi_chk.isChecked()
assert cp.axis_combo.currentText() == "image center"
assert abs(float(cp.edits["gamma"].text()) - 1.21) < 1e-6
assert float(cp.edits["inlet_p0"].text()) == 20e5
assert win.tabs.currentIndex() == 0                 # switched to Simulation
assert win.img_nx > 0 and win.img_ny > 0
print(f"send -> grid {win.img_nx}x{win.img_ny}, p0={cp.edits['inlet_p0'].text()}")

# 4. arrow keys step the geometry fields by 1 mm (Shift 10, Ctrl 0.1)
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
e = d.edits["throat_d"]
e.setText("30")
QTest.keyClick(e, Qt.Key_Up)
assert e.text() == "31", e.text()
QTest.keyClick(e, Qt.Key_Down)
QTest.keyClick(e, Qt.Key_Down)
assert e.text() == "29", e.text()
QTest.keyClick(e, Qt.Key_Up, Qt.ShiftModifier)
assert e.text() == "39", e.text()
print("arrow-key stepping OK")

print("designer OK")
