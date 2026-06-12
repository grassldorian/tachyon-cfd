"""Offscreen test: designer canvas drawing, mirroring, save, send-to-sim."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage

app = QApplication([])
from rocketcfd.gui.main import MainWindow, apply_claude_theme
apply_claude_theme(app, dark=True)
win = MainWindow()
win.show()

canvas = win.designer.canvas
assert canvas.mirror and canvas.snap_angle

# draw a chamber + converging line programmatically (upper half; mirror
# should duplicate to the lower half)
canvas.pen_width = 10
canvas.push_undo()
canvas._draw_line(QPointF(100, 380), QPointF(400, 380))   # chamber top
canvas._draw_line(QPointF(400, 380), QPointF(500, 460))   # converging
canvas._draw_line(QPointF(500, 460), QPointF(800, 350))   # bell
canvas.pen_color = canvas.pen_color  # walls

# check mirroring: pixel at (250, 380) black AND its mirror (250, 619)
def px(x, y):
    c = canvas.compose().pixelColor(x, y)
    return (c.red(), c.green(), c.blue())
assert px(250, 380) == (0, 0, 0), px(250, 380)
assert px(250, 999 - 380) == (0, 0, 0), px(250, 999 - 380)

# endpoint editing: move the first line's start point, image must follow
canvas.lines[0]["a"] = type(canvas.lines[0]["a"])(100, 300)
canvas._invalidate()
assert px(105, 300) == (0, 0, 0)
canvas.lines[0]["a"] = type(canvas.lines[0]["a"])(100, 380)
canvas._invalidate()

# handle hit test
canvas.resize(1000, 1000)
hit = canvas._hit_handle(type(canvas.lines[0]["a"])(101, 381))
assert hit == (0, "a"), hit

# spline: build, commit, render through control points, handle hit
canvas.mode = "spline"
canvas._spline_pts = [QPointF(300, 200), QPointF(500, 150), QPointF(700, 250)]
canvas._commit_spline()
sp = canvas.lines[-1]
assert sp["type"] == "spline" and len(sp["pts"]) == 3
assert px(500, 150) == (0, 0, 0)              # curve passes through point
assert px(500, 999 - 150) == (0, 0, 0)        # mirrored copy
hit = canvas._hit_handle(QPointF(501, 151))
assert hit == (len(canvas.lines) - 1, ("pt", 1)), hit
# move a spline control point, image follows
sp["pts"][1] = QPointF(500, 120)
canvas._invalidate()
assert px(500, 120) == (0, 0, 0)
# eraser removes the spline
canvas.pen_color = canvas.lines[0]["color"].__class__(255, 255, 255)
n0 = len(canvas.lines)
canvas._erase_lines_near(QPointF(500, 120))
assert len(canvas.lines) == n0 - 1
canvas.mode = "line"

# pan/zoom bookkeeping
canvas.zoom = 4.0
canvas._pan = [50.0, -30.0]
s, ox, oy = canvas._view_geom()
canvas.reset_view()
assert canvas.zoom == 1.0 and canvas._pan == [0.0, 0.0]

# inlet strip; undo removes exactly this stroke
from rocketcfd.gui.designer import COL_INLET
canvas.pen_color = COL_INLET
canvas.push_undo()
canvas._draw_line(QPointF(104, 380), QPointF(104, 500))
assert px(104, 450)[2] > 200
n_lines = len(canvas.lines)
canvas.undo()
assert len(canvas.lines) == n_lines - 1

# freehand strokes are vector "path" objects now
canvas.mode = "free"
canvas.pen_color = COL_INLET
canvas.lines.append(dict(type="path",
                         pts=[QPointF(200, 300), QPointF(220, 310),
                              QPointF(240, 330)],
                         color=COL_INLET, width=8, mirror=False))
canvas._invalidate()
assert px(220, 310)[2] > 200
assert canvas._object_hit(QPointF(221, 311), canvas.lines[-1], 6.0)
canvas.lines.pop(); canvas._invalidate()
canvas.mode = "line"

# double-right-click deletion path: _delete_at removes the hit object
n0 = len(canvas.lines)
assert canvas._delete_at(QPointF(250, 380))
assert len(canvas.lines) == n0 - 1
canvas.undo()
assert len(canvas.lines) == n0

# canvas resize keeps objects
canvas.set_size(800, 600)
assert canvas.cw == 800 and canvas.ch == 600
assert canvas.compose().width() == 800
canvas.set_size(1000, 1000)

# send to sim (saves PNG + loads it in sim tab)
win.designer.name_edit.setText("_test_design")
win.designer.send_to_sim()
assert win.png_path and win.png_path.endswith("_test_design.png")
assert win.tabs.currentIndex() == 0
print("mask loaded:", win.img_nx, "x", win.img_ny)

# 3D viewer volume builder (no GL needed for the math)
from rocketcfd.gui.viewer3d import build_volume, HAS_GL
from rocketcfd.gui.main import get_cmap
field = np.random.rand(320, 320).astype(np.float32)
ct = np.zeros((320, 320), dtype=np.uint8)
ct[100:110, :] = 1
vol, f, lo, hi = build_volume(field, ct, 159.5, get_cmap("turbo"), 96)
print("volume:", vol.shape, "downsample", f, "HAS_GL:", HAS_GL)
assert vol.ndim == 4 and vol.shape[3] == 4

# cleanup the test design file
p = Path(win.png_path)
if p.exists():
    p.unlink()
print("designer OK")
