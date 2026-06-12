"""Engine designer: draw nozzle geometries directly inside RocketCFD.

Draw walls (black), pressure inlets (blue) and pressure outlets (red).
Everything is a vector object (lines, splines, freehand paths) so it can be
moved via its handles or deleted with a double right-click. Supports
mirroring across the center axis, a snap grid, 45-degree angle snapping,
middle-mouse panning, wheel zoom, mm/m measurement rulers and a resizable
canvas. Drawings can be saved to the engine library or sent straight to the
simulation tab.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QVBoxLayout, QWidget, QSizePolicy,
)

CANVAS = 1000                  # default canvas size
RULER = 26                     # ruler strip width [screen px]
COL_WALL = QColor(0, 0, 0)
COL_INLET = QColor(0, 80, 255)
COL_OUTLET = QColor(255, 40, 30)
COL_ERASE = QColor(255, 255, 255)


def _seg_dist(p: QPointF, a: QPointF, b: QPointF) -> float:
    """Distance from point p to segment a-b."""
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(p.x() - ax, p.y() - ay)
    t = max(0.0, min(1.0, ((p.x() - ax) * dx + (p.y() - ay) * dy) / L2))
    return math.hypot(p.x() - (ax + t * dx), p.y() - (ay + t * dy))


def _nice_step(target: float) -> float:
    """Round target to a 1/2/5 ladder value."""
    k = 10.0 ** math.floor(math.log10(max(target, 1e-9)))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * k >= target:
            return m * k
    return 10.0 * k


class DesignerCanvas(QWidget):
    """Vector drawing canvas with pan/zoom, rulers and editable objects."""

    changed = Signal()
    HANDLE_PX = 9              # screen-pixel hit radius for handles

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.cw = CANVAS
        self.ch = CANVAS
        # vector objects: {"type": "line", a, b} | {"type": "spline"|"path",
        # "pts": [...]}, each with color, width, mirror
        self.lines: list[dict] = []
        self._composed: QImage | None = None
        self.pen_color = COL_WALL
        self.pen_width = 8
        self.mode = "line"              # "line" | "free" | "spline"
        self.mirror = True
        self.show_grid = True
        self.grid_px = 25
        self.snap_grid = False
        self.snap_angle = True          # 45-degree multiples (line mode)
        self.show_axis = True
        self.show_rulers = True
        self.mm_per_px = 1.0            # measurement scale for the rulers
        self.undo_stack: list[list] = []
        self._drag = None               # ("new",)|("end",idx,key)|("free",idx)
        self._anchor: QPointF | None = None
        self._cursor: QPointF | None = None
        self._spline_pts: list[QPointF] = []
        self._hover: QPointF | None = None
        self.zoom = 1.0
        self._pan = [0.0, 0.0]
        self._pan_last = None
        self._last_rclick = None        # (time, QPointF) for double-rclick
        self.setMouseTracking(True)

    @property
    def line_mode(self) -> bool:        # kept for compatibility
        return self.mode == "line"

    def reset_view(self):
        self.zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

    def set_size(self, w: int, h: int):
        """Change drawing dimensions (objects are kept, view is reset)."""
        self.push_undo()
        self.cw = max(100, int(w))
        self.ch = max(100, int(h))
        self.reset_view()
        self._invalidate()

    # ---------------------------------------------------------- compose
    @staticmethod
    def _spline_path(pts: list[QPointF]) -> QPainterPath:
        """Catmull-Rom spline through the control points."""
        path = QPainterPath()
        if not pts:
            return path
        path.moveTo(pts[0])
        if len(pts) == 1:
            return path
        n = len(pts)
        for i in range(n - 1):
            p0 = pts[max(i - 1, 0)]
            p1, p2 = pts[i], pts[i + 1]
            p3 = pts[min(i + 2, n - 1)]
            c1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6.0,
                         p1.y() + (p2.y() - p0.y()) / 6.0)
            c2 = QPointF(p2.x() - (p3.x() - p1.x()) / 6.0,
                         p2.y() - (p3.y() - p1.y()) / 6.0)
            path.cubicTo(c1, c2, p2)
        return path

    @staticmethod
    def _poly_path(pts: list[QPointF]) -> QPainterPath:
        path = QPainterPath()
        if not pts:
            return path
        path.moveTo(pts[0])
        for q in pts[1:]:
            path.lineTo(q)
        return path

    def _object_paths(self, ln: dict) -> list[QPainterPath]:
        """Render paths of an object incl. its mirrored copy."""
        if ln["type"] == "line":
            pts_sets = [[ln["a"], ln["b"]]]
        else:
            pts_sets = [ln["pts"]]
        if ln["mirror"]:
            pts_sets.append([self._mirrored(q) for q in pts_sets[0]])
        make = self._spline_path if ln["type"] == "spline" else self._poly_path
        return [make(pts) for pts in pts_sets]

    def _draw_object(self, p: QPainter, ln: dict):
        pen = QPen(ln["color"], ln["width"],
                   Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for path in self._object_paths(ln):
            p.drawPath(path)

    def compose(self) -> QImage:
        """White canvas + all vector objects, as exported/simulated."""
        if self._composed is None:
            img = QImage(self.cw, self.ch, QImage.Format_RGB32)
            img.fill(Qt.white)
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            for ln in self.lines:
                self._draw_object(p, ln)
            p.end()
            self._composed = img
        return self._composed

    @property
    def image(self) -> QImage:          # backwards compatibility
        return self.compose()

    def _mirrored(self, p: QPointF) -> QPointF:
        return QPointF(p.x(), self.ch - 1 - p.y())

    def _invalidate(self):
        self._composed = None
        self.update()
        self.changed.emit()

    # ---------------------------------------------------------- mapping
    def _view_geom(self):
        avail_w = max(self.width() - RULER, 50)
        avail_h = max(self.height() - RULER, 50)
        s = min(avail_w / self.cw, avail_h / self.ch) * self.zoom
        ox = RULER + (avail_w - self.cw * s) / 2 + self._pan[0]
        oy = (avail_h - self.ch * s) / 2 + self._pan[1]
        return s, ox, oy

    def wheelEvent(self, ev):
        factor = 1.18 if ev.angleDelta().y() > 0 else 1.0 / 1.18
        new_zoom = min(max(self.zoom * factor, 0.4), 25.0)
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        s, ox, oy = self._view_geom()
        mx, my = ev.position().x(), ev.position().y()
        ix, iy = (mx - ox) / s, (my - oy) / s
        old_zoom = self.zoom
        self.zoom = new_zoom
        self._pan = [0.0, 0.0]
        s2, ox2, oy2 = self._view_geom()
        self._pan[0] = mx - ix * s2 - ox2
        self._pan[1] = my - iy * s2 - oy2
        self.update()

    def _to_img(self, pos) -> QPointF:
        s, ox, oy = self._view_geom()
        x = (pos.x() - ox) / s
        y = (pos.y() - oy) / s
        return QPointF(min(max(x, 0.0), self.cw - 1),
                       min(max(y, 0.0), self.ch - 1))

    def _snap(self, p: QPointF, anchor: QPointF | None) -> QPointF:
        if self.snap_grid:
            g = self.grid_px
            p = QPointF(round(p.x() / g) * g, round(p.y() / g) * g)
        if self.snap_angle and self.line_mode and anchor is not None:
            dx, dy = p.x() - anchor.x(), p.y() - anchor.y()
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                ang = math.atan2(dy, dx)
                snap = round(ang / (math.pi / 4)) * (math.pi / 4)
                length = math.hypot(dx, dy)
                proj = length * math.cos(ang - snap)
                p = QPointF(anchor.x() + proj * math.cos(snap),
                            anchor.y() + proj * math.sin(snap))
        return p

    # ---------------------------------------------------------- handles
    def _handle_points(self, ln: dict):
        if ln["type"] == "line":
            return [("a", ln["a"]), ("b", ln["b"])]
        if ln["type"] == "spline":
            return [(("pt", i), q) for i, q in enumerate(ln["pts"])]
        return []                        # freehand paths: no handles

    def _hit_handle(self, p: QPointF):
        s, _, _ = self._view_geom()
        tol = max(self.HANDLE_PX / max(s, 1e-9), 3.0)
        best, best_d = None, tol
        for i, ln in enumerate(self.lines):
            for key, q in self._handle_points(ln):
                d = math.hypot(p.x() - q.x(), p.y() - q.y())
                if d < best_d:
                    best, best_d = (i, key), d
        return best

    # ---------------------------------------------------------- editing
    def push_undo(self):
        self.undo_stack.append([self._copy_obj(ln) for ln in self.lines])
        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)

    @staticmethod
    def _copy_obj(ln: dict) -> dict:
        c = dict(ln)
        if "pts" in c:
            c["pts"] = [QPointF(q) for q in c["pts"]]
        else:
            c["a"], c["b"] = QPointF(c["a"]), QPointF(c["b"])
        c["color"] = QColor(c["color"])
        return c

    def undo(self):
        if self.undo_stack:
            self.lines = self.undo_stack.pop()
            self._invalidate()

    def clear(self):
        self.push_undo()
        self.lines.clear()
        self._invalidate()

    def _draw_line(self, a: QPointF, b: QPointF):
        self.lines.append(dict(type="line", a=QPointF(a), b=QPointF(b),
                               color=QColor(self.pen_color),
                               width=self.pen_width, mirror=self.mirror))
        self._invalidate()

    def _commit_spline(self):
        if len(self._spline_pts) >= 2:
            self.lines.append(dict(type="spline",
                                   pts=[QPointF(q) for q in self._spline_pts],
                                   color=QColor(self.pen_color),
                                   width=self.pen_width, mirror=self.mirror))
            self._invalidate()
        self._spline_pts = []
        self._last_rclick = None         # commit click must not arm a delete
        self.update()

    def _object_hit(self, p: QPointF, ln: dict, tol: float) -> bool:
        if ln["type"] == "line":
            if _seg_dist(p, ln["a"], ln["b"]) < tol:
                return True
            return ln["mirror"] and _seg_dist(
                p, self._mirrored(ln["a"]), self._mirrored(ln["b"])) < tol
        variants = [ln["pts"]]
        if ln["mirror"]:
            variants.append([self._mirrored(q) for q in ln["pts"]])
        for pts in variants:
            if ln["type"] == "spline":
                path = self._spline_path(pts)
                n = max(int(path.length() / 5.0), 8)
                samples = [path.pointAtPercent(k / n) for k in range(n + 1)]
            else:
                samples = pts
            for a, b in zip(samples[:-1], samples[1:]):
                if _seg_dist(p, a, b) < tol:
                    return True
        return False

    def _erase_lines_near(self, p: QPointF):
        keep = [ln for ln in self.lines
                if not self._object_hit(p, ln,
                                        (ln["width"] + self.pen_width) / 2)]
        if len(keep) != len(self.lines):
            self.lines = keep
            self._invalidate()

    def _delete_at(self, p: QPointF):
        """Delete the topmost object under p (double right-click)."""
        s, _, _ = self._view_geom()
        tol = max(self.HANDLE_PX / max(s, 1e-9), 4.0)
        for i in range(len(self.lines) - 1, -1, -1):
            ln = self.lines[i]
            if self._object_hit(p, ln, max(ln["width"] / 2 + 2, tol)):
                self.push_undo()
                self.lines.pop(i)
                self._invalidate()
                return True
        return False

    # ---------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan_last = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() == Qt.RightButton:
            if self.mode == "spline" and self._spline_pts:
                self._commit_spline()
                return
            p = self._to_img(ev.position())
            now = time.monotonic()
            if self._last_rclick is not None:
                t0, p0 = self._last_rclick
                if now - t0 < 0.45 and math.hypot(
                        p.x() - p0.x(), p.y() - p0.y()) < 14:
                    self._delete_at(p)
                    self._last_rclick = None
                    return
            self._last_rclick = (now, p)
            return
        if ev.button() != Qt.LeftButton:
            return
        p = self._to_img(ev.position())
        if self.mode in ("line", "spline") and not self._spline_pts:
            hit = self._hit_handle(p)
            if hit is not None:
                self.push_undo()
                self._drag = ("end", hit[0], hit[1])
                return
        if self.mode == "spline":
            if not self._spline_pts:
                self.push_undo()
            self._spline_pts.append(self._snap(p, None))
            self.update()
            return
        self.push_undo()
        ps = self._snap(p, None)
        self._anchor = ps
        self._cursor = ps
        if self.mode == "line":
            self._drag = ("new",)
        else:                            # freehand: start a path object
            if self.pen_color == COL_ERASE:
                self._drag = ("erase",)
                self._erase_lines_near(ps)
            else:
                self.lines.append(dict(type="path", pts=[ps],
                                       color=QColor(self.pen_color),
                                       width=self.pen_width,
                                       mirror=self.mirror))
                self._drag = ("free", len(self.lines) - 1)
                self._invalidate()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.mode == "spline" \
                and self._spline_pts:
            self._commit_spline()

    def mouseMoveEvent(self, ev):
        if self._pan_last is not None:
            d = ev.position() - self._pan_last
            self._pan[0] += d.x()
            self._pan[1] += d.y()
            self._pan_last = ev.position()
            self.update()
            return
        p = self._to_img(ev.position())
        self._hover = p
        if self._drag is None:
            if self.mode == "spline" and self._spline_pts:
                self.update()
            return
        kind = self._drag[0]
        if kind == "new":
            self._cursor = self._snap(p, self._anchor)
            self.update()
        elif kind == "end":
            _, idx, key = self._drag
            ln = self.lines[idx]
            if isinstance(key, tuple):
                ln["pts"][key[1]] = self._snap(p, None)
            else:
                other = ln["b"] if key == "a" else ln["a"]
                ln[key] = self._snap(p, other)
            self._invalidate()
        elif kind == "erase":
            self._erase_lines_near(p)
        else:                            # freehand path
            pts = self.lines[self._drag[1]]["pts"]
            if math.hypot(p.x() - pts[-1].x(), p.y() - pts[-1].y()) > 1.5:
                pts.append(QPointF(p))
                self._invalidate()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)
            return
        if ev.button() != Qt.LeftButton or self._drag is None:
            return
        if self._drag[0] == "new" and self._anchor is not None \
                and self._cursor is not None:
            self._draw_line(self._anchor, self._cursor)
        self._drag = None
        self._anchor = None
        self._cursor = None
        self.update()

    # ---------------------------------------------------------- painting
    def _draw_rulers(self, painter: QPainter, s, ox, oy):
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, RULER, h, QColor(46, 45, 43))
        painter.fillRect(0, h - RULER, w, RULER, QColor(46, 45, 43))
        painter.setPen(QPen(QColor(150, 147, 140), 1))
        fm = painter.fontMetrics()

        mmpx = self.mm_per_px
        span_mm = (w - RULER) / s * mmpx
        step_mm = _nice_step(span_mm / max((w - RULER) / 70, 2))
        use_m = step_mm >= 1000.0

        def label(mm):
            return f"{mm / 1000:g} m" if use_m else f"{mm:g} mm"

        # horizontal ruler: x measured from the left edge of the drawing
        x_mm0 = max(0.0, (RULER - ox) / s * mmpx)
        k = math.floor(x_mm0 / step_mm)
        while True:
            mm = k * step_mm
            xs = ox + mm / mmpx * s
            if xs > w:
                break
            if xs >= RULER and 0 <= mm <= self.cw * mmpx:
                painter.drawLine(int(xs), h - RULER, int(xs), h - RULER + 6)
                painter.drawText(int(xs) + 3, h - RULER + 6 + fm.ascent(),
                                 label(mm))
            k += 1
        # vertical ruler: y measured from the center axis (radius)
        cy = (self.ch - 1) / 2
        r_top = (cy - (0 - oy) / s) * mmpx           # radius at screen top
        k = math.floor(-abs(r_top) / step_mm) - 1
        kmax = int(self.ch * mmpx / step_mm) + 1
        for k in range(-kmax, kmax + 1):
            mm = k * step_mm                          # signed radius
            ys = oy + (cy - mm / mmpx) * s
            if ys < 0 or ys > h - RULER:
                continue
            painter.drawLine(RULER - 6, int(ys), RULER, int(ys))
            painter.save()
            painter.translate(RULER - 9, ys - 3)
            painter.rotate(-90)
            painter.drawText(0, 0, label(abs(mm)))
            painter.restore()
        painter.fillRect(0, h - RULER, RULER, RULER, QColor(46, 45, 43))

    def paintEvent(self, ev):
        s, ox, oy = self._view_geom()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(58, 57, 54))
        painter.drawImage(
            int(ox), int(oy), self.compose().scaled(
                int(self.cw * s), int(self.ch * s),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.show_grid and self.grid_px * s >= 7:
            painter.setPen(QPen(QColor(120, 120, 120, 60), 1))
            for g in range(0, self.cw + 1, self.grid_px):
                painter.drawLine(int(ox + g * s), int(oy),
                                 int(ox + g * s), int(oy + self.ch * s))
            for g in range(0, self.ch + 1, self.grid_px):
                painter.drawLine(int(ox), int(oy + g * s),
                                 int(ox + self.cw * s), int(oy + g * s))
        if self.show_axis:
            pen = QPen(QColor(20, 180, 220, 180), 1.5, Qt.DashLine)
            painter.setPen(pen)
            ymid = oy + (self.ch - 1) / 2 * s
            painter.drawLine(int(ox), int(ymid),
                             int(ox + self.cw * s), int(ymid))
        preview_pen = QPen(QColor(self.pen_color.red(), self.pen_color.green(),
                                  self.pen_color.blue(), 150),
                           max(1.0, self.pen_width * s),
                           Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        if self._drag is not None and self._drag[0] == "new" \
                and self._anchor is not None and self._cursor is not None:
            painter.setPen(preview_pen)
            a, c = self._anchor, self._cursor
            painter.drawLine(QPointF(ox + a.x() * s, oy + a.y() * s),
                             QPointF(ox + c.x() * s, oy + c.y() * s))
            if self.mirror:
                painter.drawLine(
                    QPointF(ox + a.x() * s, oy + (self.ch - 1 - a.y()) * s),
                    QPointF(ox + c.x() * s, oy + (self.ch - 1 - c.y()) * s))
        if self._spline_pts:
            painter.setRenderHint(QPainter.Antialiasing)
            pts = list(self._spline_pts)
            if self._hover is not None:
                pts.append(self._hover)
            painter.save()
            painter.translate(ox, oy)
            painter.scale(s, s)
            pen_img = QPen(preview_pen)
            pen_img.setWidthF(self.pen_width)
            painter.setPen(pen_img)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._spline_path(pts))
            if self.mirror:
                painter.drawPath(self._spline_path(
                    [self._mirrored(q) for q in pts]))
            painter.restore()
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.setBrush(QColor(20, 180, 220))
            for q in self._spline_pts:
                painter.drawEllipse(QPointF(ox + q.x() * s, oy + q.y() * s),
                                    4.0, 4.0)
        painter.setRenderHint(QPainter.Antialiasing)
        for ln in self.lines:
            for _, q in self._handle_points(ln):
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.setBrush(QColor(217, 119, 87))
                painter.drawEllipse(QPointF(ox + q.x() * s, oy + q.y() * s),
                                    4.5, 4.5)
        if self.show_rulers:
            self._draw_rulers(painter, s, ox, oy)
        painter.end()


class DesignerTab(QWidget):
    """Designer tab: canvas + tool panel. send_cb(path)."""

    def __init__(self, send_cb):
        super().__init__()
        self.send_cb = send_cb
        self.canvas = DesignerCanvas()

        side = QWidget()
        sl = QVBoxLayout(side)
        side.setMinimumWidth(300)
        side.setMaximumWidth(340)

        tool_box = QGroupBox("Tool")
        tl = QVBoxLayout(tool_box)
        self.tool_group = QButtonGroup(self)
        row = QHBoxLayout()
        for name, col in (("Wall", COL_WALL), ("Inlet", COL_INLET),
                          ("Outlet", COL_OUTLET), ("Erase", COL_ERASE)):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setStyleSheet(
                f"QPushButton:checked {{ background: {col.name()}; "
                f"color: {'#FFFFFF' if name != 'Erase' else '#141413'}; }}")
            self.tool_group.addButton(b)
            b.clicked.connect(lambda _, c=col: setattr(self.canvas, "pen_color", c))
            row.addWidget(b)
            if name == "Wall":
                b.setChecked(True)
        tl.addLayout(row)
        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Line", "Freehand", "Spline"])
        self.mode_combo.currentIndexChanged.connect(self._set_mode)
        form.addRow("Draw mode", self.mode_combo)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(2, 40)
        self.width_spin.setValue(8)
        self.width_spin.valueChanged.connect(
            lambda v: setattr(self.canvas, "pen_width", v))
        form.addRow("Thickness [px]", self.width_spin)
        tl.addLayout(form)
        sl.addWidget(tool_box)

        canv_box = QGroupBox("Canvas")
        cf = QFormLayout(canv_box)
        size_row = QHBoxLayout()
        self.w_edit = QLineEdit(str(CANVAS))
        self.h_edit = QLineEdit(str(CANVAS))
        for e in (self.w_edit, self.h_edit):
            e.setMinimumWidth(68)
            e.setMaximumWidth(86)
            e.setAlignment(Qt.AlignCenter)
            e.returnPressed.connect(self._apply_size)
        for e in (self.w_edit, self.h_edit):
            e.setToolTip("Press Enter to resize the canvas")
        size_row.addWidget(self.w_edit)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.h_edit)
        size_row.addStretch(1)
        cf.addRow("Size [px]", size_row)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 1000.0)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setValue(1.0)
        self.scale_spin.valueChanged.connect(
            lambda v: (setattr(self.canvas, "mm_per_px", v),
                       self.canvas.update()))
        cf.addRow("Ruler scale [mm/px]", self.scale_spin)
        sl.addWidget(canv_box)

        snap_box = QGroupBox("Symmetry && snapping")
        nf = QFormLayout(snap_box)
        self.mirror_chk = QCheckBox("Mirror across axis")
        self.mirror_chk.setChecked(True)
        self.mirror_chk.toggled.connect(
            lambda v: setattr(self.canvas, "mirror", v))
        nf.addRow(self.mirror_chk)
        self.grid_chk = QCheckBox("Show grid")
        self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(
            lambda v: (setattr(self.canvas, "show_grid", v), self.canvas.update()))
        nf.addRow(self.grid_chk)
        self.gridsize_spin = QSpinBox()
        self.gridsize_spin.setRange(5, 200)
        self.gridsize_spin.setValue(25)
        self.gridsize_spin.valueChanged.connect(
            lambda v: (setattr(self.canvas, "grid_px", v), self.canvas.update()))
        nf.addRow("Grid size [px]", self.gridsize_spin)
        self.snapgrid_chk = QCheckBox("Snap to grid")
        self.snapgrid_chk.toggled.connect(
            lambda v: setattr(self.canvas, "snap_grid", v))
        nf.addRow(self.snapgrid_chk)
        self.snapangle_chk = QCheckBox("Snap lines to 45°")
        self.snapangle_chk.setChecked(True)
        self.snapangle_chk.toggled.connect(
            lambda v: setattr(self.canvas, "snap_angle", v))
        nf.addRow(self.snapangle_chk)
        sl.addWidget(snap_box)

        edit_row = QHBoxLayout()
        b_undo = QPushButton("↶ Undo")
        b_undo.clicked.connect(self.canvas.undo)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self.canvas.clear)
        b_fit = QPushButton("Fit view")
        b_fit.clicked.connect(self.canvas.reset_view)
        edit_row.addWidget(b_undo)
        edit_row.addWidget(b_clear)
        edit_row.addWidget(b_fit)
        sl.addLayout(edit_row)
        nav_hint = QLabel("Middle mouse: pan · Wheel: zoom\n"
                          "Spline: click points, right-click finishes\n"
                          "Double right-click: delete a segment")
        nav_hint.setStyleSheet("color: #87837A; font-size: 11px;")
        sl.addWidget(nav_hint)

        save_box = QGroupBox("Save && simulate")
        vf = QFormLayout(save_box)
        self.name_edit = QLineEdit("my_engine")
        vf.addRow("Name", self.name_edit)
        b_save = QPushButton("Save to library")
        b_save.clicked.connect(self.save_to_library)
        vf.addRow(b_save)
        b_send = QPushButton("Send to simulation  ▶︎")
        b_send.setProperty("accent", True)
        b_send.clicked.connect(self.send_to_sim)
        vf.addRow(b_send)
        sl.addWidget(save_box)
        sl.addStretch(1)

        lay = QHBoxLayout(self)
        lay.addWidget(side)
        lay.addWidget(self.canvas, 1)

    # ------------------------------------------------------------------
    def _apply_size(self):
        """Read width/height fields and resize the canvas."""
        try:
            w = int(float(self.w_edit.text().strip()))
            h = int(float(self.h_edit.text().strip()))
        except ValueError:
            QMessageBox.warning(self, "RocketCFD",
                                "Width and height must be numbers.")
            return
        w = min(max(w, 100), 8000)
        h = min(max(h, 100), 8000)
        self.w_edit.setText(str(w))
        self.h_edit.setText(str(h))
        self.canvas.set_size(w, h)

    def _set_mode(self, i: int):
        if self.canvas._spline_pts:
            self.canvas._commit_spline()
        self.canvas.mode = ("line", "free", "spline")[i]

    def _save_png(self, path: Path) -> bool:
        return self.canvas.image.save(str(path), "PNG")

    def save_to_library(self):
        from .library import library_dir, ensure_library
        d = library_dir()
        ensure_library(d)
        name = (self.name_edit.text().strip() or "my_engine") + ".png"
        if self._save_png(d / name):
            QMessageBox.information(self, "RocketCFD", f"Saved {d / name}")
        else:
            QMessageBox.warning(self, "RocketCFD", "Could not save the drawing.")

    def send_to_sim(self):
        from .library import library_dir, ensure_library
        d = library_dir()
        ensure_library(d)
        name = (self.name_edit.text().strip() or "my_engine") + ".png"
        path = d / name
        if not self._save_png(path):
            QMessageBox.warning(self, "RocketCFD", "Could not save the drawing.")
            return
        self.send_cb(str(path))
