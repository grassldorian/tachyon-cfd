"""Line-probe / XY-plot dialog: profiles of the solver fields along a line.

Three modes:
  * **line**       — an arbitrary segment the user clicks in the field view;
                     plots the field versus arc length s [m].
  * **centerline** — the symmetry axis (radius 0); plots versus axial x [m].
  * **wall**       — the wall-adjacent value along the nozzle contour; plots
                     versus axial x [m] (the classic wall-pressure distribution).

The dialog is non-modal and pulls the latest snapshot from the MainWindow each
time it replots, so it stays useful while a run continues. CSV export writes
every field sampled along the current line.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout,
)

from .. import probe

PROFILE_FIELDS = [
    "Mach", "Pressure [Pa]", "Temperature [K]", "Density [kg/m^3]",
    "Velocity |V| [m/s]",
]


class ProbeDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Line probe — field profiles")
        self.resize(720, 460)
        self.setModal(False)
        self._p0 = None
        self._p1 = None
        self._mode = "line"

        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.addItems(PROFILE_FIELDS)
        self.field_combo.currentIndexChanged.connect(self.replot)
        top.addWidget(self.field_combo)

        self.mode_group = QButtonGroup(self)
        for key, label in (("line", "Clicked line"), ("centerline", "Centerline"),
                           ("wall", "Wall distribution")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_mode(k))
            self.mode_group.addButton(b)
            top.addWidget(b)
            if key == "line":
                b.setChecked(True)
        top.addStretch(1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.replot)
        top.addWidget(btn_refresh)
        btn_csv = QPushButton("Export CSV…")
        btn_csv.clicked.connect(self.export_csv)
        top.addWidget(btn_csv)
        lay.addLayout(top)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen("#D97757", width=2))
        lay.addWidget(self.plot, 1)

        self.info = QLabel("Click two points on the field view, or pick a preset.")
        self.info.setStyleSheet("color: #87837A;")
        lay.addWidget(self.info)

    # ------------------------------------------------------------------
    def set_line(self, p0, p1):
        self._p0, self._p1 = p0, p1
        self._select_mode_button("line")
        self._mode = "line"
        self.replot()

    def _set_mode(self, key):
        self._mode = key
        # wall distribution defaults to pressure
        if key == "wall" and self.field_combo.currentText() == "Mach":
            self.field_combo.blockSignals(True)
            self.field_combo.setCurrentText("Pressure [Pa]")
            self.field_combo.blockSignals(False)
        self.replot()

    def _select_mode_button(self, key):
        labels = {"line": "Clicked line", "centerline": "Centerline",
                  "wall": "Wall distribution"}
        for b in self.mode_group.buttons():
            b.setChecked(b.text() == labels[key])

    # ------------------------------------------------------------------
    def _profile(self, name):
        """Return (xaxis, values, xlabel) for the current mode, or None."""
        snap = self.main.last_snap
        if snap is None:
            return None
        field = snap["fields"].get(name)
        if field is None:
            return None
        dx = self.main.dx
        y_off = self.main.y_off
        axis_row = self.main._axis_row_interior()
        if self._mode == "centerline":
            out = probe.centerline(field, dx, axis_row, y_off=y_off)
            return out["x"], out["values"], "axial x [m]"
        if self._mode == "wall":
            ct = self.main.mask_ct
            if ct is None:
                return None
            out = probe.wall_pressure(field, ct, dx, axis_row)
            return out["x"], out["values"], "axial x [m]"
        # clicked line
        if self._p0 is None or self._p1 is None:
            return None
        out = probe.sample_line(field, dx, self._p0, self._p1, y_off=y_off)
        return out["s"], out["values"], "distance s [m]"

    def replot(self):
        name = self.field_combo.currentText()
        res = self._profile(name)
        if res is None:
            self.curve.setData([], [])
            self.info.setText("No data — run a simulation, then set a line.")
            return
        xax, vals, xlabel = res
        finite = np.isfinite(vals)
        self.curve.setData(np.asarray(xax)[finite], np.asarray(vals)[finite])
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", name)
        if finite.any():
            self.info.setText(
                f"{name}: min {np.nanmin(vals):.4g}, max {np.nanmax(vals):.4g} "
                f"over {finite.sum()} points")
        else:
            self.info.setText("No finite samples along this line.")

    # ------------------------------------------------------------------
    def export_csv(self):
        snap = self.main.last_snap
        if snap is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export profile CSV", "profile.csv", "CSV (*.csv)")
        if not path:
            return
        dx = self.main.dx
        y_off = self.main.y_off
        axis_row = self.main._axis_row_interior()
        cols = {}
        if self._mode == "wall" and self.main.mask_ct is not None:
            con = probe.wall_contour(self.main.mask_ct, dx, axis_row)
            base = probe.wall_pressure(
                snap["fields"]["Pressure [Pa]"], self.main.mask_ct, dx,
                axis_row, contour=con)
            cols["x_m"] = base["x"]
            cols["r_m"] = base["r"]
            for name in PROFILE_FIELDS:
                wp = probe.wall_pressure(snap["fields"][name], self.main.mask_ct,
                                         dx, axis_row, contour=con)
                cols[name] = wp["values"]
        elif self._mode == "centerline":
            xs = None
            for name in PROFILE_FIELDS:
                out = probe.centerline(snap["fields"][name], dx, axis_row,
                                       y_off=y_off)
                if xs is None:
                    cols["x_m"] = out["x"]
                cols[name] = out["values"]
        else:
            if self._p0 is None:
                return
            first = True
            for name in PROFILE_FIELDS:
                out = probe.sample_line(snap["fields"][name], dx, self._p0,
                                        self._p1, y_off=y_off)
                if first:
                    cols["s_m"] = out["s"]
                    cols["x_m"] = out["x"]
                    cols["y_m"] = out["y"]
                    first = False
                cols[name] = out["values"]
        keys = list(cols.keys())
        n = len(cols[keys[0]])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for i in range(n):
                fh.write(",".join(f"{cols[k][i]:.6g}" for k in keys) + "\n")
        self.info.setText(f"Wrote {n} rows to {path}")
