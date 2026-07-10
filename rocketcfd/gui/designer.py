"""Engine designer tab.

Parametric liquid-engine geometry (chamber / throat / nozzle), sized with the
classic 1-D ideal-rocket model and rendered live as a cross section. Pressing a
button converts the geometry into a Tachyon mask PNG (black wall / white flow)
with a blue pressure inlet at the injector face, then hands it to the solver.
``send_cb(path, meta)``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .. import engine_design as ed

# parametric geometry fields: (key, label, default mm)
GEOM_FIELDS = [
    ("chamber_l", "Chamber length", 120.0),
    ("chamber_d", "Chamber Ø",       80.0),
    ("throat_d",  "Throat Ø",        30.0),
    ("nozzle_l",  "Nozzle length",   90.0),
    ("exit_d",    "Exit Ø",          90.0),
]


class StepLineEdit(QLineEdit):
    """Numeric line edit: Up/Down arrow keys step the value (geometry fields
    step 1 mm). Shift = 10x step, Ctrl = 0.1x step for fine trims."""

    def __init__(self, text="", step=1.0, minimum=0.0, decimals=1):
        super().__init__(text)
        self._step = step
        self._min = minimum
        self._dec = decimals
        self.setAlignment(Qt.AlignRight)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Up, Qt.Key_Down):
            try:
                v = float(self.text().replace(",", "."))
            except ValueError:
                ev.accept()
                return
            s = self._step
            if ev.modifiers() & Qt.ShiftModifier:
                s *= 10.0
            elif ev.modifiers() & Qt.ControlModifier:
                s *= 0.1
            v += s if ev.key() == Qt.Key_Up else -s
            v = max(self._min, v)
            self.setText(f"{v:.{self._dec}f}".rstrip("0").rstrip(".")
                         if self._dec else f"{v:g}")
            ev.accept()
            return
        super().keyPressEvent(ev)


class DesignerTab(QWidget):
    """Parametric engine -> mask PNG -> solver. send_cb(path, meta=dict)."""

    def __init__(self, send_cb):
        super().__init__()
        self.send_cb = send_cb
        self.edits: dict[str, QLineEdit] = {}
        self._rgb = None
        self._info = None
        self._build()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = QHBoxLayout(self)
        left = QWidget()
        left.setMaximumWidth(360)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 6, 6)
        root.addWidget(left)

        ll.addWidget(QLabel("<b>Engine geometry</b>"))

        geo = QGroupBox("Dimensions [mm]")
        gf = QFormLayout(geo)
        for key, label, default in GEOM_FIELDS:
            e = StepLineEdit(f"{default:g}", step=1.0, minimum=1.0)  # ±1 mm
            e.setToolTip("↑ / ↓ arrow keys: ±1 mm  (Shift ±10, Ctrl ±0.1)")
            e.textChanged.connect(self._schedule)
            self.edits[key] = e
            gf.addRow(label, e)
        self.nozzle_combo = QComboBox()
        self.nozzle_combo.addItems(["Conical (15°)", "Bell (Rao 80%)"])
        self.nozzle_combo.currentIndexChanged.connect(self._schedule)
        gf.addRow("Nozzle", self.nozzle_combo)
        ll.addWidget(geo)

        op = QGroupBox("Operating point")
        of = QFormLayout(op)
        self.prop_combo = QComboBox()
        self.prop_combo.addItems(list(ed.PROPELLANTS.keys()))
        self.prop_combo.setCurrentText("LOX / Ethanol")
        self.prop_combo.currentIndexChanged.connect(self._schedule)
        of.addRow("Propellant", self.prop_combo)
        self.pc_edit = StepLineEdit("20", step=1.0, minimum=0.1)     # ±1 bar
        self.pc_edit.textChanged.connect(self._schedule)
        of.addRow("Chamber p [bar]", self.pc_edit)
        self.alt_edit = StepLineEdit("0", step=1.0, minimum=0.0)     # ±1 km
        self.alt_edit.textChanged.connect(self._schedule)
        of.addRow("Altitude [km]", self.alt_edit)
        self.thrust_edit = StepLineEdit("10", step=1.0, minimum=0.1)  # ±1 kN
        of.addRow("Target thrust [kN]", self.thrust_edit)
        b_opt = QPushButton("★  Optimize dimensions")
        b_opt.clicked.connect(self.optimize)
        of.addRow(b_opt)
        ll.addWidget(op)

        msk = QGroupBox("Mesh && pressure inlet")
        mf = QFormLayout(msk)
        self.res_edit = StepLineEdit("600", step=20.0, minimum=120.0, decimals=0)
        self.res_edit.textChanged.connect(self._schedule)
        self.res_edit.setToolTip("Engine length in pixels = mesh resolution. "
                                 "Higher = finer grid (more cells, slower).")
        mf.addRow("Engine length [px]", self.res_edit)
        self.plume_edit = StepLineEdit("1.6", step=0.1, minimum=0.2)
        self.plume_edit.textChanged.connect(self._schedule)
        self.plume_edit.setToolTip("Downstream plume length as a multiple of the "
                                   "engine length (white space + red outlet edge).")
        mf.addRow("Plume length ×", self.plume_edit)
        self.margin_edit = StepLineEdit("0.3", step=0.1, minimum=0.05)
        self.margin_edit.textChanged.connect(self._schedule)
        self.margin_edit.setToolTip(
            "Radial white space above/below the engine, as a multiple of the\n"
            "biggest engine radius. Increase (e.g. 1–3) to give high-altitude\n"
            "plumes room to balloon radially; more cells = slower.")
        mf.addRow("Radial margin ×", self.margin_edit)
        self.inlet_chk = QCheckBox("Pressure inlet at injector face")
        self.inlet_chk.setChecked(True)
        self.inlet_chk.toggled.connect(self._schedule)
        mf.addRow(self.inlet_chk)
        self.half_chk = QCheckBox("Half domain (axis on edge)")
        self.half_chk.setChecked(False)
        self.half_chk.toggled.connect(self._schedule)
        self.half_chk.setToolTip(
            "Simulate only the upper half of the engine with the symmetry\n"
            "axis along the domain edge: results are EXACTLY mirror-\n"
            "symmetric by construction and the run is ~2x faster (half the\n"
            "cells). In full-section mode the two halves are computed\n"
            "independently, so unsteady plumes slowly de-synchronize at\n"
            "rounding level — visible asymmetry on chaotic runs.")
        mf.addRow(self.half_chk)
        self.enclose_chk = QCheckBox("Solid fill around engine")
        self.enclose_chk.setChecked(True)
        self.enclose_chk.toggled.connect(self._schedule)
        self.enclose_chk.setToolTip(
            "Fill everything beside and behind the engine (up to the exit\n"
            "plane) with wall. The ambient pocket around the engine body is\n"
            "a resonating cavity — startup blasts and pressure waves bounce\n"
            "between the engine and the domain edges. Filling it leaves only\n"
            "chamber + plume fluid (also fewer cells = faster). Untick for a\n"
            "free-standing engine with external flow around it.")
        mf.addRow(self.enclose_chk)
        self.inlet_edit = StepLineEdit("75", step=1.0, minimum=5.0, decimals=0)
        self.inlet_edit.textChanged.connect(self._schedule)
        self.inlet_edit.setToolTip("Blue inlet diameter as a percent of the "
                                   "chamber diameter.")
        mf.addRow("Inlet Ø [% chamber]", self.inlet_edit)
        ll.addWidget(msk)

        self.btn_send = QPushButton("Send to solver  →")
        self.btn_send.setProperty("accent", True)
        self.btn_send.clicked.connect(self.send)
        ll.addWidget(self.btn_send)
        ll.addStretch(1)

        # ---- right: preview + performance ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 8, 8, 8)
        root.addWidget(right, 1)

        self.preview = QLabel("…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        rl.addWidget(self.preview, 1)

        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setStyleSheet("color: #888;")
        rl.addWidget(self.caption)

        self.perf = QLabel("")
        self.perf.setStyleSheet("font-family: Consolas, monospace;")
        self.perf.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(self.perf)

    # ------------------------------------------------------------ helpers
    def _schedule(self, *_):
        self._timer.start(80)

    def _read_geom(self):
        return {k: float(self.edits[k].text()) / 1000.0 for k, _, _ in GEOM_FIELDS}

    def _set_geom(self, geom):
        for k, _, _ in GEOM_FIELDS:
            self.edits[k].setText(f"{geom[k] * 1000.0:.1f}")

    def _nozzle(self):
        return self.nozzle_combo.currentText()

    # ------------------------------------------------------------ render
    def _refresh(self):
        try:
            geom = self._read_geom()
            res = max(120, int(float(self.res_edit.text())))
            plume = max(0.2, float(self.plume_edit.text()))
            margin = max(0.05, float(self.margin_edit.text()))
            inlet_frac = max(5.0, min(98.0, float(self.inlet_edit.text()))) / 100.0
            pc = float(self.pc_edit.text()) * 1e5
            pa = ed.ambient_pressure(float(self.alt_edit.text()))
        except ValueError:
            return                                   # mid-edit; ignore quietly
        if min(geom["throat_d"], geom["exit_d"]) <= 0:
            self.caption.setText("throat / exit Ø must be > 0")
            return
        if geom["exit_d"] < geom["throat_d"]:
            self.caption.setText("exit Ø must be ≥ throat Ø")
            return

        prop = ed.PROPELLANTS[self.prop_combo.currentText()]
        nozzle = self._nozzle()
        eps = (geom["exit_d"] / geom["throat_d"]) ** 2
        lam = ed.divergence_efficiency(nozzle, eps)
        perf = ed.solve_engine(geom, prop, pc, pa, lam=lam)

        rgb, info = ed.rasterize_mask(
            geom, nozzle, engine_px=res, plume_factor=plume,
            margin_factor=margin,
            add_inlet=self.inlet_chk.isChecked(), inlet_frac=inlet_frac,
            enclose=self.enclose_chk.isChecked(),
            half=self.half_chk.isChecked())
        self._rgb, self._info = rgb, info

        self._show_preview(rgb)
        self.caption.setText(
            f"mask {info['nx']}×{info['ny']} cells · "
            f"{info['meters_per_pixel']*1000:.3f} mm/px · "
            f"throat ≈ {info['throat_px']:.0f} px")
        note = ("UNDER-expanded" if perf["pe"] > pa * 1.15 else
                "OVER-expanded" if perf["pe"] < pa * 0.85 else
                "~ optimally expanded")
        self.perf.setText(
            f"Thrust   {perf['thrust']/1000:8.2f} kN\n"
            f"Isp      {perf['isp']:8.1f} s\n"
            f"mdot     {perf['mdot']:8.3f} kg/s\n"
            f"c*       {perf['cstar']:8.1f} m/s\n"
            f"epsilon  {perf['eps']:8.2f}\n"
            f"Exit M   {perf['Me']:8.2f}    pe {perf['pe']/1000:.1f} kPa  ({note})\n"
            f"(ideal 1-D estimate — run the solver for the CFD result)")

    def _show_preview(self, rgb):
        h, w, _ = rgb.shape
        buf = np.ascontiguousarray(rgb)
        qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        avail = self.preview.size()
        pm = pm.scaled(max(avail.width(), 200), max(avail.height(), 200),
                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)

    def resizeEvent(self, ev):                       # rescale preview to fit
        super().resizeEvent(ev)
        if self._rgb is not None:
            self._show_preview(self._rgb)

    # ------------------------------------------------------------ actions
    def optimize(self):
        try:
            prop = ed.PROPELLANTS[self.prop_combo.currentText()]
            pc = float(self.pc_edit.text()) * 1e5
            pa = ed.ambient_pressure(float(self.alt_edit.text()))
            target = float(self.thrust_edit.text()) * 1000.0
            if target <= 0:
                raise ValueError("target thrust must be > 0")
            geom = ed.optimize_geometry(prop, pc, pa, target,
                                        nozzle_type=self._nozzle())
        except Exception as exc:                     # noqa: BLE001
            QMessageBox.warning(self, "Tachyon CFD", f"Optimise error:\n{exc}")
            return
        self._set_geom(geom)
        self._refresh()

    def send(self):
        if self._rgb is None:
            QMessageBox.warning(self, "Tachyon CFD",
                                "Adjust the geometry first.")
            return
        from PIL import Image
        # re-render with the exact analytic level set (zero raster ripple);
        # the live preview skips this because it costs ~a second
        try:
            geom = self._read_geom()
            res = max(120, int(float(self.res_edit.text())))
            plume = max(0.2, float(self.plume_edit.text()))
            margin = max(0.05, float(self.margin_edit.text()))
            inlet_frac = max(5.0, min(98.0, float(self.inlet_edit.text()))) / 100.0
            rgb, info = ed.rasterize_mask(
                geom, self._nozzle(), engine_px=res, plume_factor=plume,
                margin_factor=margin,
                add_inlet=self.inlet_chk.isChecked(), inlet_frac=inlet_frac,
                enclose=self.enclose_chk.isChecked(),
                half=self.half_chk.isChecked(), analytic=True)
            self._rgb, self._info = rgb, info
        except ValueError:
            rgb, info = self._rgb, self._info      # keep the last valid mask
        path = str(Path(tempfile.gettempdir()) / "tachyon_designed_engine.png")
        Image.fromarray(rgb).save(path)
        prop = ed.PROPELLANTS[self.prop_combo.currentText()]
        try:
            pc = float(self.pc_edit.text()) * 1e5
        except ValueError:
            pc = 2.0e6
        meta = dict(
            meters_per_pixel=info["meters_per_pixel"],
            axisymmetric=True,
            axis_location=info.get("axis_location", "center"),
            gamma=prop["gamma"], R_gas=ed.R_UNIVERSAL / prop["M"],
            inlet_T0=prop["Tc"], inlet_p0=pc,
        )
        if "node_phi" in info:
            phi_path = str(Path(tempfile.gettempdir())
                           / "tachyon_designed_engine_phi.npy")
            np.save(phi_path, info["node_phi"])
            meta["node_phi_path"] = phi_path
        self.send_cb(path, meta)
