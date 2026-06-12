"""Altitude-sweep dialog: batch-run the current engine across altitudes and
plot thrust & specific impulse versus altitude (aerospike compensation, etc.).

The sweep runs on a background QThread so the GUI stays responsive; results are
also handed back to the MainWindow so they can be embedded in the PDF report.
"""
from __future__ import annotations

import traceback

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
)

from ..sweep import isa_pressure, altitude_for_pressure


class SweepWorker(QThread):
    progress = Signal(int, int, float)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, png, cfg, pressures, steps, warm):
        super().__init__()
        self.png = png
        self.cfg = cfg
        self.pressures = pressures
        self.steps = steps
        self.warm = warm

    def run(self):
        try:
            from ..sweep import sweep
            res = sweep(self.png, self.cfg, self.pressures, steps=self.steps,
                        warm_start=self.warm,
                        progress=lambda d, t, pa: self.progress.emit(d, t, pa))
            self.finished_ok.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SweepDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.worker = None
        self.setWindowTitle("Altitude sweep — thrust & Isp vs altitude")
        self.resize(760, 560)
        self.setModal(False)

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.alt_min = QDoubleSpinBox(); self.alt_min.setRange(0, 80); self.alt_min.setValue(0)
        self.alt_max = QDoubleSpinBox(); self.alt_max.setRange(0, 80); self.alt_max.setValue(30)
        self.alt_min.setSuffix(" km"); self.alt_max.setSuffix(" km")
        form.addRow("Altitude from", self.alt_min)
        form.addRow("Altitude to", self.alt_max)
        self.n_pts = QSpinBox(); self.n_pts.setRange(2, 40); self.n_pts.setValue(7)
        form.addRow("Points", self.n_pts)
        self.steps = QSpinBox(); self.steps.setRange(200, 200000); self.steps.setValue(4000)
        self.steps.setSingleStep(500)
        form.addRow("Steps / point", self.steps)
        self.warm_chk = QCheckBox("Warm-start (continue from previous point)")
        self.warm_chk.setChecked(True)
        form.addRow(self.warm_chk)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run sweep")
        self.run_btn.setProperty("accent", True)
        self.run_btn.clicked.connect(self.start)
        row.addWidget(self.run_btn)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        row.addWidget(self.bar, 1)
        lay.addLayout(row)

        self.note = QLabel(
            "Tip: an unconverged toy run is noisy — give each point enough steps "
            "to settle. Thrust should rise as altitude (lower back-pressure) climbs.")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #87837A;")
        lay.addWidget(self.note)

        self.p_thrust = pg.PlotWidget(title="Thrust vs altitude")
        self.p_thrust.showGrid(x=True, y=True, alpha=0.3)
        self.p_thrust.setLabel("bottom", "altitude", units="km")
        self.c_thrust = self.p_thrust.plot(pen=pg.mkPen("#D97757", width=2),
                                           symbol="o", symbolBrush="#D97757")
        lay.addWidget(self.p_thrust, 1)
        self.p_isp = pg.PlotWidget(title="Specific impulse vs altitude")
        self.p_isp.showGrid(x=True, y=True, alpha=0.3)
        self.p_isp.setLabel("bottom", "altitude", units="km")
        self.p_isp.setLabel("left", "Isp", units="s")
        self.c_isp = self.p_isp.plot(pen=pg.mkPen("#0E7490", width=2),
                                     symbol="s", symbolBrush="#0E7490")
        lay.addWidget(self.p_isp, 1)

    # ------------------------------------------------------------------
    def start(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.main.png_path:
            self.note.setText("Load an engine in the Simulation tab first.")
            return
        try:
            cfg = self.main.cfg_panel.get_config()
        except ValueError as e:
            self.note.setText(str(e))
            return
        a0 = self.alt_min.value(); a1 = self.alt_max.value()
        if a1 < a0:
            a0, a1 = a1, a0
        alts = np.linspace(a0, a1, self.n_pts.value())
        pressures = [isa_pressure(a * 1000.0) for a in alts]
        self.run_btn.setEnabled(False)
        self.bar.setValue(0)
        self.note.setText("Running… kernels recompile per back-pressure.")
        self.worker = SweepWorker(self.main.png_path, cfg, pressures,
                                  self.steps.value(), self.warm_chk.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_progress(self, done, total, pa):
        self.bar.setValue(int(100 * done / max(total, 1)))
        self.note.setText(
            f"Point {done}/{total}: p_amb = {pa:.0f} Pa "
            f"(alt {altitude_for_pressure(pa)/1000:.1f} km)")

    def _on_done(self, res):
        self.run_btn.setEnabled(True)
        self.bar.setValue(100)
        alt = [r["alt_km"] for r in res]
        self.c_thrust.setData(alt, [r["F"] for r in res])
        unit = res[0]["force_unit"] if res else "N"
        self.p_thrust.setLabel("left", "thrust F", units=unit)
        self.c_isp.setData(alt, [r["Isp"] for r in res])
        self.main._last_sweep = res
        self.note.setText(
            f"Done — {len(res)} points. Stored for the PDF report "
            "(use “Report PDF…”).")

    def _on_fail(self, msg):
        self.run_btn.setEnabled(True)
        self.note.setText("Sweep failed — see dialog.")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Altitude sweep", msg[-3000:])
