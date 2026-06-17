"""RocketCFD GUI — PySide6 + pyqtgraph.

Run with:  python -m rocketcfd.gui   (or python run_gui.py)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import math

import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import (QAction, QColor, QImage, QPalette, QPixmap,
                           QTransform)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QStatusBar, QTabWidget, QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from ..config import SimConfig, PROPELLANTS, PROPELLANT_MIX
from ..cuda_kernels import axis_j
from ..mask import load_mask, load_image, WALL, INLET, OUTLET, FLUID


def get_cmap(name: str) -> pg.ColorMap:
    try:
        return pg.colormap.get(name)
    except Exception:
        try:
            return pg.colormap.get(name, source="matplotlib")
        except Exception:
            return pg.colormap.get("viridis")


# ====================================================================== worker
class SolverWorker(QThread):
    snapshot_ready = Signal(dict)
    status_msg = Signal(str)
    error = Signal(str)
    initialized = Signal()

    def __init__(self, png_path: str, cfg: SimConfig):
        super().__init__()
        self.png_path = png_path
        self.cfg = cfg
        self.running = False          # paused vs running
        self.stop_requested = False
        self.solver = None

    def run(self):
        try:
            from ..solver import GPUSolver
            mask = load_mask(self.png_path, self.cfg.meters_per_pixel,
                             self.cfg.svg_raster_px,
                             smooth=self.cfg.smooth_boundary,
                             sigma=self.cfg.boundary_sigma,
                             mesh_scale=self.cfg.mesh_scale)
            if mask.n_fluid == 0:
                raise RuntimeError("No fluid (white) cells found in the image.")
            self.status_msg.emit("Compiling CUDA kernels…")
            self.solver = GPUSolver(mask, self.cfg)
            self.solver.step(1)       # warm-up / sanity step
            snap = self.solver.snapshot()
            snap["thrust_history"] = list(self.solver.thrust_history)
            self.snapshot_ready.emit(snap)
            self.initialized.emit()
            self.status_msg.emit(f"Ready — {mask.n_fluid:,} fluid cells, "
                                 f"{mask.n_inlet:,} inlet cells.")
        except Exception:
            self.error.emit(traceback.format_exc())
            return

        while not self.stop_requested:
            if not self.running:
                self.msleep(50)
                continue
            try:
                n = max(1, self.cfg.viz_interval)
                self.solver.step(n)
                snap = self.solver.snapshot()
                snap["thrust_history"] = list(self.solver.thrust_history)
                self.snapshot_ready.emit(snap)
                if self.solver.step_count >= self.cfg.max_steps:
                    self.running = False
                    self.status_msg.emit(f"Reached max steps ({self.cfg.max_steps}).")
                elif self.solver.residual < self.cfg.residual_target:
                    self.running = False
                    self.status_msg.emit(
                        f"Converged: residual {self.solver.residual:.2e} "
                        f"< {self.cfg.residual_target:.0e}")
            except Exception:
                self.running = False
                self.error.emit(traceback.format_exc())

    def export_npz(self, path: str):
        if self.solver is not None:
            self.solver.save_npz(path)


# ================================================================ config form
FLOAT_FIELDS = [
    # (attr, label, group)
    ("meters_per_pixel", "Meters per pixel [m]", "Geometry"),
    ("mesh_scale", "Mesh density ×", "Geometry"),
    ("plume_stretch", "Plume stretch (1=off)", "Geometry"),
    ("gamma", "Heat capacity ratio γ [-]", "Gas"),
    ("R_gas", "Gas constant R [J/(kg·K)]", "Gas"),
    ("mu_ref", "Sutherland μ_ref [Pa·s]", "Gas"),
    ("Pr", "Prandtl number [-]", "Gas"),
    ("inlet_p0", "Total pressure p₀ [Pa]", "Chamber inlet (blue)"),
    ("inlet_T0", "Total temperature T₀ [K]", "Chamber inlet (blue)"),
    ("eta_cstar", "Combustion eff. η_c* [-]", "Chamber inlet (blue)"),
    ("ambient_gamma", "Ambient gas γ [-]", "Gas"),
    ("ambient_R", "Ambient gas R [J/(kg·K)]", "Gas"),
    ("inlet_turb_intensity", "Turbulence intensity [-]", "Chamber inlet (blue)"),
    ("farfield_p", "Static pressure [Pa]", "Farfield / outlet (edges)"),
    ("farfield_T", "Static temperature [K]", "Farfield / outlet (edges)"),
    ("wall_T", "Wall temperature [K], 0=adiab.", "Numerics"),
    ("cfl", "CFL number [-]", "Numerics"),
    ("residual_target", "Residual target [-]", "Run control"),
]
INT_FIELDS = [
    ("svg_raster_px", "SVG raster size [px]", "Geometry"),
    ("inlet_ramp_steps", "Soft-start ramp [steps]", "Chamber inlet (blue)"),
    ("max_steps", "Max steps", "Run control"),
    ("viz_interval", "GUI update every N steps", "Run control"),
]


class ConfigPanel(QWidget):
    """Auto-generated form bound to a SimConfig."""

    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.edits: dict[str, QLineEdit] = {}
        groups: dict[str, QFormLayout] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        def form(group: str) -> QFormLayout:
            if group not in groups:
                box = QGroupBox(group)
                f = QFormLayout(box)
                f.setLabelAlignment(Qt.AlignRight)
                groups[group] = f
                lay.addWidget(box)
            return groups[group]

        for attr, label, group in FLOAT_FIELDS:
            e = QLineEdit()
            e.setAlignment(Qt.AlignRight)
            self.edits[attr] = e
            form(group).addRow(label, e)
        for attr, label, group in INT_FIELDS:
            e = QLineEdit()
            e.setAlignment(Qt.AlignRight)
            self.edits[attr] = e
            form(group).addRow(label, e)

        geo = form("Geometry")
        self.smooth_chk = QCheckBox("Smooth sub-pixel walls (cut cells)")
        self.smooth_chk.setToolTip(
            "Convert the drawing into a cut-cell mesh with a smooth embedded\n"
            "surface instead of pixel-staircase walls.")
        geo.addRow(self.smooth_chk)
        self.axi_chk = QCheckBox("Axisymmetric (rocket engine)")
        geo.addRow(self.axi_chk)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["image center", "top edge", "bottom edge"])
        geo.addRow("Symmetry axis", self.axis_combo)

        gas = form("Gas")
        self.gasmodel_combo = QComboBox()
        self.gasmodel_combo.addItems(
            ["Calorically perfect (constant γ)",
             "Thermally perfect (cp(T), frozen mix)",
             "Equilibrium (shifting, recombination)"])
        self.gasmodel_combo.setToolTip(
            "Calorically perfect: constant γ and cp (fast, classic).\n"
            "Thermally perfect: cp and γ vary with temperature using the\n"
            "frozen chamber composition of the selected propellant.\n"
            "Equilibrium: the composition shifts (recombines) as the gas\n"
            "expands — most realistic Isp and exit pressure; needs a\n"
            "combustion propellant preset (LOX/RP-1, LOX/LH2, etc).")
        gas.insertRow(0, "Gas model", self.gasmodel_combo)
        self.prop_combo = QComboBox()
        self.prop_combo.addItems(["Custom"] + list(PROPELLANTS.keys()))
        self.prop_combo.activated.connect(self._apply_propellant)
        gas.insertRow(0, "Propellant", self.prop_combo)
        self.twogamma_chk = QCheckBox("Two-gamma plume mixing (exhaust + air)")
        self.twogamma_chk.setToolTip(
            "Transport an exhaust mass fraction and blend the gas properties\n"
            "between the exhaust and the ambient air across the plume mixing\n"
            "layer. Adds 'Mixture fraction' and 'Local gamma' fields. The\n"
            "engine core (pure exhaust) is unchanged.")
        gas.addRow(self.twogamma_chk)

        num = form("Numerics")
        self.flux_combo = QComboBox()
        self.flux_combo.addItems(["HLLC", "HLL", "Roe", "AUSM+"])
        self.flux_combo.setToolTip(
            "Roe assumes a perfect gas and is not available with the\n"
            "equilibrium gas model (it falls back to HLLC there).")
        num.addRow("Riemann solver", self.flux_combo)
        self.order_combo = QComboBox()
        self.order_combo.addItems(["2nd order (MUSCL)", "1st order",
                                   "5th order (WENO)"])
        self.order_combo.setToolTip(
            "WENO5 has far lower numerical dissipation than MUSCL — shock\n"
            "diamonds and shear layers survive much further downstream.\n"
            "It runs in the interior fluid and falls back to MUSCL next to\n"
            "walls. Slower per step; pairs well with a fine mesh.")
        num.addRow("Spatial order", self.order_combo)
        self.limiter_combo = QComboBox()
        self.limiter_combo.addItems(["minmod", "van Albada", "van Leer",
                                     "superbee"])
        self.limiter_combo.setToolTip(
            "MUSCL slope limiter, least → most compressive:\n"
            "minmod (robust, diffusive) · van Albada · van Leer\n"
            "(sharper, good default) · superbee (sharpest, can over-steepen).")
        num.addRow("Limiter", self.limiter_combo)
        self.wall_combo = QComboBox(); self.wall_combo.addItems(["no-slip", "slip"])
        num.addRow("Wall condition", self.wall_combo)
        self.viscous_chk = QCheckBox("Viscous (Navier–Stokes)")
        self.turb_chk = QCheckBox("Turbulence model (k-ω SST)")
        self.localdt_chk = QCheckBox("Local time stepping (steady)")
        self.carbuncle_chk = QCheckBox("Carbuncle cure (HLLC shocks)")
        self.carbuncle_chk.setToolTip(
            "Blend HLLC toward HLL only at strong shocks (Ducros-gated) to\n"
            "cure the Mach-disk carbuncle instability. No effect away from\n"
            "strong shocks.")
        self.compcorr_chk = QCheckBox("Compressibility correction (SST)")
        self.compcorr_chk.setToolTip(
            "Wilcox dilatational-dissipation correction for high-Mach shear\n"
            "layers — slows the plume spreading rate to match experiment.")
        num.addRow(self.viscous_chk)
        num.addRow(self.turb_chk)
        num.addRow(self.localdt_chk)
        num.addRow(self.carbuncle_chk)
        num.addRow(self.compcorr_chk)

        note = QLabel("Changes take effect on  ⟲ Initialize.")
        note.setStyleSheet("color: #888; font-style: italic;")
        lay.addWidget(note)
        lay.addStretch(1)
        self.set_config(cfg)

    def _apply_propellant(self):
        name = self.prop_combo.currentText()
        preset = PROPELLANTS.get(name)
        if preset:
            for attr, val in preset.items():
                self.edits[attr].setText(f"{val:g}")

    def set_config(self, cfg: SimConfig):
        for attr, _, _ in FLOAT_FIELDS:
            self.edits[attr].setText(f"{getattr(cfg, attr):g}")
        for attr, _, _ in INT_FIELDS:
            self.edits[attr].setText(str(getattr(cfg, attr)))
        schemes = ["hllc", "hll", "roe", "ausm"]
        s = cfg.flux_scheme.lower().rstrip("+")
        self.flux_combo.setCurrentIndex(schemes.index(s) if s in schemes else 0)
        self.order_combo.setCurrentIndex(
            2 if cfg.muscl_order >= 5 else 0 if cfg.muscl_order >= 2 else 1)
        self.limiter_combo.setCurrentIndex(
            {"minmod": 0, "vanalbada": 1, "vanleer": 2, "superbee": 3}
            .get(cfg.limiter.lower().replace(" ", ""), 0))
        self.wall_combo.setCurrentIndex(0 if cfg.wall_type == "noslip" else 1)
        self.viscous_chk.setChecked(cfg.viscous)
        self.turb_chk.setChecked(cfg.turbulence)
        self.localdt_chk.setChecked(cfg.local_dt)
        self.carbuncle_chk.setChecked(getattr(cfg, "carbuncle_fix", True))
        self.compcorr_chk.setChecked(
            getattr(cfg, "compressibility_correction", False))
        self.axi_chk.setChecked(cfg.axisymmetric)
        self.smooth_chk.setChecked(cfg.smooth_boundary)
        self.axis_combo.setCurrentIndex(
            {"center": 0, "top": 1, "bottom": 2}.get(cfg.axis_location, 0))
        idx = self.prop_combo.findText(cfg.propellant)
        self.prop_combo.setCurrentIndex(idx if idx >= 0 else 0)
        gm = cfg.gas_model.lower()
        self.gasmodel_combo.setCurrentIndex(
            2 if gm.startswith("equilibrium")
            else 1 if gm.startswith("thermally") else 0)
        self.twogamma_chk.setChecked(bool(cfg.two_gamma))

    def get_config(self) -> SimConfig:
        cfg = SimConfig()
        for attr, label, _ in FLOAT_FIELDS:
            try:
                setattr(cfg, attr, float(self.edits[attr].text().replace(",", ".")))
            except ValueError:
                raise ValueError(f"Invalid number for '{label}'")
        for attr, label, _ in INT_FIELDS:
            try:
                setattr(cfg, attr, int(float(self.edits[attr].text())))
            except ValueError:
                raise ValueError(f"Invalid integer for '{label}'")
        cfg.flux_scheme = ["hllc", "hll", "roe", "ausm"][self.flux_combo.currentIndex()]
        cfg.muscl_order = {0: 2, 1: 1, 2: 5}[self.order_combo.currentIndex()]
        cfg.limiter = ["minmod", "vanalbada", "vanleer",
                       "superbee"][self.limiter_combo.currentIndex()]
        cfg.wall_type = "noslip" if self.wall_combo.currentIndex() == 0 else "slip"
        cfg.viscous = self.viscous_chk.isChecked()
        cfg.turbulence = self.turb_chk.isChecked()
        cfg.local_dt = self.localdt_chk.isChecked()
        cfg.carbuncle_fix = self.carbuncle_chk.isChecked()
        cfg.compressibility_correction = self.compcorr_chk.isChecked()
        cfg.axisymmetric = self.axi_chk.isChecked()
        cfg.smooth_boundary = self.smooth_chk.isChecked()
        cfg.axis_location = ["center", "top", "bottom"][self.axis_combo.currentIndex()]
        cfg.propellant = self.prop_combo.currentText()
        cfg.gas_model = ["calorically perfect", "thermally perfect",
                         "equilibrium"][self.gasmodel_combo.currentIndex()]
        cfg.two_gamma = self.twogamma_chk.isChecked()
        return cfg


# ================================================================= main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RocketCFD — GPU nozzle flow solver")
        # restore-size when un-maximized: ~90% of the available screen
        scr = QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.resize(int(g.width() * 0.9), int(g.height() * 0.9))
        else:
            self.resize(1500, 950)

        self.worker: SolverWorker | None = None
        self.png_path: str | None = None
        self.last_snap: dict | None = None
        self.overlay_rgba: np.ndarray | None = None
        # plume-stretch display remap: nearest computational column per
        # physical-uniform display column (None when the grid is uniform)
        self._disp_idx: np.ndarray | None = None
        self._disp_w = 0
        self.dx = 0.001

        # ---------- left: controls ----------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 6, 6)

        self.title_lbl = QLabel("RocketCFD")
        self.subtitle_lbl = QLabel("GPU rocket nozzle flow solver")
        # custom logo: drop a PNG at <project>/assets/logo.png
        self._has_logo = False
        self._logo_dark = self._logo_light = None
        from .library import project_root
        logo = project_root() / "assets" / "logo.png"
        if logo.exists():
            pm = QPixmap(str(logo))
            if not pm.isNull():
                img = pm.toImage()
                self._logo_dark = QPixmap.fromImage(img).scaledToHeight(
                    84, Qt.SmoothTransformation)
                inv = img.copy()              # light mode: inverted colors
                inv.invertPixels(QImage.InvertRgb)
                self._logo_light = QPixmap.fromImage(inv).scaledToHeight(
                    84, Qt.SmoothTransformation)
                self.title_lbl.setPixmap(
                    self._logo_dark if ACTIVE_DARK[0] else self._logo_light)
                self._has_logo = True
        ll.addWidget(self.title_lbl)
        ll.addWidget(self.subtitle_lbl)
        ll.addSpacing(8)

        self.btn_load = QPushButton("Load engine…")
        self.btn_load.clicked.connect(self.load_png)
        self.lbl_png = QLabel("<i>no image loaded</i>")
        self.lbl_png.setWordWrap(True)
        ll.addWidget(self.btn_load)
        ll.addWidget(self.lbl_png)

        row = QHBoxLayout()
        self.btn_init = QPushButton("⟲  Initialize")
        self.btn_init.clicked.connect(self.initialize)
        self.btn_run = QPushButton("▶︎  Run")
        self.btn_run.setProperty("accent", True)
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_run.setEnabled(False)
        row.addWidget(self.btn_init)
        row.addWidget(self.btn_run)
        ll.addLayout(row)

        perf_box = QGroupBox("Engine performance")
        pf = QFormLayout(perf_box)
        self.lbl_thrust = QLabel("–")
        self.lbl_mdot = QLabel("–")
        self.lbl_isp = QLabel("–")
        self.lbl_ceff = QLabel("–")
        for lbl in (self.lbl_thrust, self.lbl_mdot, self.lbl_isp, self.lbl_ceff):
            # Ignored horizontal policy: changing text never resizes the layout
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            lbl.setMinimumWidth(110)
        self.lbl_split = QLabel("–")
        self.lbl_split.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_split.setMinimumWidth(110)
        pf.addRow("Thrust F", self.lbl_thrust)
        pf.addRow("Mass flow ṁ", self.lbl_mdot)
        pf.addRow("Fuel / oxidizer", self.lbl_split)
        pf.addRow("Specific impulse", self.lbl_isp)
        pf.addRow("Eff. exhaust vel.", self.lbl_ceff)
        self.lbl_conv = QLabel("–")
        self.lbl_conv.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_conv.setMinimumWidth(110)
        self.lbl_conv.setToolTip(
            "Steady-state check: peak-to-peak thrust variation over the last\n"
            "10% of the run. Below 0.5% the quoted numbers are converged;\n"
            "above that they are still settling.")
        pf.addRow("Convergence", self.lbl_conv)
        ll.addWidget(perf_box)

        self.cfg_panel = ConfigPanel(SimConfig())
        self.cfg_panel.axi_chk.toggled.connect(lambda _: self._update_geometry())
        self.cfg_panel.axis_combo.currentIndexChanged.connect(
            lambda _: self._update_geometry())
        self.cfg_panel.edits["meters_per_pixel"].editingFinished.connect(
            self._apply_scale)
        self.cfg_panel.edits["mesh_scale"].editingFinished.connect(
            self._reload_geometry)
        self.cfg_panel.edits["mesh_scale"].setToolTip(
            "Mesh density multiplier. >1 makes a finer grid (more cells,\n"
            "slower, sharper); <1 a coarser grid. The engine's physical size\n"
            "is unchanged. Takes effect immediately on the preview and run.")
        self.cfg_panel.edits["plume_stretch"].setToolTip(
            "Downstream x-mesh stretch ratio per column past the nozzle exit\n"
            "(1.0 = uniform/off; try 1.02–1.05). Extends the plume domain and\n"
            "keeps the near-exit grid fine so shock diamonds survive further.\n"
            "Walls stay on the uniform grid, so thrust/Isp are unchanged.\n"
            "The field view is shown across the true physical length, so the\n"
            "plume appears longer; the far field is coarser (bigger cells).")
        self.cfg_panel.edits["eta_cstar"].setToolTip(
            "Combustion (c*) efficiency. Incomplete combustion releases less\n"
            "energy, so the effective chamber temperature is η² · T₀.\n"
            "1.0 = ideal combustion (theoretical ceiling); real engines run\n"
            "0.90–0.98 (the F-1 was ≈0.93). Lowers Isp by ~η and raises\n"
            "mass flow by ~1/η; thrust is nearly unchanged.")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.cfg_panel)
        ll.addWidget(scroll, 1)

        # (config/export/tool buttons live in the bottom bar, built below)

        # ---------- center: field view ----------
        view_widget = QWidget()
        vl = QVBoxLayout(view_widget)
        vl.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.addItems([
            "Mach", "Pressure [Pa]", "Temperature [K]", "Density [kg/m^3]",
            "Velocity |V| [m/s]", "Velocity u [m/s]", "Velocity v [m/s]",
            "Turb. kinetic energy k [m^2/s^2]", "Specific dissipation omega [1/s]",
            "Eddy viscosity ratio mu_t/mu [-]", "Schlieren |grad rho|",
            "Wall heat flux [W/m^2]",
            "Mixture fraction [-]", "Local gamma [-]",
        ])
        self.field_combo.currentTextChanged.connect(self.refresh_view)
        bar.addWidget(self.field_combo)
        bar.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno",
                                  "magma", "cividis", "RdYlBu", "Spectral",
                                  "jet", "hot", "coolwarm", "twilight",
                                  "nipy_spectral"])
        self.cmap_combo.currentTextChanged.connect(self.refresh_view)
        bar.addWidget(self.cmap_combo)
        self.auto_chk = QCheckBox("Auto range")
        self.auto_chk.setChecked(True)
        self.auto_chk.toggled.connect(self.refresh_view)
        bar.addWidget(self.auto_chk)
        self.mesh_chk = QCheckBox("Mesh")
        self.mesh_chk.setToolTip(
            "Show the computational mesh: the smooth embedded wall surface\n"
            "(orange) and, when zoomed in, the cell edges.")
        self.mesh_chk.toggled.connect(self.on_mesh_toggled)
        bar.addWidget(self.mesh_chk)
        self.btn_probe = QPushButton("Probe")
        self.btn_probe.setCheckable(True)
        self.btn_probe.setToolTip(
            "Line probe: click two points on the field to plot pressure,\n"
            "Mach and temperature along that line. Presets in the dialog:\n"
            "centerline profile and wall-pressure distribution.")
        self.btn_probe.toggled.connect(self.toggle_probe)
        bar.addWidget(self.btn_probe)
        self.btn_theme = QPushButton("◐")
        self.btn_theme.setFixedWidth(40)
        self.btn_theme.setToolTip("Invert GUI colors (light / dark)")
        self.btn_theme.clicked.connect(self.toggle_theme)
        bar.addWidget(self.btn_theme)
        self.lvl_min = QLineEdit("0"); self.lvl_min.setMaximumWidth(90)
        self.lvl_max = QLineEdit("1"); self.lvl_max.setMaximumWidth(90)
        self.lvl_min.editingFinished.connect(self.refresh_view)
        self.lvl_max.editingFinished.connect(self.refresh_view)
        bar.addWidget(QLabel("min")); bar.addWidget(self.lvl_min)
        bar.addWidget(QLabel("max")); bar.addWidget(self.lvl_max)
        bar.addStretch(1)
        vl.addLayout(bar)

        # ---- replay bar (appears once frames have been recorded) ----
        self.replay_bar = QWidget()
        rb = QHBoxLayout(self.replay_bar)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.addWidget(QLabel("Replay:"))
        self.replay_play = QPushButton("▶︎")
        self.replay_play.setCheckable(True)
        self.replay_play.setFixedWidth(36)
        self.replay_play.toggled.connect(self._replay_play_toggled)
        rb.addWidget(self.replay_play)
        from PySide6.QtWidgets import QSlider
        self.replay_slider = QSlider(Qt.Horizontal)
        self.replay_slider.setRange(0, 0)
        self.replay_slider.valueChanged.connect(self._show_replay_frame)
        rb.addWidget(self.replay_slider, 1)
        self.replay_speed = QComboBox()
        self.replay_speed.addItems(["2 fps", "5 fps", "10 fps", "20 fps"])
        self.replay_speed.setCurrentIndex(2)
        self.replay_speed.currentIndexChanged.connect(self._replay_speed_changed)
        rb.addWidget(self.replay_speed)
        self.replay_lbl = QLabel("0/0")
        self.replay_lbl.setMinimumWidth(70)
        rb.addWidget(self.replay_lbl)
        self.replay_live = QPushButton("⏹︎ Live")
        self.replay_live.clicked.connect(self._replay_exit)
        rb.addWidget(self.replay_live)
        self.replay_bar.setVisible(False)
        vl.addWidget(self.replay_bar)

        from PySide6.QtCore import QTimer
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick)
        self.replay_frames: list[tuple] = []
        self._replay_stride = 1
        self._replay_count = 0
        self.y_off = 0.0

        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.vb = self.plot.getViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")
        self.plot.showGrid(x=False, y=False)
        # fixed axis sizes so the view does not resize as tick labels change
        self.plot.getAxis("left").setWidth(74)
        self.plot.getAxis("bottom").setHeight(34)
        self.img_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item.setZValue(10)
        self.plot.addItem(self.img_item)
        self.plot.addItem(self.overlay_item)
        self.axis_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("#0E7490", width=1.5, style=Qt.DashLine),
            label="axis", labelOpts={"color": "#0E7490", "position": 0.05})
        self.axis_line.setZValue(20)
        self.plot.addItem(self.axis_line)
        self.axis_line.hide()
        # mesh view: embedded smooth surface + cell-edge grid (when zoomed in)
        self.mesh_iso = pg.IsocurveItem(level=0.5,
                                        pen=pg.mkPen("#D97757", width=2))
        self.mesh_iso.setParentItem(self.overlay_item)
        self.mesh_iso.setTransform(QTransform().translate(0.5, 0.5))
        self.mesh_iso.setZValue(30)
        self.mesh_iso.hide()
        self.mesh_grid = pg.PlotCurveItem(
            pen=pg.mkPen((128, 128, 128, 110), width=1), connect="pairs")
        self.mesh_grid.setZValue(25)
        self.plot.addItem(self.mesh_grid)
        self.mesh_grid.hide()
        self.vb.sigRangeChanged.connect(self._update_mesh_grid)
        self.mask_lam = None
        self.scalebar = None
        self.img_nx = self.img_ny = 0
        self.world_rect: QRectF | None = None
        self.cbar = pg.ColorBarItem(colorMap=get_cmap("turbo"), width=18)
        self.cbar.setImageItem(self.img_item)
        try:                                  # fixed width: no jitter when
            self.cbar.getAxis("right").setWidth(80)   # tick labels change
        except Exception:
            pass
        self.glw.addItem(self.cbar)
        self.glw.scene().sigMouseMoved.connect(self.on_mouse_move)
        # ---- line-probe overlay + click capture ----
        self.glw.scene().sigMouseClicked.connect(self._on_scene_click)
        self.probe_dialog = None
        self._probe_mode = False
        self._probe_p0: tuple | None = None
        self.sweep_dialog = None
        self._last_sweep = None
        self.probe_line = pg.PlotCurveItem(
            pen=pg.mkPen("#0E7490", width=1.5, style=Qt.DashLine))
        self.probe_line.setZValue(40)
        self.plot.addItem(self.probe_line)
        self.probe_pts = pg.ScatterPlotItem(
            size=9, brush=pg.mkBrush("#0E7490"), pen=pg.mkPen("w", width=1))
        self.probe_pts.setZValue(41)
        self.plot.addItem(self.probe_pts)
        vl.addWidget(self.glw, 1)

        # thrust history plot
        self.res_plot = pg.PlotWidget(title="Thrust")
        self.res_plot.showGrid(x=True, y=True, alpha=0.3)
        self.res_plot.setMaximumHeight(190)
        self.res_plot.getPlotItem().getAxis("left").setWidth(74)
        self.res_plot.setLabel("bottom", "step")
        self.res_curve = self.res_plot.plot(pen=pg.mkPen("#D97757", width=2))

        split = QSplitter(Qt.Vertical)
        split.addWidget(view_widget)
        split.addWidget(self.res_plot)
        split.setStretchFactor(0, 1)
        split.setSizes([1400, 150])      # thrust strip starts snapped down

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(split)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([475, 1445])
        left.setMinimumWidth(420)

        from .designer import DesignerTab
        from .viewer3d import Viewer3DTab
        self.tabs = QTabWidget()
        self.tabs.addTab(main_split, "Simulation")
        self.designer = DesignerTab(self._design_to_sim)
        self.tabs.addTab(self.designer, "Engine designer")
        self.viewer3d = Viewer3DTab(self)
        self.tabs.addTab(self.viewer3d, "3D exhaust")

        # ---- bottom bar: config / export / tool buttons, horizontal ----
        bottom = QWidget()
        bb = QHBoxLayout(bottom)
        bb.setContentsMargins(10, 4, 10, 6)
        b_save = QPushButton("Save config…"); b_save.clicked.connect(self.save_config)
        b_loadc = QPushButton("Load config…"); b_loadc.clicked.connect(self.load_config)
        b_npz = QPushButton("Export NPZ…"); b_npz.clicked.connect(self.export_npz)
        b_png = QPushButton("Export view PNG…"); b_png.clicked.connect(self.export_view)
        b_sweep = QPushButton("Altitude sweep…")
        b_sweep.setToolTip(
            "Batch-run this engine across a range of altitudes (ambient\n"
            "back-pressures) and plot thrust and Isp vs altitude — useful\n"
            "for measuring aerospike altitude compensation.")
        b_sweep.clicked.connect(self.open_sweep)
        b_report = QPushButton("Report PDF…")
        b_report.setToolTip(
            "Generate a PDF report: geometry, mesh, fields, thrust curve,\n"
            "wall pressure + Bartz heat flux, performance table, and the\n"
            "last altitude sweep (if one was run).")
        b_report.clicked.connect(self.export_report)
        b_mp4 = QPushButton("Export video MP4…")
        b_mp4.setToolTip(
            "Encode the recorded replay frames (the field shown while the\n"
            "solver ran) into an MP4 video.")
        b_mp4.clicked.connect(self.export_mp4)
        for b in (b_save, b_loadc, b_npz, b_png, b_sweep, b_report, b_mp4):
            bb.addWidget(b)
        bb.addStretch(1)

        central = QWidget()
        cv = QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.tabs, 1)
        cv.addWidget(bottom)
        self.setCentralWidget(central)
        self.mask_ct = None

        # status bar (fixed minimum widths so text changes do not shift layout)
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_step = QLabel("step –")
        self.lbl_step.setMinimumWidth(95)
        self.lbl_res = QLabel("res –")
        self.lbl_res.setMinimumWidth(105)
        self.lbl_perf = QLabel("")
        self.lbl_perf.setMinimumWidth(95)
        self.lbl_probe = QLabel("")
        for w in (self.lbl_step, self.lbl_res, self.lbl_perf):
            sb.addPermanentWidget(w)
        sb.addWidget(self.lbl_probe)

        self.dark_mode = ACTIVE_DARK[0]
        self.btn_theme.setText("◐")
        self._scalebar_width = 0.0
        self._apply_widget_theme()

    # ------------------------------------------------------------- actions
    def load_png(self):
        from .library import EngineLibraryDialog
        dlg = EngineLibraryDialog(self)
        if dlg.exec() and dlg.selected:
            self.load_image_path(dlg.selected)

    def _design_to_sim(self, path: str):
        """Designer tab callback: load a drawn engine into the simulation."""
        self.load_image_path(path)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            "Engine loaded from the designer — press Initialize.", 10000)

    def _axis_offset_m(self) -> float:
        """World-y of the symmetry axis: the y ruler measures radius from it."""
        if not self.img_ny:
            return 0.0
        try:
            cfg = self.cfg_panel.get_config()
        except ValueError:
            cfg = SimConfig()
        return (axis_j(cfg, self.img_ny) - 2.0 + 0.5) * self.dx

    def _axis_row_interior(self) -> float:
        """Axis position in interior row coordinates (half-integer)."""
        if not self.img_ny:
            return 0.0
        try:
            cfg = self.cfg_panel.get_config()
        except ValueError:
            cfg = SimConfig()
        return axis_j(cfg, self.img_ny) - 2.0

    def _update_geometry(self):
        """Re-map view geometry (rect, rulers, axis line) from the config."""
        if self.img_nx == 0:
            return
        self.y_off = self._axis_offset_m()
        # displayed x-extent: stretched plume is shown across its TRUE physical
        # length (the field is remapped column-wise), not the drawn frame width
        wx = (self._disp_w if self._disp_idx is not None else self.img_nx) * self.dx
        self.world_rect = QRectF(0.0, -self.y_off, wx, self.img_ny * self.dx)
        self.img_item.setRect(self.world_rect)
        self.overlay_item.setRect(self.world_rect)
        self._update_scalebar(wx)
        self._update_axis_line()
        self._update_mesh_grid()

    def _setup_stretch_display(self, snap):
        """Build the nearest-column remap that shows a stretched plume across
        its true physical length. Cheap; computed once per geometry."""
        xc = snap.get("x_centers")
        stretched = snap.get("meta", {}).get("stretched", False)
        if not stretched or xc is None or self.dx <= 0:
            if self._disp_idx is not None:
                self._disp_idx = None
                self._update_geometry()
            return
        xc = np.asarray(xc, dtype=np.float64)
        n_disp = max(int(np.ceil(xc[-1] / self.dx)), len(xc))
        if self._disp_idx is not None and self._disp_w == n_disp:
            return                                   # already built
        xu = (np.arange(n_disp) + 0.5) * self.dx     # display cell centers
        idx = np.clip(np.searchsorted(xc, xu), 1, len(xc) - 1)
        left = xu - xc[idx - 1]
        right = xc[idx] - xu
        idx = np.where(left < right, idx - 1, idx)   # nearest computational col
        self._disp_idx = idx.astype(np.intp)
        self._disp_w = n_disp
        self.set_overlay(self.mask_ct) if self.mask_ct is not None else None
        self._update_geometry()

    def _apply_scale(self):
        """Re-map the view coordinate system when meters-per-pixel changes."""
        try:
            cfg = self.cfg_panel.get_config()
        except ValueError:
            return
        eff = cfg.meters_per_pixel / max(cfg.mesh_scale, 1e-9)
        if self.img_nx == 0 or abs(eff - self.dx) < 1e-15:
            return
        self.dx = eff
        self._update_geometry()
        self.vb.autoRange()

    def _reload_geometry(self):
        """Re-rasterize the loaded engine when the mesh density changes."""
        if self.png_path:
            self.load_image_path(self.png_path)

    def load_image_path(self, path: str):
        self.png_path = path
        self.lbl_png.setText(Path(path).name)
        try:
            cfg = self.cfg_panel.get_config()
        except ValueError:
            cfg = SimConfig()
        try:
            mask = load_mask(path, cfg.meters_per_pixel, cfg.svg_raster_px,
                             smooth=cfg.smooth_boundary,
                             sigma=cfg.boundary_sigma,
                             mesh_scale=cfg.mesh_scale)
        except Exception as e:
            QMessageBox.critical(self, "RocketCFD", f"Could not load image:\n{e}")
            return
        self.dx = mask.dx
        self.img_nx, self.img_ny = mask.nx, mask.ny
        self._disp_idx = None                 # rebuilt from the next snapshot
        self.mask_lam = mask.lam[2:-2, 2:-2].copy() if mask.smooth else None
        self.mask_ct = mask.cell_type[2:-2, 2:-2].copy()
        # show the raw drawing until the solver is initialized
        gray = mask.rgb.mean(axis=2).astype(np.float32)
        self.img_item.setImage(gray, autoLevels=True)
        self.set_overlay(self.mask_ct)
        self._update_geometry()
        if self.mesh_chk.isChecked():
            self._refresh_mesh_items()
            self.mesh_iso.setVisible(self.mask_lam is not None)
        self.replay_frames.clear()
        self._replay_stride = 1
        self._replay_count = 0
        self.replay_bar.setVisible(False)
        self.vb.autoRange()
        self.statusBar().showMessage(
            f"Loaded {mask.nx}x{mask.ny} ({mask.nx*mask.dx:.3g} x {mask.ny*mask.dx:.3g} m): "
            f"{mask.n_fluid:,} fluid, {mask.n_inlet:,} inlet cells. Press Initialize.")

    def _update_scalebar(self, width_m: float):
        if width_m <= 0:
            return
        self._scalebar_width = width_m
        target = width_m * 0.2
        k = 10.0 ** math.floor(math.log10(target))
        nice = k
        for mult in (5.0, 2.0, 1.0):
            if mult * k <= target:
                nice = mult * k
                break
        if self.scalebar is not None:
            scene = self.scalebar.scene()
            if scene is not None:
                scene.removeItem(self.scalebar)
            self.scalebar = None
        self.scalebar = pg.ScaleBar(size=nice, suffix="m",
                                    brush=pg.mkBrush(*ACTIVE_COLORS["scalebar"]))
        self.scalebar.setParentItem(self.vb)
        self.scalebar.anchor((1, 1), (1, 1), offset=(-40, -40))

    def _update_axis_line(self):
        try:
            cfg = self.cfg_panel.get_config()
        except ValueError:
            return
        if cfg.axisymmetric and self.img_ny:
            row_c = axis_j(cfg, self.img_ny) - 2.0
            self.axis_line.setPos((row_c + 0.5) * self.dx - self.y_off)
            self.axis_line.show()
        else:
            self.axis_line.hide()

    def set_overlay(self, ctype: np.ndarray):
        h, w = ctype.shape
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        rgba[ctype == WALL] = (255, 255, 255, 255)
        rgba[ctype == INLET] = (30, 110, 255, 200)
        rgba[ctype == OUTLET] = (230, 60, 50, 200)
        self.overlay_rgba = rgba
        if self._disp_idx is not None:          # remap to physical x extent
            rgba = rgba[:, self._disp_idx]
        self.overlay_item.setImage(rgba)
        if self.world_rect is not None:
            self.overlay_item.setRect(self.world_rect)

    def initialize(self):
        if not self.png_path:
            QMessageBox.information(self, "RocketCFD", "Load a nozzle PNG first.")
            return
        try:
            cfg = self.cfg_panel.get_config()
        except ValueError as e:
            QMessageBox.warning(self, "RocketCFD", str(e))
            return
        if cfg.gas_model.lower().startswith("equilibrium"):
            from ..equilibrium import REACTANTS
            if cfg.propellant not in REACTANTS:
                QMessageBox.information(
                    self, "RocketCFD",
                    "Equilibrium mode needs a propellant selection\n"
                    "(e.g. LOX/RP-1, LOX/LH2, LOX/Ethanol or UDMH/N2O4).")
                return
        self._apply_scale()
        self.replay_frames.clear()
        self._replay_stride = 1
        self._replay_count = 0
        self.replay_bar.setVisible(False)
        self.shutdown_worker()
        self.worker = SolverWorker(self.png_path, cfg)
        self.worker.snapshot_ready.connect(self.on_snapshot)
        self.worker.status_msg.connect(lambda m: self.statusBar().showMessage(m, 15000))
        self.worker.error.connect(self.on_error)
        self.worker.initialized.connect(lambda: self.btn_run.setEnabled(True))
        self.btn_run.setChecked(False)
        self.btn_run.setText("▶︎  Run")
        self.btn_run.setEnabled(False)
        self.statusBar().showMessage("Initializing solver (CUDA compile)…")
        self.worker.start()

    def toggle_run(self, checked: bool):
        if self.worker is None:
            self.btn_run.setChecked(False)
            return
        self.worker.running = checked
        self.btn_run.setText("⏸︎  Pause" if checked else "▶︎  Run")

    def shutdown_worker(self):
        if self.worker is not None:
            self.worker.running = False
            self.worker.stop_requested = True
            self.worker.wait(5000)
            self.worker = None

    def closeEvent(self, ev):
        self.shutdown_worker()
        super().closeEvent(ev)

    # ------------------------------------------------------------- slots
    @Slot(dict)
    def on_snapshot(self, snap: dict):
        self.last_snap = snap
        self._setup_stretch_display(snap)
        meta = snap["meta"]
        self.lbl_step.setText(f"step {meta['step']:,}")
        self.lbl_res.setText(f"res {meta['residual']:.2e}")
        self.lbl_perf.setText(f"{meta['steps_per_sec']:.1f} steps/s")
        perf = meta.get("performance")
        if perf:
            self.lbl_thrust.setText(f"{perf['F']:.4g} {perf['force_unit']}")
            self.lbl_thrust.setToolTip(
                f"Fx = {perf['Fx']:.4g} {perf['force_unit']}\n"
                f"Fy = {perf['Fy']:.4g} {perf['force_unit']}")
            self.lbl_mdot.setText(f"{perf['mdot']:.4g} {perf['mdot_unit']}")
            self.lbl_isp.setText(f"{perf['Isp']:.1f} s")
            self.lbl_ceff.setText(f"{perf['c_eff']:.0f} m/s")
            self.res_plot.setLabel("left", "F", units=perf["force_unit"])
            mix = PROPELLANT_MIX.get(self.cfg_panel.prop_combo.currentText())
            if mix and perf["mdot"] > 1e-12:
                fuel, ox, of = mix
                mf = perf["mdot"] / (1.0 + of)
                self.lbl_split.setText(
                    f"{fuel} {mf:.3g} + {ox} {perf['mdot'] - mf:.3g} "
                    f"{perf['mdot_unit']}")
                self.lbl_split.setToolTip(
                    f"O/F mass ratio {of:g}\n"
                    f"{fuel}: {mf:.4g} {perf['mdot_unit']}\n"
                    f"{ox}: {perf['mdot'] - mf:.4g} {perf['mdot_unit']}")
            else:
                self.lbl_split.setText("–")
        hist = snap.get("thrust_history")
        if hist:
            arr = np.array(hist, dtype=np.float64)
            self.res_curve.setData(arr[:, 0], arr[:, 1])
            from ..postproc import thrust_convergence
            conv, rel = thrust_convergence(hist)
            if rel == rel:                      # not NaN
                ok = ACTIVE_COLORS["perf"] if conv else "#C62828"
                self.lbl_conv.setText(
                    f"±{rel*100:.2f}% {'✓' if conv else '(settling)'}")
                self.lbl_conv.setStyleSheet(f"color: {ok};")
            else:
                self.lbl_conv.setText("–")
        self.refresh_view()
        self._record_frame(snap)

    # ------------------------------------------------------------- replay
    def _record_frame(self, snap):
        """Record the displayed field. The buffer covers the WHOLE run: when
        it fills up, every 2nd frame is dropped and the recording rate is
        halved (decimating recorder), so memory stays bounded while the
        replay always spans from step 0 to the latest step."""
        name = self.field_combo.currentText()
        arr = snap["fields"].get(name)
        if arr is None:
            return
        self._replay_count += 1
        if (self._replay_count - 1) % self._replay_stride != 0:
            return
        ds = max(1, max(arr.shape) // 500)
        self.replay_frames.append(
            (snap["meta"]["step"], name, arr[::ds, ::ds].astype(np.float16)))
        if len(self.replay_frames) > 400:
            self.replay_frames = self.replay_frames[::2]
            self._replay_stride *= 2
        n = len(self.replay_frames)
        self.replay_slider.blockSignals(True)
        self.replay_slider.setRange(0, n - 1)
        if not self.replay_timer.isActive():
            self.replay_slider.setValue(n - 1)
        self.replay_slider.blockSignals(False)
        self.replay_lbl.setText(f"{n}/{n}")
        if n >= 2:
            self.replay_bar.setVisible(True)

    def _replay_speed_changed(self, i):
        if self.replay_timer.isActive():
            self.replay_timer.start(int(1000 / (2, 5, 10, 20)[i]))

    def _replay_play_toggled(self, on):
        if on:
            if self.worker is not None and self.worker.running:
                self.btn_run.setChecked(False)
                self.toggle_run(False)
            if self.replay_slider.value() >= len(self.replay_frames) - 1:
                self.replay_slider.setValue(0)
            fps = (2, 5, 10, 20)[self.replay_speed.currentIndex()]
            self.replay_timer.start(int(1000 / fps))
            self.replay_play.setText("⏸︎")
        else:
            self.replay_timer.stop()
            self.replay_play.setText("▶︎")

    def _replay_tick(self):
        v = self.replay_slider.value()
        if v < len(self.replay_frames) - 1:
            self.replay_slider.setValue(v + 1)
        else:
            self.replay_play.setChecked(False)

    def _show_replay_frame(self, i):
        if not (0 <= i < len(self.replay_frames)):
            return
        step, name, arr16 = self.replay_frames[i]
        arr = arr16.astype(np.float32)
        finite = np.isfinite(arr)
        lo = float(np.nanmin(arr)) if finite.any() else 0.0
        hi = float(np.nanmax(arr)) if finite.any() else 1.0
        if hi <= lo:
            hi = lo + 1e-12
        disp = np.nan_to_num(arr, nan=lo)
        self.img_item.setImage(disp, autoLevels=False, levels=(lo, hi))
        if self.world_rect is not None:
            self.img_item.setRect(self.world_rect)
        self.cbar.setLevels((lo, hi))
        self.replay_lbl.setText(f"{i + 1}/{len(self.replay_frames)}")
        self.lbl_probe.setText(f"Replay — step {step:,} ({name})")

    def _replay_exit(self):
        self.replay_play.setChecked(False)
        if self.replay_frames:
            self.replay_slider.blockSignals(True)
            self.replay_slider.setValue(len(self.replay_frames) - 1)
            self.replay_slider.blockSignals(False)
        self.refresh_view()
        self.lbl_probe.setText("")

    @Slot(str)
    def on_error(self, msg: str):
        self.btn_run.setChecked(False)
        self.btn_run.setText("▶︎  Run")
        QMessageBox.critical(self, "Solver error", msg[-3000:])

    def on_mesh_toggled(self, on: bool):
        self.mesh_iso.setVisible(on and self.mask_lam is not None)
        self.mesh_grid.setVisible(on)
        if on:
            self._refresh_mesh_items()
        self._update_mesh_grid()

    def _refresh_mesh_items(self):
        if self.mask_lam is not None:
            # IsocurveItem expects col-major data; lam is row-major (ny, nx)
            self.mesh_iso.setData(self.mask_lam.T)

    def _update_mesh_grid(self, *args):
        if not self.mesh_chk.isChecked() or self.img_nx == 0:
            self.mesh_grid.setData([], [])
            return
        (x0, x1), (y0, y1) = self.vb.viewRange()
        dx = self.dx
        off = self.y_off
        x0 = max(x0, 0.0); x1 = min(x1, self.img_nx * dx)
        y0 = max(y0, -off); y1 = min(y1, self.img_ny * dx - off)
        if x1 <= x0 or y1 <= y0:
            self.mesh_grid.setData([], [])
            return
        i0, i1 = int(np.ceil(x0 / dx)), int(np.floor(x1 / dx))
        j0, j1 = int(np.ceil((y0 + off) / dx)), int(np.floor((y1 + off) / dx))
        n_lines = (i1 - i0 + 1) + (j1 - j0 + 1)
        if n_lines <= 0 or n_lines > 700:        # too far out: hide cell edges
            self.mesh_grid.setData([], [])
            return
        xv = np.arange(i0, i1 + 1) * dx
        yv = np.arange(j0, j1 + 1) * dx - off
        X = np.concatenate([np.repeat(xv, 2), np.tile([x0, x1], len(yv))])
        Y = np.concatenate([np.tile([y0, y1], len(xv)), np.repeat(yv, 2)])
        self.mesh_grid.setData(X, Y)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        apply_claude_theme(QApplication.instance(), self.dark_mode)
        self._apply_widget_theme()
        self.btn_theme.setText("◐")

    def _apply_widget_theme(self):
        """Re-style the pyqtgraph widgets and themed labels at runtime."""
        c = ACTIVE_COLORS
        if self._has_logo:
            self.title_lbl.setPixmap(
                self._logo_dark if self.dark_mode else self._logo_light)
        if not self._has_logo:
            self.title_lbl.setStyleSheet(
                "font-family: Georgia, 'Times New Roman', serif;"
                f"font-size: 26px; color: {c['text']}; background: transparent;")
        self.subtitle_lbl.setStyleSheet(
            f"color: {c['subtext']}; font-size: 12px; background: transparent;")
        for lbl in (self.lbl_thrust, self.lbl_mdot, self.lbl_isp, self.lbl_ceff):
            lbl.setStyleSheet(f"font-weight: bold; color: {c['perf']};"
                              "background: transparent;")
        fg = c["plot_fg"]
        self.glw.setBackground(c["bg"])
        self.res_plot.setBackground(c["bg"])
        axes = [self.plot.getAxis("left"), self.plot.getAxis("bottom"),
                self.res_plot.getPlotItem().getAxis("left"),
                self.res_plot.getPlotItem().getAxis("bottom")]
        try:
            axes.append(self.cbar.getAxis("right"))
        except Exception:
            pass
        for ax in axes:
            ax.setPen(pg.mkPen(fg))
            ax.setTextPen(pg.mkPen(fg))
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")
        self.res_plot.getPlotItem().setTitle("Thrust", color=fg)
        self.axis_line.setPen(pg.mkPen(c["axis_line"], width=1.5,
                                       style=Qt.DashLine))
        try:
            self.axis_line.label.setColor(QColor(c["axis_line"]))
        except Exception:
            pass
        if self._scalebar_width > 0:
            self._update_scalebar(self._scalebar_width)

    def refresh_view(self):
        if not self.last_snap:
            return
        name = self.field_combo.currentText()
        arr = self.last_snap["fields"].get(name)
        if arr is None:
            return
        rect = self.world_rect
        if self.auto_chk.isChecked():
            lo = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
            hi = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
            if hi <= lo:
                hi = lo + 1e-12
            self.lvl_min.setText(f"{lo:g}")
            self.lvl_max.setText(f"{hi:g}")
        else:
            try:
                lo = float(self.lvl_min.text()); hi = float(self.lvl_max.text())
            except ValueError:
                lo, hi = 0.0, 1.0
        sel = self.cmap_combo.currentText()
        cmap = get_cmap({"RdYlBu": "RdYlBu_r",
                         "Spectral": "Spectral_r"}.get(sel, sel))
        disp = np.nan_to_num(arr, nan=lo)
        if self._disp_idx is not None:          # remap to physical x extent
            disp = disp[:, self._disp_idx]
        self.img_item.setImage(disp, autoLevels=False, levels=(lo, hi))
        if rect is not None:
            self.img_item.setRect(rect)
        self.cbar.setColorMap(cmap)
        self.cbar.setLevels((lo, hi))

    def on_mouse_move(self, pos):
        if not self.last_snap:
            return
        p = self.vb.mapSceneToView(pos)
        xm, ym = p.x(), p.y()
        i, j = int(xm / self.dx), int((ym + self.y_off) / self.dx)
        if self._disp_idx is not None and 0 <= i < self._disp_w:
            i = int(self._disp_idx[i])      # display col -> computational col
        f = self.last_snap["fields"]
        m = f["Mach"]
        if 0 <= j < m.shape[0] and 0 <= i < m.shape[1]:
            if np.isnan(m[j, i]):
                self.lbl_probe.setText(f"({xm:.4f} m, {ym:.4f} m)  wall")
            else:
                self.lbl_probe.setText(
                    f"({xm:.4f} m, {ym:.4f} m)  "
                    f"M={m[j,i]:.3f}  p={f['Pressure [Pa]'][j,i]:.4g} Pa  "
                    f"T={f['Temperature [K]'][j,i]:.1f} K  "
                    f"ρ={f['Density [kg/m^3]'][j,i]:.4g} kg/m³  "
                    f"|V|={f['Velocity |V| [m/s]'][j,i]:.1f} m/s")

    # ------------------------------------------------------------- probe
    def toggle_probe(self, on: bool):
        self._probe_mode = on
        self._probe_p0 = None
        # freeze view dragging while picking so clicks register as points
        self.vb.setMouseEnabled(not on, not on)
        if on:
            if self.probe_dialog is None:
                from .probe_tools import ProbeDialog
                self.probe_dialog = ProbeDialog(self)
            self.probe_dialog.show()
            self.probe_dialog.raise_()
            self.statusBar().showMessage(
                "Probe: click the start point, then the end point "
                "(or use a preset in the dialog).", 8000)

    def _on_scene_click(self, ev):
        if not self._probe_mode or self.last_snap is None:
            return
        try:
            if ev.button() != Qt.LeftButton:
                return
        except Exception:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        pt = (float(p.x()), float(p.y()))
        if self._probe_p0 is None:
            self._probe_p0 = pt
            self.probe_pts.setData([pt[0]], [pt[1]])
            self.probe_line.setData([], [])
            self.statusBar().showMessage("Probe: now click the end point.", 8000)
        else:
            self._show_probe(self._probe_p0, pt)
            self._probe_p0 = None
        try:
            ev.accept()
        except Exception:
            pass

    def _show_probe(self, p0, p1):
        self.probe_line.setData([p0[0], p1[0]], [p0[1], p1[1]])
        self.probe_pts.setData([p0[0], p1[0]], [p0[1], p1[1]])
        if self.probe_dialog is None:
            from .probe_tools import ProbeDialog
            self.probe_dialog = ProbeDialog(self)
        self.probe_dialog.set_line(p0, p1)
        self.probe_dialog.show()
        self.probe_dialog.raise_()

    # ------------------------------------------------------------- IO
    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "config.json", "JSON (*.json)")
        if path:
            try:
                self.cfg_panel.get_config().save(path)
            except ValueError as e:
                QMessageBox.warning(self, "RocketCFD", str(e))

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if path:
            self.cfg_panel.set_config(SimConfig.load(path))

    def export_npz(self):
        if self.worker is None or self.worker.solver is None:
            QMessageBox.information(self, "RocketCFD", "Initialize and run the solver first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export fields", "fields.npz", "NumPy (*.npz)")
        if path:
            was = self.worker.running
            self.worker.running = False
            self.worker.export_npz(path)
            self.worker.running = was

    def export_view(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export view", "view.png", "PNG (*.png)")
        if path:
            from pyqtgraph.exporters import ImageExporter
            ImageExporter(self.glw.scene()).export(path)

    # ------------------------------------------------- sweep / report / video
    def open_sweep(self):
        if not self.png_path:
            QMessageBox.information(self, "RocketCFD", "Load an engine first.")
            return
        if self.sweep_dialog is None:
            from .sweep_tools import SweepDialog
            self.sweep_dialog = SweepDialog(self)
        self.sweep_dialog.show()
        self.sweep_dialog.raise_()

    def export_report(self):
        if self.last_snap is None or self.mask_ct is None:
            QMessageBox.information(
                self, "RocketCFD", "Initialize and run the solver first.")
            return
        name = Path(self.png_path).stem if self.png_path else "engine"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF report", f"{name}_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        was = None
        if self.worker is not None:
            was = self.worker.running
            self.worker.running = False
        self.statusBar().showMessage("Generating PDF report…")
        QApplication.processEvents()
        try:
            from ..report import generate_report
            out = generate_report(
                path, self.last_snap, self.cfg_panel.get_config(),
                mask_ct=self.mask_ct, dx=self.dx,
                axis_row=self._axis_row_interior(), y_off=self.y_off,
                thrust_history=self.last_snap.get("thrust_history"),
                mask_lam=self.mask_lam, engine_name=name,
                sweep_results=self._last_sweep)
            self.statusBar().showMessage(f"Report saved: {out}", 10000)
        except Exception:
            import traceback
            self.statusBar().clearMessage()
            QMessageBox.critical(self, "Report", traceback.format_exc()[-3000:])
        finally:
            if self.worker is not None and was:
                self.worker.running = was

    def export_mp4(self):
        if len(self.replay_frames) < 2:
            QMessageBox.information(
                self, "RocketCFD",
                "Run the solver first — the replay recorder needs at least "
                "two frames before a video can be exported.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export video", "run.mp4", "MP4 video (*.mp4)")
        if not path:
            return
        was = None
        if self.worker is not None:
            was = self.worker.running
            self.worker.running = False
        frames = list(self.replay_frames)      # snapshot: solver may resume
        fps = (2, 5, 10, 20)[self.replay_speed.currentIndex()]
        self.statusBar().showMessage("Encoding MP4…")
        QApplication.processEvents()
        try:
            import imageio.v2 as imageio
            cmap = get_cmap(self.cmap_combo.currentText())
            lut = cmap.getLookupTable(0.0, 1.0, 256)[:, :3].astype(np.uint8)
            # global color range so the video does not flicker frame-to-frame
            lo, hi = math.inf, -math.inf
            for _, _, a16 in frames:
                a = a16.astype(np.float32)
                fin = np.isfinite(a)
                if fin.any():
                    lo = min(lo, float(a[fin].min()))
                    hi = max(hi, float(a[fin].max()))
            if not math.isfinite(lo) or hi <= lo:
                lo, hi = 0.0, 1.0
            with imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=1) as wr:
                for step, fname, a16 in frames:
                    a = a16.astype(np.float32)
                    a = np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo)
                    idx = ((a - lo) * (255.0 / (hi - lo))).clip(0, 255)
                    rgb = lut[idx.astype(np.uint8)]
                    # upscale small fields so players cope better; even dims
                    k = max(1, 720 // max(rgb.shape[0], 1))
                    if k > 1:
                        rgb = rgb.repeat(k, axis=0).repeat(k, axis=1)
                    if rgb.shape[0] % 2:
                        rgb = rgb[:-1]
                    if rgb.shape[1] % 2:
                        rgb = rgb[:, :-1]
                    wr.append_data(rgb)
            self.statusBar().showMessage(
                f"Video saved ({len(frames)} frames @ {fps} fps): {path}", 10000)
        except Exception:
            import traceback
            self.statusBar().clearMessage()
            QMessageBox.critical(self, "Video export",
                                 traceback.format_exc()[-3000:])
        finally:
            if self.worker is not None and was:
                self.worker.running = was


# ---- Claude visual style: warm surfaces, terracotta accent; light + dark ----
LIGHT_COLORS = dict(
    bg="#FAF9F5", panel="#F0EEE5", card="#FFFFFF",
    border="#E3DFD3", border2="#D5D0C2",
    text="#141413", subtext="#87837A",
    accent="#D97757", accent_h="#CB6B4A", accent_p="#B85B3E",
    accent_dis="#E9C9BA", btn_press="#E8E4D8", text_dis="#B5B1A5",
    plot_fg="#57534A", perf="#B0522F", axis_line="#0E7490",
    scalebar=(40, 40, 38, 220),
)
DARK_COLORS = dict(
    bg="#262624", panel="#30302E", card="#1F1E1D",
    border="#3E3C38", border2="#4A4843",
    text="#F0EEE5", subtext="#A8A49B",
    accent="#D97757", accent_h="#E08D6D", accent_p="#B85B3E",
    accent_dis="#6E4434", btn_press="#3A3936", text_dis="#6F6C64",
    plot_fg="#C2BEB3", perf="#E5926F", axis_line="#33C6E8",
    scalebar=(235, 233, 226, 220),
)

# module-level so MainWindow can read the active palette
ACTIVE_COLORS = dict(LIGHT_COLORS)
ACTIVE_DARK = [False]                 # mutable: set by apply_claude_theme

# kept for back-compat with the QSS template below
C_BG      = "{bg}"
C_PANEL   = "{panel}"
C_BORDER  = "{border}"
C_BORDER2 = "{border2}"
C_TEXT    = "{text}"
C_SUBTEXT = "{subtext}"
C_ACCENT  = "{accent}"
C_ACCENT_H = "{accent_h}"
C_ACCENT_P = "{accent_p}"
C_CARD    = "{card}"
C_BTNPRESS = "{btn_press}"
C_TEXTDIS = "{text_dis}"
C_ACCENTDIS = "{accent_dis}"

CLAUDE_QSS_TEMPLATE = f"""
QMainWindow, QDialog {{ background: {C_BG}; }}
QWidget {{ color: {C_TEXT}; font-size: 13px; }}
QGroupBox {{
    background: {C_PANEL}; border: 1px solid {C_BORDER};
    border-radius: 10px; margin-top: 16px; padding: 10px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 5px;
    color: {C_SUBTEXT}; font-weight: 600;
}}
QLineEdit {{
    background: {C_CARD}; border: 1px solid {C_BORDER2}; border-radius: 6px;
    padding: 3px 7px; selection-background-color: {C_ACCENT};
    selection-color: #FFFFFF;
}}
QLineEdit:focus {{ border: 1px solid {C_ACCENT}; }}
QComboBox {{
    background: {C_CARD}; border: 1px solid {C_BORDER2}; border-radius: 6px;
    padding: 3px 7px;
}}
QComboBox:focus {{ border: 1px solid {C_ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {C_CARD}; border: 1px solid {C_BORDER2};
    selection-background-color: {C_ACCENT}; selection-color: #FFFFFF;
}}
QPushButton {{
    background: {C_CARD}; border: 1px solid {C_BORDER2}; border-radius: 8px;
    padding: 6px 12px;
}}
QPushButton:hover {{ background: {C_PANEL}; }}
QPushButton:pressed {{ background: {C_BTNPRESS}; }}
QPushButton:disabled {{ color: {C_TEXTDIS}; background: {C_BG}; }}
QPushButton[accent="true"] {{
    background: {C_ACCENT}; color: #FFFFFF; border: none; font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background: {C_ACCENT_H}; }}
QPushButton[accent="true"]:pressed,
QPushButton[accent="true"]:checked {{ background: {C_ACCENT_P}; }}
QPushButton[accent="true"]:disabled {{ background: {C_ACCENTDIS}; color: #FFFFFF; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {C_BORDER2};
    border-radius: 4px; background: {C_CARD};
}}
QCheckBox::indicator:checked {{ background: {C_ACCENT}; border-color: {C_ACCENT}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {C_BORDER2}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QTabWidget::pane {{ border: 1px solid {C_BORDER}; border-radius: 6px; }}
QTabBar::tab {{
    background: {C_PANEL}; border: 1px solid {C_BORDER};
    padding: 6px 16px; margin-right: 2px; color: {C_SUBTEXT};
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ background: {C_BG}; color: {C_TEXT}; font-weight: 600; }}
QSpinBox {{
    background: {C_CARD}; border: 1px solid {C_BORDER2}; border-radius: 6px;
    padding: 3px 6px;
}}
QSlider::groove:horizontal {{
    height: 5px; background: {C_BORDER2}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_ACCENT}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QStatusBar {{ background: {C_PANEL}; border-top: 1px solid {C_BORDER}; }}
QStatusBar QLabel {{ color: {C_SUBTEXT}; }}
QSplitter::handle {{ background: {C_BORDER}; }}
QToolTip {{
    background: {C_TEXT}; color: {C_BG}; border: none; padding: 4px 8px;
}}
"""


def build_qss(c: dict) -> str:
    qss = CLAUDE_QSS_TEMPLATE
    for k, v in c.items():
        if isinstance(v, str):
            qss = qss.replace("{" + k + "}", v)
    return qss


def apply_claude_theme(app: QApplication, dark: bool = False):
    c = DARK_COLORS if dark else LIGHT_COLORS
    ACTIVE_COLORS.clear()
    ACTIVE_COLORS.update(c)
    ACTIVE_DARK[0] = dark
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["card"]))
    pal.setColor(QPalette.AlternateBase, QColor(c["panel"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["card"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.PlaceholderText, QColor(c["subtext"]))
    app.setPalette(pal)
    app.setStyleSheet(build_qss(c))
    pg.setConfigOptions(background=c["bg"], foreground=c["plot_fg"],
                        antialias=True)


# kept for backwards compatibility (tests, scripts)
apply_dark_theme = apply_claude_theme


def main():
    app = QApplication(sys.argv)
    apply_claude_theme(app, dark=True)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
