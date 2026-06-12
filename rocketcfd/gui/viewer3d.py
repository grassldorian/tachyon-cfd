"""3D exhaust viewer: revolve the axisymmetric solution into a volume and
render it with OpenGL (rotate/zoom with the mouse).

The engine structure is rendered as a smooth surface-of-revolution mesh
(triangulated from the sub-pixel embedded-boundary level set), while the
exhaust is a voxel volume colored by the selected field. A colormap range
slider clips low/high values out of the volume so the inner structure of
the plume stays visible. Requires PyOpenGL.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
except Exception:                       # PyOpenGL missing
    gl = None
    HAS_GL = False

import pyqtgraph as pg

from ..cuda_kernels import axis_j
from ..mask import WALL

FIELDS_3D = ["Mach", "Temperature [K]", "Density [kg/m^3]",
             "Velocity |V| [m/s]", "Pressure [Pa]", "Schlieren |grad rho|"]


def _radial_profile(arr: np.ndarray, axis_row: float):
    """NaN-aware radial profile (r, x) averaged over both halves."""
    ny, nx = arr.shape
    ju = int(np.floor(axis_row))
    r_max = max(ju + 1, ny - ju - 1)
    r_idx = np.arange(r_max)
    j_up = ju - r_idx
    j_dn = ju + 1 + r_idx
    up = arr[np.clip(j_up, 0, ny - 1), :].astype(np.float32)
    dn = arr[np.clip(j_dn, 0, ny - 1), :].astype(np.float32)
    up[j_up < 0] = np.nan
    dn[j_dn > ny - 1] = np.nan
    both = np.isfinite(up) & np.isfinite(dn)
    prof = np.where(both, 0.5 * (up + dn),
                    np.where(np.isfinite(up), up, dn))
    return prof                          # (r_max, nx), NaN where no data


def build_volume(field: np.ndarray, ct: np.ndarray, axis_row: float,
                 cmap: pg.ColorMap, n_target: int = 144,
                 alpha_max: float = 110.0, clip=(0.0, 1.0),
                 cutaway: bool = False):
    """Voxel RGBA volume (nx, n, n, 4) of the revolved field.

    clip = (lo_frac, hi_frac): normalized values outside this window are
    rendered fully transparent. cutaway removes the z > 0 half so the
    meridional cross-section is visible.
    """
    prof = _radial_profile(field, axis_row)
    solid = _radial_profile((ct == WALL).astype(np.float32), axis_row)
    r_max = prof.shape[0]
    nx = prof.shape[1]

    f = max(1, int(np.ceil(max(nx, 2 * r_max) / n_target)))
    prof_ds = prof[::f, ::f]
    solid_ds = solid[::f, ::f]
    nr, nxd = prof_ds.shape

    finite = np.isfinite(prof_ds)
    lo = float(np.nanmin(prof_ds)) if finite.any() else 0.0
    hi = float(np.nanmax(prof_ds)) if finite.any() else 1.0
    if hi <= lo:
        hi = lo + 1e-12
    norm = np.clip((np.nan_to_num(prof_ds, nan=lo) - lo) / (hi - lo), 0, 1)
    lut = cmap.getLookupTable(nPts=256)            # (256, 3)
    clip_lo, clip_hi = clip

    n = 2 * nr
    yy, zz = np.meshgrid(np.arange(n) - (nr - 0.5),
                         np.arange(n) - (nr - 0.5), indexing="ij")
    rr = np.sqrt(yy ** 2 + zz ** 2)
    inside = rr < nr
    # linear interpolation in radius (removes the low-poly ring artifacts)
    rf = np.clip(rr - 0.5, 0.0, nr - 1.001)
    r0 = rf.astype(np.int32)
    w1 = (rf - r0).astype(np.float32)
    r1 = np.minimum(r0 + 1, nr - 1)

    vol = np.zeros((nxd, n, n, 4), dtype=np.ubyte)
    for ix in range(nxd):
        col = norm[:, ix]
        v = col[r0] * (1.0 - w1) + col[r1] * w1     # (n, n), smooth
        ci = (v * 255).astype(np.int32)
        rgb = lut[ci]
        a = (alpha_max * v ** 1.6).astype(np.ubyte)
        a[(v < clip_lo) | (v > clip_hi)] = 0        # range-slider clipping
        scol = solid_ds[:, ix]
        s = scol[r0] * (1.0 - w1) + scol[r1] * w1
        a[s > 0.5] = 0                              # engine: mesh, not voxels
        a[~inside] = 0
        vol[ix, :, :, :3] = rgb
        vol[ix, :, :, 3] = a
    if cutaway:
        vol[:, :, nr:, 3] = 0          # remove the z > 0 half
    return vol, f, lo, hi


def build_engine_mesh(solid_half: np.ndarray, dx: float, n_theta: int = 160,
                      half: bool = False):
    """Surface-of-revolution mesh of the engine from the smooth solid field.

    solid_half: (r, x) solidity in [0, 1] (1 inside walls). The 0.5
    iso-contour is revolved with n_theta segments (half=True revolves only
    the z <= 0 half for a cutaway section). Returns (verts, faces)
    or (None, None) when there is no contour.
    """
    data = np.nan_to_num(solid_half, nan=0.0)
    try:
        paths = pg.functions.isocurve(data, 0.5, connected=True)
    except Exception:
        return None, None
    if not paths:
        return None, None
    if half:
        theta = np.linspace(np.pi, 2.0 * np.pi, n_theta // 2 + 1)
        wrap = False
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        wrap = True
    nt = len(theta)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    all_v, all_f, off = [], [], 0
    for path in paths:
        P = np.asarray(path, dtype=np.float32)      # (n, 2): (r_idx, x_idx)
        if len(P) < 3:
            continue
        r = (P[:, 0] + 0.5) * dx
        x = (P[:, 1] + 0.5) * dx
        npts = len(P)
        V = np.empty((npts, nt, 3), dtype=np.float32)
        V[..., 0] = x[:, None]
        V[..., 1] = r[:, None] * cos_t[None, :]
        V[..., 2] = r[:, None] * sin_t[None, :]
        closed = np.hypot(P[0, 0] - P[-1, 0], P[0, 1] - P[-1, 1]) < 1e-6
        rows = npts if closed else npts - 1
        p_i = np.arange(rows)
        p_n = (p_i + 1) % npts
        k_i = np.arange(nt if wrap else nt - 1)
        k_n = (k_i + 1) % nt
        a = (p_i[:, None] * nt + k_i[None, :]).ravel()
        b = (p_n[:, None] * nt + k_i[None, :]).ravel()
        c = (p_n[:, None] * nt + k_n[None, :]).ravel()
        d = (p_i[:, None] * nt + k_n[None, :]).ravel()
        f1 = np.stack([a, b, c], axis=1)
        f2 = np.stack([a, c, d], axis=1)
        all_v.append(V.reshape(-1, 3))
        all_f.append(np.concatenate([f1, f2], axis=0) + off)
        off += npts * nt
    if not all_v:
        return None, None
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)


class CmapRangeSlider(QWidget):
    """Colormap ramp with two draggable end handles to clip the range."""

    rangeChanged = Signal(float, float)             # emitted on release

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(30)
        self.lo = 0.0
        self.hi = 1.0
        self._lut = None
        self._drag = None                            # "lo" | "hi"

    def set_cmap(self, cmap: pg.ColorMap):
        self._lut = cmap.getLookupTable(nPts=128)
        self.update()

    # ---- geometry: bar with 8 px margins for the handles
    def _frac_to_x(self, f):
        return 8 + f * (self.width() - 16)

    def _x_to_frac(self, x):
        return min(max((x - 8) / max(self.width() - 16, 1), 0.0), 1.0)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bar_y, bar_h = 10, 12
        if self._lut is not None:
            grad = QLinearGradient(8, 0, self.width() - 8, 0)
            for i in range(0, 128, 4):
                c = self._lut[i]
                grad.setColorAt(i / 127.0, QColor(int(c[0]), int(c[1]), int(c[2])))
            p.setPen(Qt.NoPen)
            p.setBrush(grad)
            p.drawRoundedRect(8, bar_y, self.width() - 16, bar_h, 4, 4)
        # gray out the clipped parts
        p.setBrush(QColor(70, 70, 68, 200))
        x_lo, x_hi = self._frac_to_x(self.lo), self._frac_to_x(self.hi)
        if self.lo > 0.001:
            p.drawRoundedRect(8, bar_y, int(x_lo) - 8, bar_h, 4, 4)
        if self.hi < 0.999:
            p.drawRoundedRect(int(x_hi), bar_y, self.width() - 8 - int(x_hi),
                              bar_h, 4, 4)
        # handles
        for x in (x_lo, x_hi):
            p.setPen(QPen(QColor(250, 250, 248), 1.5))
            p.setBrush(QColor(217, 119, 87))
            p.drawEllipse(int(x) - 6, bar_y - 3, 12, bar_h + 6)
        p.end()

    def mousePressEvent(self, ev):
        f = self._x_to_frac(ev.position().x())
        self._drag = "lo" if abs(f - self.lo) <= abs(f - self.hi) else "hi"
        self.mouseMoveEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        f = self._x_to_frac(ev.position().x())
        if self._drag == "lo":
            self.lo = min(f, self.hi - 0.02)
        else:
            self.hi = max(f, self.lo + 0.02)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            self._drag = None
            self.rangeChanged.emit(self.lo, self.hi)


class Viewer3DTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.vol_item = None
        self.mesh_item = None
        lay = QHBoxLayout(self)

        side = QWidget()
        side.setMaximumWidth(380)
        side.setMinimumWidth(340)
        sl = QVBoxLayout(side)
        box = QGroupBox("3D exhaust view")
        f = QFormLayout(box)
        self.field_combo = QComboBox()
        self.field_combo.addItems(FIELDS_3D)
        f.addRow("Field", self.field_combo)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(20, 255)
        self.alpha_slider.setValue(110)
        f.addRow("Opacity", self.alpha_slider)
        self.range_slider = CmapRangeSlider()
        self.range_slider.setToolTip(
            "Drag the handles inward to hide low/high values\n"
            "and reveal the structure inside the plume.")
        f.addRow("Show range", self.range_slider)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128 (fast)", "176", "224 (fine)", "288 (max)"])
        self.res_combo.setCurrentIndex(1)
        f.addRow("Resolution", self.res_combo)
        from PySide6.QtWidgets import QCheckBox
        self.cut_chk = QCheckBox("Cutaway (half section)")
        self.cut_chk.setToolTip("Slice the engine and plume in half to look\n"
                                "at the inside cross-section.")
        self.cut_chk.toggled.connect(self._rebuild_if_active)
        f.addRow(self.cut_chk)
        btn = QPushButton("⟳  Update from simulation")
        btn.setProperty("accent", True)
        btn.clicked.connect(self.update_from_sim)
        f.addRow(btn)
        views = QHBoxLayout()
        for label, elev, azim in (("Top", 90, -90), ("Bottom", -90, -90),
                                  ("Left", 0, 180), ("Right", 0, 0),
                                  ("Iso", 28, 135)):
            b = QPushButton(label)
            b.setMinimumWidth(58)
            b.clicked.connect(lambda _, e=elev, a=azim: self._set_view(e, a))
            views.addWidget(b)
        f.addRow(views)
        hint = QLabel("Run an axisymmetric simulation, then update.\n"
                      "Drag to orbit, wheel to zoom.")
        hint.setStyleSheet("color: #87837A; font-size: 11px;")
        f.addRow(hint)
        sl.addWidget(box)
        sl.addStretch(1)
        lay.addWidget(side)

        self.range_slider.rangeChanged.connect(self._on_range_changed)

        if HAS_GL:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor(38, 38, 36)
            grid = gl.GLGridItem()
            grid.scale(0.1, 0.1, 0.1)
            self.view.addItem(grid)
            lay.addWidget(self.view, 1)
        else:
            self.view = None
            lbl = QLabel("3D view requires PyOpenGL:\n\n    pip install PyOpenGL")
            lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(lbl, 1)

    def _on_range_changed(self, lo, hi):
        self._rebuild_if_active()

    def _rebuild_if_active(self, *args):
        if self.vol_item is not None:
            self.update_from_sim()

    def _set_view(self, elevation, azimuth):
        if self.view is not None:
            dist = self.view.opts.get("distance", 1.0)
            self.view.setCameraPosition(distance=dist, elevation=elevation,
                                        azimuth=azimuth)

    def _cmap(self):
        from .main import get_cmap
        sel = self.main.cmap_combo.currentText()
        return get_cmap({"RdYlBu": "RdYlBu_r",
                         "Spectral": "Spectral_r"}.get(sel, sel))

    def update_from_sim(self):
        if not HAS_GL:
            return
        snap = self.main.last_snap
        if snap is None or self.main.mask_ct is None:
            QMessageBox.information(self, "RocketCFD",
                                    "Initialize and run a simulation first.")
            return
        name = self.field_combo.currentText()
        arr = snap["fields"].get(name)
        if arr is None:
            return
        try:
            cfg = self.main.cfg_panel.get_config()
            axis_row = axis_j(cfg, arr.shape[0]) - 2.0
        except ValueError:
            axis_row = arr.shape[0] / 2.0 - 0.5
        cmap = self._cmap()
        self.range_slider.set_cmap(cmap)
        n_target = (128, 176, 224, 288)[self.res_combo.currentIndex()]
        cutaway = self.cut_chk.isChecked()
        vol, f, lo, hi = build_volume(
            arr, self.main.mask_ct, axis_row, cmap, n_target=n_target,
            alpha_max=float(self.alpha_slider.value()),
            clip=(self.range_slider.lo, self.range_slider.hi),
            cutaway=cutaway)

        # ---- exhaust volume ----
        if self.vol_item is not None:
            self.view.removeItem(self.vol_item)
            self.vol_item = None
        s = self.main.dx * f
        self.vol_item = gl.GLVolumeItem(vol, sliceDensity=2, smooth=True)
        nx_, n_, _ = vol.shape[:3]
        self.vol_item.translate(-nx_ / 2 * s, -n_ / 2 * s, -n_ / 2 * s,
                                local=False)
        self.vol_item.scale(s, s, s, local=True)

        # ---- engine surface mesh (full resolution, smooth) ----
        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None
        if self.main.mask_lam is not None:
            solid_half = _radial_profile(1.0 - self.main.mask_lam, axis_row)
        else:
            solid_half = _radial_profile(
                (self.main.mask_ct == WALL).astype(np.float32), axis_row)
        verts, faces = build_engine_mesh(solid_half, self.main.dx,
                                         half=cutaway)
        if verts is not None:
            md = gl.MeshData(vertexes=verts, faces=faces)
            self.mesh_item = gl.GLMeshItem(
                meshdata=md, smooth=True, shader="shaded",
                color=(0.82, 0.82, 0.85, 1.0), glOptions="opaque")
            # mesh is in meters with the axis already at y = z = 0; only the
            # x origin needs to match the volume placement
            self.mesh_item.translate(-nx_ / 2 * s, 0.0, 0.0, local=False)
            self.view.addItem(self.mesh_item)
        self.view.addItem(self.vol_item)
        self.view.setCameraPosition(distance=nx_ * s * 1.6)
