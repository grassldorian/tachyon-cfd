"""Engine library: browse all PNG/SVG drawings in a folder with previews."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout,
)

from ..mask import load_image

_SETTINGS = ("RocketCFD", "RocketCFD")


import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def library_dir() -> Path:
    d = QSettings(*_SETTINGS).value("library_dir", "")
    if d:
        return Path(d)
    if getattr(sys, "frozen", False):       # packaged EXE: keep user data
        return Path.home() / "Documents" / "RocketCFD" / "engines"
    return project_root() / "engines"


def set_library_dir(d: Path):
    QSettings(*_SETTINGS).setValue("library_dir", str(d))


def ensure_library(d: Path):
    """Create the library folder; seed it with the bundled examples."""
    if d.exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    ex = project_root() / "examples"
    if ex.exists():
        for f in list(ex.glob("*.png")) + list(ex.glob("*.svg")):
            try:
                shutil.copy(f, d / f.name)
            except OSError:
                pass


def make_thumb(path: Path, size: int = 110) -> QIcon:
    try:
        rgb = load_image(str(path), svg_raster_px=192)
        h, w = rgb.shape[:2]
        s = max(1, max(h, w) // 256)
        rgb = np.ascontiguousarray(rgb[::s, ::s])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(img).scaled(
            size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QIcon(pm)
    except Exception:
        return QIcon()


class EngineLibraryDialog(QDialog):
    """Folder-based engine browser. `selected` holds the chosen file path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Engine library")
        self.resize(760, 540)
        self.selected: str | None = None
        self.lib_dir = library_dir()
        ensure_library(self.lib_dir)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.lbl_dir = QLabel(str(self.lib_dir))
        self.lbl_dir.setStyleSheet("color: #87837A;")
        top.addWidget(self.lbl_dir, 1)
        b_dir = QPushButton("Change folder…")
        b_dir.clicked.connect(self.change_dir)
        b_imp = QPushButton("Import file…")
        b_imp.clicked.connect(self.import_file)
        top.addWidget(b_dir)
        top.addWidget(b_imp)
        lay.addLayout(top)

        self.listw = QListWidget()
        self.listw.setViewMode(QListWidget.IconMode)
        self.listw.setIconSize(QSize(110, 110))
        self.listw.setGridSize(QSize(140, 150))
        self.listw.setResizeMode(QListWidget.Adjust)
        self.listw.setWordWrap(True)
        self.listw.itemDoubleClicked.connect(lambda *_: self.open_selected())
        lay.addWidget(self.listw, 1)

        bottom = QHBoxLayout()
        b_other = QPushButton("Open other file…")
        b_other.clicked.connect(self.open_other)
        bottom.addWidget(b_other)
        bottom.addStretch(1)
        b_open = QPushButton("Open")
        b_open.setProperty("accent", True)
        b_open.clicked.connect(self.open_selected)
        b_cancel = QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        bottom.addWidget(b_open)
        bottom.addWidget(b_cancel)
        lay.addLayout(bottom)
        self.reload()

    def reload(self):
        self.listw.clear()
        files = sorted(
            list(self.lib_dir.glob("*.png")) + list(self.lib_dir.glob("*.svg")),
            key=lambda p: p.name.lower())
        for f in files:
            it = QListWidgetItem(make_thumb(f), f.name)
            it.setData(Qt.UserRole, str(f))
            self.listw.addItem(it)

    def change_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Engine library folder",
                                             str(self.lib_dir))
        if d:
            self.lib_dir = Path(d)
            set_library_dir(self.lib_dir)
            self.lbl_dir.setText(d)
            self.reload()

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import engine drawing", "",
            "Engine drawings (*.png *.svg)")
        if path:
            try:
                shutil.copy(path, self.lib_dir / Path(path).name)
            except OSError:
                pass
            self.reload()

    def open_selected(self):
        it = self.listw.currentItem()
        if it is not None:
            self.selected = it.data(Qt.UserRole)
            self.accept()

    def open_other(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open engine drawing", "",
            "Engine drawings (*.png *.svg)")
        if path:
            self.selected = path
            self.accept()
