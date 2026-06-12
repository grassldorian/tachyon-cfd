# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Tachyon CFD.

Build from the project root:
    pyinstaller packaging\\RocketCFD.spec --noconfirm

Bundles CuPy with the pip-wheel NVRTC compiler + CUDA headers so kernels
compile at runtime on any machine with an NVIDIA driver (CUDA 12.x).
Verify a build with:  dist\\TachyonCFD\\TachyonCFD.exe --selftest
"""
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

SITE = Path(sysconfig.get_paths()["purelib"])
ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "examples"), "examples"),
    (str(SITE / "nvidia"), "nvidia"),          # nvrtc DLLs + CUDA headers
]
if (ROOT / "assets").exists():
    datas.append((str(ROOT / "assets"), "assets"))
if (ROOT / "docs").exists():
    datas.append((str(ROOT / "docs"), "docs"))
binaries = []
hiddenimports = ["fastrlock", "fastrlock.rlock", "pyqtgraph.opengl",
                 # stdlib modules that dynamic imports (CuPy etc.) hide from
                 # PyInstaller's static analysis:
                 "select", "selectors", "socket", "unicodedata", "graphlib",
                 "queue", "secrets", "statistics", "csv", "configparser",
                 "uuid", "zoneinfo", "decimal", "fractions", "tempfile",
                 "http.client", "xml.etree.ElementTree"]

for pkg in ("cupy", "cupy_backends", "pyqtgraph", "OpenGL",
            # report/sweep plotting + MP4 export (bundled ffmpeg binary)
            "matplotlib", "imageio", "imageio_ffmpeg"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "run_gui.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[str(ROOT / "packaging" / "pyi_rth_cuda.py")],
    excludes=["tkinter", "IPython", "jedi"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TachyonCFD",
    icon=str(ROOT / "assets" / "tachyon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TachyonCFD",
)
