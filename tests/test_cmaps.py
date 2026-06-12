"""Verify all GUI colormaps resolve (including matplotlib fallbacks)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication
app = QApplication([])

from rocketcfd.gui.main import get_cmap

for name in ("turbo", "viridis", "plasma", "inferno", "magma", "cividis",
             "RdYlBu", "Spectral"):
    cm = get_cmap(name)
    lut = cm.getLookupTable(nPts=16)
    assert lut.shape[0] == 16, name
    print(f"{name:10s} OK  first={lut[0].tolist()} last={lut[-1].tolist()}")
