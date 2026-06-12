"""Exercise MainWindow.export_mp4 offscreen with synthetic replay frames."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

app = QApplication([])
from rocketcfd.gui.main import MainWindow

w = MainWindow()
rng = np.random.default_rng(0)
ny, nx = 120, 300
for i in range(30):
    arr = (np.linspace(0, 1, nx)[None, :] * np.ones((ny, 1)) * (i + 1)
           + rng.normal(0, 0.05, (ny, nx))).astype(np.float16)
    arr[:10, :10] = np.nan                      # NaN-masked walls
    w.replay_frames.append((i * 25, "Mach", arr))

out = os.path.join(tempfile.gettempdir(), "tachyon_test_video.mp4")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, "MP4"))
w.export_mp4()
assert os.path.exists(out), "no mp4 written"
sz = os.path.getsize(out)
print(f"video written: {out} ({sz/1024:.0f} KiB)")
assert sz > 5_000, "mp4 suspiciously small"
print("mp4 export OK")
