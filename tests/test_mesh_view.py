"""Screenshot test for the mesh view toggle (no solver needed)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rocketcfd.gui.main import MainWindow, apply_claude_theme

app = QApplication([])
apply_claude_theme(app)
win = MainWindow()
win.resize(1500, 950)
win.show()

win.load_image_path("examples/nozzle_small.png")
win.mesh_chk.setChecked(True)

# zoom into the throat region (around x=0.15 m, y=0.16 m)
win.vb.setRange(xRange=(0.130, 0.180), yRange=(0.135, 0.185), padding=0)

def shoot():
    win.grab().save("gui_mesh_test.png")
    print("screenshot -> gui_mesh_test.png")
    app.quit()

QTimer.singleShot(900, shoot)
app.exec()
