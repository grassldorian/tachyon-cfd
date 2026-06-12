"""Cycle through all display fields and report their ranges + screenshots."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rocketcfd.gui.main import MainWindow, apply_claude_theme

app = QApplication([])
apply_claude_theme(app)
win = MainWindow()
win.resize(1400, 900)
win.show()
win.load_image_path("examples/nozzle_small.png")
win.initialize()

def on_init():
    win.btn_run.setChecked(True)
    win.toggle_run(True)
    QTimer.singleShot(6000, check)

def check():
    win.toggle_run(False)
    snap = win.last_snap
    if snap is None:
        print("NO SNAPSHOT")
        app.quit()
        return
    for name, arr in snap["fields"].items():
        finite = np.isfinite(arr)
        print(f"{name:42s} finite={finite.sum():7d}/{arr.size}  "
              f"min={np.nanmin(arr):.5g}  max={np.nanmax(arr):.5g}")
    state = {"i": 0, "names": list(snap["fields"].keys())}

    def next_field():
        if state["i"] >= len(state["names"]):
            app.quit()
            return
        name = state["names"][state["i"]]
        win.field_combo.setCurrentText(name)
        safe = name.split(" [")[0].replace(" ", "_").replace("|", "").replace("/", "_").lower()
        def grab(n=name, s=safe):
            win.glw.grab().save(f"field_{state['i']:02d}_{s}.png")
            state["i"] += 1
            next_field()
        QTimer.singleShot(400, grab)
    next_field()

win.worker.initialized.connect(on_init)
win.worker.error.connect(lambda m: (print("ERROR:", m), app.quit()))
app.exec()
