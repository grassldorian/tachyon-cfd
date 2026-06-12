"""End-to-end GUI test: load drawing, init solver, run, screenshot the window.

Usage: python tests/test_gui_drive.py [image] [shot.png] [run_ms] [--axi]
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rocketcfd.gui.main import MainWindow, apply_dark_theme

args = [a for a in sys.argv[1:] if a != "--axi"]
AXI = "--axi" in sys.argv
IMG = args[0] if len(args) > 0 else "examples/nozzle_small.png"
SHOT = args[1] if len(args) > 1 else "gui_test.png"
RUN_MS = int(args[2]) if len(args) > 2 else 8000

app = QApplication([])
apply_dark_theme(app)
win = MainWindow()
win.resize(1500, 950)
win.show()

if AXI:
    win.cfg_panel.axi_chk.setChecked(True)

win.load_image_path(IMG)
win.initialize()

def on_init():
    win.btn_run.setChecked(True)
    win.toggle_run(True)
    QTimer.singleShot(RUN_MS, finish)

def finish():
    win.toggle_run(False)
    QTimer.singleShot(800, shoot)

def shoot():
    pix = win.grab()
    pix.save(SHOT)
    snap = win.last_snap
    if snap:
        m = snap["meta"]
        print(f"step={m['step']} residual={m['residual']:.3e} "
              f"steps/s={m['steps_per_sec']:.1f}")
        perf = m.get("performance")
        if perf:
            print(f"F={perf['F']:.4g} {perf['force_unit']}  "
                  f"mdot={perf['mdot']:.4g} {perf['mdot_unit']}  "
                  f"Isp={perf['Isp']:.1f} s")
    print(f"screenshot -> {SHOT}")
    # second shot with inverted (dark) colors
    win.toggle_theme()
    QTimer.singleShot(800, shoot_dark)

def shoot_dark():
    shotd = SHOT.replace(".png", "_dark.png")
    win.grab().save(shotd)
    print(f"screenshot -> {shotd}")
    win.shutdown_worker()
    app.quit()

win.worker.initialized.connect(on_init)
win.worker.error.connect(lambda m: (print("ERROR:", m), app.quit()))

app.exec()
