"""Screenshot of the designer tab with a drawn engine."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF, QTimer
from PySide6.QtWidgets import QApplication

app = QApplication([])
from rocketcfd.gui.main import MainWindow, apply_claude_theme
from rocketcfd.gui.designer import COL_INLET, COL_OUTLET, COL_WALL
apply_claude_theme(app, dark=True)
win = MainWindow()
win.resize(1500, 950)
win.show()
win.tabs.setCurrentIndex(1)

c = win.designer.canvas
c.pen_width = 10
c.pen_color = COL_WALL
c._draw_line(QPointF(100, 360), QPointF(380, 360))
c._draw_line(QPointF(380, 360), QPointF(480, 455))
c._draw_line(QPointF(480, 455), QPointF(850, 330))
c._draw_line(QPointF(100, 360), QPointF(100, 500))
c.pen_color = COL_INLET
c._draw_line(QPointF(112, 372), QPointF(112, 500))
c.pen_color = COL_OUTLET
c._draw_line(QPointF(985, 20), QPointF(985, 980))

def shoot():
    win.grab().save("designer_shot.png")
    print("screenshot -> designer_shot.png")
    app.quit()

QTimer.singleShot(700, shoot)
app.exec()
