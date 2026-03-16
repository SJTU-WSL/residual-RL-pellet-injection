"""Theme helpers for the Qt dashboard."""
from __future__ import annotations

import pyqtgraph as pg


APP_STYLESHEET = """
QMainWindow {
    background-color: #0e1116;
}
QWidget {
    background-color: #0e1116;
    color: #d8dee9;
    font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2a2f3a;
    border-radius: 12px;
    margin-top: 14px;
    padding: 12px;
    background-color: #151a22;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #8f9cb3;
}
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
    background-color: #11161d;
    border: 1px solid #2d3340;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #2f80ed;
}
QPushButton {
    background-color: #1f2733;
    border: 1px solid #364154;
    border-radius: 10px;
    padding: 8px 12px;
    color: #ecf2ff;
    font-weight: 600;
}
QPushButton:hover {
    border-color: #50a3ff;
}
QPushButton:pressed {
    background-color: #17202b;
}
QPushButton:disabled {
    color: #6b7280;
    border-color: #222831;
}
QPushButton#primaryButton {
    background-color: #2268d8;
    border-color: #3a82f6;
}
QPushButton#dangerButton {
    background-color: #6a2435;
    border-color: #8c3950;
}
QLabel[class="caption"] {
    color: #7f8ba3;
}
QLabel[class="metricValue"] {
    color: #f4f7fb;
    font-size: 16px;
    font-weight: 700;
}
QScrollBar:vertical {
    width: 12px;
    background: #11161d;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2c3644;
    border-radius: 6px;
    min-height: 24px;
}
"""


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "#11161d")
    pg.setConfigOption("foreground", "#d8dee9")
