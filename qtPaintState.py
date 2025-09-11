from __future__ import annotations
from typing import Optional

from PyQt6 import QtCore, QtGui


class PaintState(QtCore.QObject):
    """Single source of truth for paint UI + tools."""
    brushSizeChanged = QtCore.pyqtSignal(int)
    brushColorChanged = QtCore.pyqtSignal(QtGui.QColor)
    activeToolChanged = QtCore.pyqtSignal(str)
    blurStrengthChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._brush_size = 24
        self._brush_color = QtGui.QColor("#ff0000")
        self._active_tool = "paint"  # paint|erase|blur|clone|select|diffuse
        self._blur_strength = 1.0

    def brush_size(self) -> int:
        return self._brush_size

    def brush_color(self) -> QtGui.QColor:
        return QtGui.QColor(self._brush_color)

    def active_tool(self) -> str:
        return self._active_tool

    def blur_strength(self) -> float:
        return self._blur_strength

    def set_brush_size(self, v: int):
        v = int(v)
        if v != self._brush_size:
            self._brush_size = v
            self.brushSizeChanged.emit(v)

    def set_brush_color(self, color: QtGui.QColor | str):
        c = QtGui.QColor(color) if isinstance(color, str) else QtGui.QColor(color)
        if c != self._brush_color:
            self._brush_color = c
            self.brushColorChanged.emit(QtGui.QColor(self._brush_color))

    def set_active_tool(self, name: str):
        n = (name or "").lower()
        if n != self._active_tool:
            self._active_tool = n
            self.activeToolChanged.emit(n)

    def set_blur_strength(self, s: float):
        s = float(max(0.5, min(4.0, s)))
        if s != self._blur_strength:
            self._blur_strength = s
            self.blurStrengthChanged.emit(s)