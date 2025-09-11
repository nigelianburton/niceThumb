

from __future__ import annotations
from typing import Optional, List

from PyQt6 import QtCore, QtWidgets


# ---------- Shared UI widgets ----------

class BrushControlsWidget(QtWidgets.QGroupBox):
    sizeChanged = QtCore.pyqtSignal(int)

    def __init__(
        self,
        initial_size: int,
        min_size: int = 2,
        max_size: int = 92,
        preview_size: int = 92,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__("Brush", parent)
        self._size = int(initial_size)
        self._preview_size = int(preview_size)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self._preview = QtWidgets.QLabel()
        self._preview.setFixedSize(self._preview_size, self._preview_size)
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._preview, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_size), int(max_size))
        self._slider.setValue(self._size)
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)

    def _on_slider(self, v: int):
        self._size = int(v)
        self.sizeChanged.emit(self._size)

    def set_preview_pixmap(self, pm: QtGui.QPixmap):
        self._preview.setPixmap(pm)

    def set_size(self, v: int):
        self._slider.setValue(int(v))


class PaletteWidget(QtWidgets.QGroupBox):
    colorSelected = QtCore.pyqtSignal(str)

    def __init__(self, colors: List[str], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Palette", parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        for i, hex_color in enumerate(colors[:64]):
            btn = QtWidgets.QPushButton()
            btn.setFixedSize(16, 16)
            btn.setStyleSheet(f"background:{hex_color}; border:1px solid #888;")
            btn.clicked.connect(lambda _, c=hex_color: self.colorSelected.emit(c))
            grid.addWidget(btn, i // 8, i % 8)


class ToolsWidget(QtWidgets.QGroupBox):
    toolSelected = QtCore.pyqtSignal(str)

    def __init__(self, tool_names: List[str], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Tools", parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QtWidgets.QPushButton] = {}

        self.setStyleSheet("""
            QPushButton { height: 24px; }
            QPushButton:checked {
                background: #3b82f6;
                color: white;
                border: 1px solid #1e40af;
            }
        """)

        for i, name in enumerate(tool_names):
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            self._group.addButton(btn)
            self._buttons[name.lower()] = btn
            btn.clicked.connect(lambda _, n=name: self.toolSelected.emit(n))
            row, col = divmod(i, 3)  # 2 columns, 3 rows
            grid.addWidget(btn, col, row)

    @QtCore.pyqtSlot(str)
    def set_active(self, tool_name: str):
        key = (tool_name or "").lower()
        btn = self._buttons.get(key)
        if btn and not btn.isChecked():
            block = btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(block)


class ActionsWidget(QtWidgets.QGroupBox):
    saveRequested = QtCore.pyqtSignal()
    cancelRequested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Actions", parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        save_btn = QtWidgets.QPushButton("SAVE")
        cancel_btn = QtWidgets.QPushButton("CANCEL")
        save_btn.clicked.connect(self.saveRequested.emit)
        cancel_btn.clicked.connect(self.cancelRequested.emit)
        lay.addWidget(save_btn)
        lay.addWidget(cancel_btn)
        lay.addStretch(1)