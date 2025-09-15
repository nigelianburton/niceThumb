"""
Modular Mask Tool Template for NiceThumb

This class implements a self-contained mask tool for use in PaintView.
It is designed for easy extension and integration with the main UI.

Features:
- Provides metadata (name, display flags) for toolbar and UI logic.
- Handles cursor size changes via dedicated methods.
- Returns a widget for tool-specific options: "Erase" checkbox and "Clear" button.
- Responds to mouse events to update cursor and apply mask/erase.
- Uses callbacks for cursor updates and stamping, allowing decoupling from canvas internals.

Usage:
- Instantiate and register with the tool system.
- Call `on_selected` when the tool is activated, passing current size and callbacks.
- Call `on_cursor_size_changed` when the brush size changes.
- Forward mouse events to `on_mouse_event`.
- Mask tool does NOT require color palette (see returned dict from on_selected).
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import ToolPaletteWidget, PALETTE, make_brush_cursor, get_brush_geometry


class PaintToolMask(QtCore.QObject):
    name = "mask"
    display_brush_controls = True
    display_palette = False
    display_tool_options = True

    def __init__(self):
        super().__init__()
        print("[DEBUG] PaintToolMask loaded from", __file__)
        self._brush_size = 24
        self._canvas = None
        self._update_cursor_cb: Optional[Callable] = None
        self._stamp_cb: Optional[Callable] = None
        self._erase_mode = False

        self._options_widget: Optional[QtWidgets.QWidget] = None
        self._cb_erase: Optional[QtWidgets.QCheckBox] = None
        self._btn_clear: Optional[QtWidgets.QPushButton] = None
        self._tool_callback: Optional[Callable[[str, str | bool], None]] = None

    def button_name(self) -> str:
        return "Mask"

    def create_options_widget(self) -> QtWidgets.QWidget:
        print("Create options widget started")
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._cb_erase = QtWidgets.QCheckBox("Erase")
        self._cb_erase.setChecked(self._erase_mode)
        self._cb_erase.toggled.connect(self._on_erase_toggled)
        layout.addWidget(self._cb_erase)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self._btn_clear)

        layout.addStretch(1)
        self._options_widget = widget
        return widget

    def on_selected(
        self,
        canvas,
        brush_size: int,
        update_cursor_cb: Callable,
        stamp_cb: Callable,
        tool_callback: Optional[Callable[[str, str | bool], None]] = None,
        mask_active: bool = False,
    ) -> dict:
        if canvas is None:
            print("[PaintToolMask] ERROR: canvas is None in on_selected")
            return {}
        if update_cursor_cb is None:
            print("[PaintToolMask] ERROR: update_cursor_cb is None in on_selected")
            return {}
        if stamp_cb is None:
            print("[PaintToolMask] ERROR: stamp_cb is None in on_selected")
            return {}
        self._canvas = canvas
        self._brush_size = int(brush_size)
        self._update_cursor_cb = update_cursor_cb
        self._stamp_cb = stamp_cb
        self._tool_callback = tool_callback
        # Always set mask layer visible when mask tool is selected
        if hasattr(self._canvas, "set_mask_visible"):
            self._canvas.set_mask_visible(True)
        # Always set mask_active to True when mask tool is selected
        if self._tool_callback:
            self._tool_callback("mask_active", True)
        # Ensure the mask checkbox is checked when mask tool is entered
        if self._cb_erase is not None:
            self._cb_erase.setChecked(False)
        self._update_cursor()
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
        }

    def on_cursor_size_changed(self, size: int):
        if size is None:
            print("[PaintToolMask] ERROR: size is None in on_cursor_size_changed")
            return
        self._brush_size = int(size)
        self._update_cursor()

    def _on_erase_toggled(self, checked: bool):
        print("[DEBUG] _on_erase_toggled called")
        self._erase_mode = checked
        self._update_cursor()
        # Optionally notify parent about erase mode change
        # if self._tool_callback:
        #     self._tool_callback("mask_erase", checked)

    def _on_clear_clicked(self):
        self.clear_mask()

    def _update_cursor(self):
        if self._update_cursor_cb is None:
            print("[PaintToolMask] ERROR: _update_cursor_cb is None in _update_cursor")
            return
        size = max(8, self._brush_size)
        if self._erase_mode:
            pm = make_brush_cursor(
                size,
                QtCore.Qt.GlobalColor.transparent,
                border_color=QtGui.QColor(128, 128, 128, 180),
                border_width=2,
            )
        else:
            pm = make_brush_cursor(size, QtGui.QColor(255, 128, 192, 180))  # Transparent pink
        self._update_cursor_cb(pm, size // 2, size // 2)

    def on_mouse_event(
        self,
        event_type: str,
        pos: QtCore.QPoint,
        left_down: bool,
        right_down: bool,
    ):
        if event_type in ("press", "move") and left_down:
            self._stamp_at(pos)

    def _stamp_at(self, pos: QtCore.QPoint):
        if self._stamp_cb is None:
            print("[PaintToolMask] ERROR: _stamp_cb is None in _stamp_at")
            return
        if self._canvas is None:
            print("[PaintToolMask] ERROR: _canvas is None in _stamp_at")
            return
        scale, dia_img, rad_img, top_left = get_brush_geometry(
            self._canvas, pos, self._brush_size
        )
        if None in (scale, dia_img, rad_img, top_left):
            return
        circ = QtGui.QImage(
            int(dia_img), int(dia_img), QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        circ.fill(0)
        p = QtGui.QPainter(circ)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        if self._erase_mode:
            p.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.transparent))
        else:
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 255)))  # Opaque white
        p.drawEllipse(QtCore.QRectF(0, 0, dia_img, dia_img))
        p.end()
        self._stamp_cb(circ, top_left, self._erase_mode, layer="mask")

    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        pass

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        return None

    def clear_mask(self):
        """
        Clears the entire mask layer.
        """
        if (
            self._canvas
            and hasattr(self._canvas, "_mask_overlay")
            and self._canvas._mask_overlay is not None
        ):
            # Fill with transparent (fully clear) using correct blending mode
            self._canvas._mask_overlay.fill(0)  # ARGB32: 0 = fully transparent
            self._canvas.update()