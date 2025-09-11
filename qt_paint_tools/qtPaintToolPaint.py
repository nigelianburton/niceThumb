"""
Modular Paint Tool Template for NiceThumb

This class implements a self-contained paint tool for use in PaintView.
It is designed for easy extension and integration with the main UI.

Features:
- Provides metadata (name, display flags) for toolbar and UI logic.
- Handles color and cursor size changes via dedicated methods.
- Returns a widget for tool-specific options (if any).
- Responds to mouse events to update cursor and apply paint.
- Uses callbacks for cursor updates and stamping, allowing decoupling from canvas internals.

Usage:
- Instantiate and register with the tool system.
- Call `on_selected` when the tool is activated, passing current color, size, and callbacks.
- Call `on_color_changed` and `on_cursor_size_changed` when those values change.
- Forward mouse events to `on_mouse_event`.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import ToolPaletteWidget, PALETTE, make_brush_cursor, get_brush_geometry

class PaintToolPaint(QtCore.QObject):
    name = "paint"
    display_brush_controls = True
    display_palette = False
    display_tool_options = True

    def __init__(self):
        super().__init__()
        self._brush_color = QtGui.QColor("#ff0000")
        self._brush_size = 24
        self._canvas = None
        self._update_cursor_cb: Optional[Callable] = None
        self._stamp_cb: Optional[Callable] = None
        self._tool_callback: Optional[Callable[[str, QtGui.QColor], None]] = None
        self._erase = False
        self._options_widget: Optional[QtWidgets.QWidget] = None
        self._palette_widget: Optional[ToolPaletteWidget] = None

    def button_name(self) -> str:
        return "Paint"

    def create_options_widget(self) -> Optional[QtWidgets.QWidget]:
        if self._options_widget is not None:
            return self._options_widget
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(12)

        # Palette column
        palette_col = QtWidgets.QVBoxLayout()
        palette_col.setContentsMargins(0, 0, 0, 0)
        palette_col.setSpacing(0)
        self._palette_widget = ToolPaletteWidget(PALETTE, initial_color=self._brush_color)
        self._palette_widget.colorSelected.connect(self._on_palette_color_selected)
        palette_col.addWidget(self._palette_widget, 1)  # stretch vertically

        # Options column
        options_col = QtWidgets.QVBoxLayout()
        options_col.setContentsMargins(0, 0, 0, 0)
        options_col.setSpacing(6)
        erase_checkbox = QtWidgets.QCheckBox("Erase")
        erase_checkbox.setChecked(self._erase)
        erase_checkbox.stateChanged.connect(self._on_erase_changed)
        options_col.addWidget(erase_checkbox)
        options_col.addStretch(1)

        main_layout.addLayout(palette_col, 1)
        main_layout.addLayout(options_col, 2)
        self._options_widget = widget
        return widget

    def _on_palette_color_selected(self, color: QtGui.QColor):
        self._brush_color = color
        self._update_cursor()
        if self._tool_callback:
            self._tool_callback("color", color)

    def _on_erase_changed(self, state: int):
        self._erase = bool(state)
        self._update_cursor()

    def on_selected(
        self,
        canvas,
        brush_color: QtGui.QColor,
        brush_size: int,
        update_cursor_cb: Callable,
        stamp_cb: Callable,
        tool_callback: Optional[Callable[[str, QtGui.QColor], None]] = None
    ) -> dict:
        self._canvas = canvas
        self._brush_color = QtGui.QColor(brush_color)
        self._brush_size = int(brush_size)
        self._update_cursor_cb = update_cursor_cb
        self._stamp_cb = stamp_cb
        self._tool_callback = tool_callback
        self._update_cursor()
        if self._palette_widget is not None:
            self._palette_widget.set_active_color(self._brush_color)
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
        }

    def on_color_changed(self, color: QtGui.QColor):
        self._brush_color = QtGui.QColor(color)
        self._update_cursor()
        if self._palette_widget is not None:
            self._palette_widget.set_active_color(self._brush_color)

    def on_cursor_size_changed(self, size: int):
        self._brush_size = int(size)
        self._update_cursor()

    def _update_cursor(self):
        if self._update_cursor_cb is None:
            return
        size = max(8, self._brush_size)
        if self._erase:
            pm = make_brush_cursor(size, QtCore.Qt.GlobalColor.transparent, border_color=QtGui.QColor("#888888"), border_width=2)
        else:
            pm = make_brush_cursor(size, self._brush_color)
        self._update_cursor_cb(pm, size // 2, size // 2)

    def on_mouse_event(
        self,
        event_type: str,
        pos: QtCore.QPoint,
        left_down: bool,
        right_down: bool
    ):
        if event_type in ("press", "move") and left_down:
            self._stamp_at(pos)

    def _stamp_at(self, pos: QtCore.QPoint):
        if self._stamp_cb is None:
            return
        if self._canvas is None:
            return
        if pos is None:
            return
        scale, dia_img, rad_img, top_left = get_brush_geometry(self._canvas, pos, self._brush_size)
        if None in (scale, dia_img, rad_img, top_left):
            return
        circ = QtGui.QImage(int(dia_img), int(dia_img), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        circ.fill(0)
        p = QtGui.QPainter(circ)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        if self._erase:
            # Draw a solid opaque ellipse (alpha=255) on a transparent background
            erase_color = QtGui.QColor(0, 0, 0, 255)
            p.setBrush(erase_color)
            p.drawEllipse(QtCore.QRectF(0, 0, dia_img, dia_img))
        else:
            c = QtGui.QColor(self._brush_color); c.setAlpha(255)
            p.setBrush(QtGui.QBrush(c))
            p.drawEllipse(QtCore.QRectF(0, 0, dia_img, dia_img))
        p.end()
        self._stamp_cb(circ, top_left, self._erase)

    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        pass

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        return None