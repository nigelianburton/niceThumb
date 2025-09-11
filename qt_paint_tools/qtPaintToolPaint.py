"""
Modular Paint Tool Template for NiceThumb

This class implements a self-contained paint tool for use in PaintView.
It is designed for easy extension and integration with the main UI.

Features:
- Provides metadata (name, display flags) for toolbar and UI logic.
- Handles color and cursor size changes via dedicated methods.
- Returns a widget for tool-specific options (Pick, Erase).
- Responds to mouse events to update cursor, apply paint, or pick a color.
- Uses callbacks for cursor updates and stamping, allowing decoupling from canvas internals.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import (
    ToolPaletteWidget, PALETTE, make_brush_cursor, get_brush_geometry, get_composed_image
)

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
        # Pick mode state
        self._pick_mode = False
        self._btn_pick: Optional[QtWidgets.QPushButton] = None
        self._last_mouse: Optional[QtCore.QPoint] = None

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

        # Pick button (toggles pick mode)
        self._btn_pick = QtWidgets.QPushButton("Pick")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setChecked(False)
        self._btn_pick.toggled.connect(self._on_pick_toggled)
        options_col.addWidget(self._btn_pick)

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
        # If user clicks a palette color while in pick mode, do not auto-exit pick mode.
        self._update_cursor()
        if self._tool_callback:
            self._tool_callback("color", color)

    def _on_erase_changed(self, state: int):
        self._erase = bool(state)
        # Erase does not apply to pick mode; cursor update will reflect current mode
        self._update_cursor()

    def _on_pick_toggled(self, checked: bool):
        self._pick_mode = bool(checked)
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
        self._pick_mode = False
        if self._btn_pick:
            block = self._btn_pick.blockSignals(True)
            self._btn_pick.setChecked(False)
            self._btn_pick.blockSignals(block)
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
        if self._pick_mode:
            self._update_picker_cursor(self._last_mouse)
            return
        size = max(8, self._brush_size)
        if self._erase:
            pm = make_brush_cursor(size, QtCore.Qt.GlobalColor.transparent, border_color=QtGui.QColor("#888888"), border_width=2)
        else:
            pm = make_brush_cursor(size, self._brush_color)
        self._update_cursor_cb(pm, size // 2, size // 2)

    def _update_picker_cursor(self, pos: Optional[QtCore.QPoint] = None):
        # Draw a transparent-filled circle with border sampled from composed image,
        # plus a cross inside. Border width 3 (or 1 if brush size < 10).
        size = max(8, self._brush_size)
        border_w = 1 if self._brush_size < 10 else 3
        border_color = QtGui.QColor("#888888")  # default until sampled

        try:
            composed = get_composed_image(self._canvas)
            if composed is not None and pos is not None:
                img_pt = self._canvas._map_widget_to_image(pos)
                if img_pt is not None:
                    x = max(0, min(int(img_pt.x()), composed.width() - 1))
                    y = max(0, min(int(img_pt.y()), composed.height() - 1))
                    rgba = composed.pixel(x, y)
                    border_color = QtGui.QColor.fromRgba(rgba)
        except Exception:
            pass

        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        # Ring
        pen = QtGui.QPen(border_color)
        pen.setWidth(border_w)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        # Inset by half pen width to keep stroke fully inside pixmap
        inset = border_w / 2.0
        p.drawEllipse(QtCore.QRectF(inset, inset, size - border_w, size - border_w))
        # Cross
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
        p.drawLine(size // 2, 0, size // 2, size)
        p.drawLine(0, size // 2, size, size // 2)
        p.end()
        self._update_cursor_cb(pm, size // 2, size // 2)

    def on_mouse_event(
        self,
        event_type: str,
        pos: QtCore.QPoint,
        left_down: bool,
        right_down: bool
    ):
        self._last_mouse = pos

        # Enter Pick mode with right button if mouse is over the image
        if event_type == "press" and right_down and self._canvas:
            if self._canvas._map_widget_to_image(pos) is not None:
                self._enter_pick_mode()
                self._update_picker_cursor(pos)
                return

        if self._pick_mode:
            # Update cursor color as we move over the image
            if event_type == "move":
                self._update_picker_cursor(pos)
            # Left click picks the color and exits pick mode
            if event_type == "press" and left_down:
                color = self._sample_color_at(pos)
                if color is not None:
                    self._set_brush_color(color)
                self._leave_pick_mode()
                self._update_cursor()
            return  # Do not paint while in pick mode

        # Normal paint/erase behavior
        if event_type in ("press", "move") and left_down:
            self._stamp_at(pos)

    def _enter_pick_mode(self):
        self._pick_mode = True
        if self._btn_pick:
            block = self._btn_pick.blockSignals(True)
            self._btn_pick.setChecked(True)
            self._btn_pick.blockSignals(block)

    def _leave_pick_mode(self):
        self._pick_mode = False
        if self._btn_pick:
            block = self._btn_pick.blockSignals(True)
            self._btn_pick.setChecked(False)
            self._btn_pick.blockSignals(block)

    def _sample_color_at(self, pos: QtCore.QPoint) -> Optional[QtGui.QColor]:
        try:
            composed = get_composed_image(self._canvas)
            if composed is None:
                return None
            img_pt = self._canvas._map_widget_to_image(pos)
            if img_pt is None:
                return None
            x = max(0, min(int(img_pt.x()), composed.width() - 1))
            y = max(0, min(int(img_pt.y()), composed.height() - 1))
            rgba = composed.pixel(x, y)
            return QtGui.QColor.fromRgba(rgba)
        except Exception:
            return None

    def _set_brush_color(self, color: QtGui.QColor):
        self._brush_color = QtGui.QColor(color)
        if self._palette_widget is not None:
            self._palette_widget.set_active_color(self._brush_color)
        if self._tool_callback:
            self._tool_callback("color", self._brush_color)

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