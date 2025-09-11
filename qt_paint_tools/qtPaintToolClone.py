"""
Modular Clone Tool Template for NiceThumb

This class implements a self-contained clone tool for use in PaintView.
It is designed for easy extension and integration with the main UI.

Features:
- Provides metadata (name, display flags) for toolbar and UI logic.
- Handles brush size changes and cursor preview.
- Returns a widget for tool-specific options: "Set Source" button.
- Responds to mouse events to set source and apply clone stamping.
- Uses callbacks for cursor updates and stamping, allowing decoupling from canvas internals.

Usage:
- Instantiate and register with the tool system.
- Call `on_selected` when the tool is activated, passing current size and callbacks.
- Call `on_cursor_size_changed` when the brush size changes.
- Forward mouse events to `on_mouse_event`.
- Clone tool does NOT require color palette (see returned dict from on_selected).
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import make_brush_cursor, make_circular_patch, get_composed_image, get_brush_geometry


class PaintToolClone(QtCore.QObject):
    name = "clone"
    display_brush_controls = True
    display_palette = False
    display_tool_options = True
    display_cursor_preview = True

    def __init__(self):
        super().__init__()
        self._brush_size = 24
        self._canvas = None
        self._update_cursor_cb: Optional[Callable] = None
        self._stamp_cb: Optional[Callable] = None
        self._tool_callback: Optional[Callable] = None

        self._options_widget: Optional[QtWidgets.QWidget] = None
        self._btn_set_source: Optional[QtWidgets.QPushButton] = None

        self._mode = "set_source"  # "set_source" or "clone"
        self._source_point: Optional[QtCore.QPointF] = None
        self._drag_start: Optional[QtCore.QPointF] = None
        self._dragging = False
        self._last_mouse: Optional[QtCore.QPoint] = None

    def button_name(self) -> str:
        return "Clone"

    def create_options_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._btn_set_source = QtWidgets.QPushButton("Set Source")
        self._btn_set_source.setCheckable(True)
        self._btn_set_source.setChecked(True)
        self._btn_set_source.toggled.connect(self._on_set_source_toggled)
        layout.addWidget(self._btn_set_source)

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
        **kwargs
    ) -> dict:
        self._canvas = canvas
        self._brush_size = int(brush_size)
        self._update_cursor_cb = update_cursor_cb
        self._stamp_cb = stamp_cb
        self._tool_callback = tool_callback
        self._mode = "set_source"
        self._source_point = None
        self._drag_start = None
        self._dragging = False
        self._last_mouse = None
        if self._btn_set_source:
            self._btn_set_source.setChecked(True)
            self._btn_set_source.setEnabled(True)
        self._update_cursor()
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
            "display_cursor_preview": self.display_cursor_preview,
        }

    def on_cursor_size_changed(self, size: int):
        self._brush_size = int(size)
        self._update_cursor()

    def _on_set_source_toggled(self, checked: bool):
        if checked:
            self._mode = "set_source"
            self._source_point = None
            self._drag_start = None
            self._dragging = False
            self._btn_set_source.setEnabled(True)
        else:
            # Only allow unchecking after source is set
            if self._source_point is not None:
                self._mode = "clone"
                self._btn_set_source.setEnabled(True)
            else:
                self._btn_set_source.setChecked(True)
        self._update_cursor()

    def _update_cursor(self, pos: Optional[QtCore.QPoint] = None):
        # Called on mouse move/press/size change
        if not self._update_cursor_cb or not self._canvas or not hasattr(self._canvas, "_pixmap") or self._canvas._pixmap is None:
            return
        size = max(8, self._brush_size)
        composed_img = get_composed_image(self._canvas)
        pm = None
        if self._mode == "set_source":
            # Show patch at mouse with cross
            if pos is None:
                pos = self._last_mouse
            if composed_img is not None and pos is not None:
                img_pt = self._canvas._map_widget_to_image(pos)
                circ = make_circular_patch(composed_img, img_pt, size, border_color=QtGui.QColor("black"), border_width=2)
                pm = QtGui.QPixmap.fromImage(circ)
                painter = QtGui.QPainter(pm)
                pen_cross = QtGui.QPen(QtGui.QColor(0, 0, 0), 2)
                painter.setPen(pen_cross)
                painter.drawLine(size // 2, 0, size // 2, size)
                painter.drawLine(0, size // 2, size, size // 2)
                painter.end()
            else:
                pm = make_brush_cursor(size, QtGui.QColor("black"), border_color=QtGui.QColor("black"), border_width=2, cross=True)
        elif self._mode == "clone" and self._source_point is not None:
            # Show patch at source offset by drag delta
            if pos is None:
                pos = self._last_mouse
            if composed_img is not None and pos is not None:
                img_pt = self._canvas._map_widget_to_image(pos)
                if self._dragging and self._drag_start is not None:
                    delta = QtCore.QPointF(img_pt.x() - self._drag_start.x(), img_pt.y() - self._drag_start.y())
                else:
                    delta = QtCore.QPointF(0, 0)
                sample_center = QtCore.QPointF(self._source_point.x() + delta.x(), self._source_point.y() + delta.y())
                circ = make_circular_patch(composed_img, sample_center, size, border_color=QtGui.QColor("black"), border_width=2)
                pm = QtGui.QPixmap.fromImage(circ)
            else:
                pm = make_brush_cursor(size, QtGui.QColor("black"), border_color=QtGui.QColor("black"), border_width=2)
        else:
            pm = make_brush_cursor(size, QtGui.QColor("black"), border_color=QtGui.QColor("black"), border_width=2)
        self._update_cursor_cb(pm, size // 2, size // 2)

    def on_mouse_event(
        self,
        event_type: str,
        pos: QtCore.QPoint,
        left_down: bool,
        right_down: bool
    ):
        self._last_mouse = pos
        if self._mode == "set_source":
            self._update_cursor(pos)
            if event_type == "press" and left_down:
                img_pt = self._canvas._map_widget_to_image(pos)
                if img_pt is not None:
                    self._source_point = QtCore.QPointF(img_pt)
                    self._mode = "clone"
                    if self._btn_set_source:
                        self._btn_set_source.setChecked(False)
                        self._btn_set_source.setEnabled(True)
                    self._drag_start = None
                    self._dragging = False
                    self._update_cursor(pos)
        elif self._mode == "clone" and self._source_point is not None:
            if event_type == "press" and left_down:
                img_pt = self._canvas._map_widget_to_image(pos)
                if img_pt is not None:
                    self._drag_start = QtCore.QPointF(img_pt)
                    self._dragging = True
                    self._update_cursor(pos)
                    self._stamp_at(pos)
            elif event_type == "move" and left_down and self._dragging:
                img_pt = self._canvas._map_widget_to_image(pos)
                if img_pt is not None and self._drag_start is not None:
                    self._update_cursor(pos)
                    self._stamp_at(pos)
            elif event_type == "release":
                self._dragging = False
                self._drag_start = None
                self._update_cursor(pos)

    def _stamp_at(self, pos: QtCore.QPoint):
        if self._stamp_cb is None or self._canvas is None or self._source_point is None:
            return
        scale, dia_img, rad_img, top_left = get_brush_geometry(self._canvas, pos, self._brush_size)
        if None in (scale, dia_img, rad_img, top_left):
            return
        img_pt = self._canvas._map_widget_to_image(pos)
        composed_img = get_composed_image(self._canvas)
        if composed_img is not None and img_pt is not None:
            if self._drag_start is not None:
                delta = QtCore.QPointF(img_pt.x() - self._drag_start.x(), img_pt.y() - self._drag_start.y())
            else:
                delta = QtCore.QPointF(0, 0)
            sample_center = QtCore.QPointF(self._source_point.x() + delta.x(), self._source_point.y() + delta.y())
            circ = make_circular_patch(composed_img, sample_center, dia_img)
            self._stamp_cb(circ, top_left)

    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        pass

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        return None
