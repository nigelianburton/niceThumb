"""
Modular Blur Tool Template for NiceThumb

This class implements a self-contained blur tool for use in PaintView.
It is designed for easy extension and integration with the main UI.

Features:
- Provides metadata (name, display flags) for toolbar and UI logic.
- Handles brush size and blur strength changes via dedicated methods.
- Returns a widget for tool-specific options: blur strength slider.
- Responds to mouse events to update cursor and apply blur.
- Uses callbacks for cursor updates and stamping, allowing decoupling from canvas internals.
- All image processing is done internally using Qt (no external helpers).

Usage:
- Instantiate and register with the tool system.
- Call `on_selected` when the tool is activated, passing current size, strength, and callbacks.
- Call `on_cursor_size_changed` and `on_blur_strength_changed` when those values change.
- Forward mouse events to `on_mouse_event`.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import (
    ToolPaletteWidget, PALETTE, make_brush_cursor, make_circular_patch,
    get_composed_image, get_brush_geometry, blur_qimage_gaussian
)

import numpy as np  # still needed for cursor preview circle masking

class PaintToolBlur(QtCore.QObject):
    name = "blur"
    display_brush_controls = True
    display_palette = False
    display_tool_options = True
    display_cursor_preview = True

    def __init__(self):
        super().__init__()
        print("[DEBUG] PaintToolBlur loaded from", __file__)
        self._brush_size = 24
        self._blur_strength = 1.0
        self._canvas = None
        self._update_cursor_cb: Optional[Callable] = None
        self._stamp_cb: Optional[Callable] = None
        self._tool_callback: Optional[Callable[[str, str | bool], None]] = None

        self._options_widget: Optional[QtWidgets.QWidget] = None
        self._slider: Optional[QtWidgets.QSlider] = None
        self._lbl_val: Optional[QtWidgets.QLabel] = None

    def button_name(self) -> str:
        return "Blur"

    def create_options_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Blur"))
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(1, 20)
        self._slider.setSingleStep(1)
        self._slider.setValue(max(1, min(20, int(self._blur_strength))))
        self._slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self._slider, 1)

        self._lbl_val = QtWidgets.QLabel(f"{self._blur_strength:.2f}")
        self._lbl_val.setMinimumWidth(36)
        layout.addWidget(self._lbl_val)
        layout.addStretch(1)
        self._options_widget = widget
        return widget

    def on_selected(
        self,
        canvas,
        brush_size: int,
        blur_strength: float,
        update_cursor_cb: Callable,
        stamp_cb: Callable,
        tool_callback: Optional[Callable[[str, str | bool], None]] = None
    ) -> dict:
        self._canvas = canvas
        self._brush_size = int(brush_size)
        self._blur_strength = float(blur_strength)
        self._update_cursor_cb = update_cursor_cb
        self._stamp_cb = stamp_cb
        self._tool_callback = tool_callback
        self._update_cursor()
        if self._slider:
            self._slider.setValue(int(self._blur_strength))
        if self._lbl_val:
            self._lbl_val.setText(f"{self._blur_strength:.2f}")
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
            "display_cursor_preview": self.display_cursor_preview,
        }

    def on_cursor_size_changed(self, size: int):
        self._brush_size = int(size)
        self._update_cursor()

    def on_blur_strength_changed(self, strength: float):
        self._blur_strength = float(strength)
        if self._lbl_val:
            self._lbl_val.setText(f"{self._blur_strength:.2f}")
        self._update_cursor()

    def _on_slider_change(self, value: int):
        self._blur_strength = float(value)
        if self._lbl_val:
            self._lbl_val.setText(f"{self._blur_strength:.2f}")
        if self._tool_callback:
            self._tool_callback("blur_strength", self._blur_strength)
        self._update_cursor()

    def _update_cursor(self):
        if not self._update_cursor_cb:
            return
        size = max(8, self._brush_size)
        pm = make_brush_cursor(size, QtGui.QColor("yellow"))
        self._update_cursor_cb(pm, size // 2, size // 2)

    def on_mouse_event(
        self,
        event_type: str,
        pos: QtCore.QPoint,
        left_down: bool,
        right_down: bool
    ):
        if event_type == "move":
            self.update_cursor_with_blur(pos)
        if event_type in ("press", "move") and left_down:
            self._stamp_at(pos)

    def _stamp_at(self, pos: QtCore.QPoint):
        if not (self._stamp_cb and self._canvas):
            return
        geom = get_brush_geometry(self._canvas, pos, self._brush_size)
        scale, dia_img, rad_img, top_left = geom
        if None in geom:
            return
        composed_img = get_composed_image(self._canvas)
        img_pt = self._canvas._map_widget_to_image(pos)
        if composed_img is None or img_pt is None:
            return
        patch = composed_img.copy(
            int(img_pt.x() - rad_img),
            int(img_pt.y() - rad_img),
            int(dia_img),
            int(dia_img)
        )
        blurred = blur_qimage_gaussian(patch, self._blur_strength)
        circ = make_circular_patch(blurred, QtCore.QPointF(blurred.width()/2, blurred.height()/2), dia_img)
        self._stamp_cb(circ, top_left)

    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        pass

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        return None

    def get_cursor_sprite(self) -> QtGui.QPixmap:
        size = max(8, self._brush_size)
        return make_brush_cursor(size, QtGui.QColor("yellow"))

    def update_cursor_with_blur(self, pos: QtCore.QPoint):
        if not (self._update_cursor_cb and self._canvas and pos):
            return
        geom = get_brush_geometry(self._canvas, pos, self._brush_size)
        scale, dia_img, rad_img, _ = geom
        composed_img = get_composed_image(self._canvas)
        img_pt = self._canvas._map_widget_to_image(pos)
        if composed_img is None or img_pt is None or None in geom:
            return
        # Sample in image space (dia_img) for correct, non-zoomed content
        patch = composed_img.copy(
            int(img_pt.x() - rad_img),
            int(img_pt.y() - rad_img),
            int(dia_img),
            int(dia_img)
        )
        blurred = blur_qimage_gaussian(patch, self._blur_strength)
        # Build circular image at image-space diameter, then scale to display diameter for cursor
        display_diam = max(8, self._brush_size)
        circ_img = make_circular_patch(
            blurred,
            QtCore.QPointF(blurred.width() / 2, blurred.height() / 2),
            dia_img,
            border_color=QtGui.QColor("black"),
            border_width=2
        )
        pm = QtGui.QPixmap.fromImage(
            circ_img.scaled(display_diam, display_diam,
                            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation)
        )
        self._update_cursor_cb(pm, display_diam // 2, display_diam // 2)