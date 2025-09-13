"""
Modular Paint Tool Template for NiceThumb

Enhancements:
- Added a "Size" sub‑mode inside the Paint tool that embeds the Page Size plugin (PaintToolPageSize).
- Sub‑modes now: Paint (normal), Pick (temporary mode via button/right click), Size (shows page size UI).
- When in Size mode: painting / picking are disabled and the PageSize tool's options widget is shown.
- Page size changes emit a "page_size" event through tool_callback: ("page_size", (height, width)).
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import (
    ToolPaletteWidget, PALETTE, make_brush_cursor, get_brush_geometry, get_composed_image
)
from qt_paint_tools.qtPaintToolPageSize import PaintToolPageSize  # <-- NEW

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
        self._tool_callback: Optional[Callable[[str, QtGui.QColor | tuple], None]] = None

        self._erase = False
        self._pick_mode = False            # Temporary color-pick sub-mode
        self._sub_mode = "paint"           # "paint" | "size"
        self._last_mouse: Optional[QtCore.QPoint] = None

        # UI widgets
        self._options_widget: Optional[QtWidgets.QWidget] = None
        self._palette_widget: Optional[ToolPaletteWidget] = None
        self._btn_pick: Optional[QtWidgets.QPushButton] = None
        self._btn_mode_paint: Optional[QtWidgets.QPushButton] = None
        self._btn_mode_size: Optional[QtWidgets.QPushButton] = None
        self._erase_checkbox: Optional[QtWidgets.QCheckBox] = None
        self._stack_modes: Optional[QtWidgets.QStackedWidget] = None
        self._page_size_container: Optional[QtWidgets.QWidget] = None

        # Embedded Page Size tool
        self._page_size_tool = PaintToolPageSize()
        self._page_size_tool.sizeChanged.connect(self._on_page_size_changed)

    def button_name(self) -> str:
        return "Paint"

    # -------- UI Creation --------
    def create_options_widget(self) -> Optional[QtWidgets.QWidget]:
        if self._options_widget is not None:
            return self._options_widget

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Sub-mode buttons row (Paint / Size)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(4)

        self._btn_mode_paint = QtWidgets.QPushButton("Paint")
        self._btn_mode_paint.setCheckable(True)
        self._btn_mode_size = QtWidgets.QPushButton("Size")
        self._btn_mode_size.setCheckable(True)

        mode_group = QtWidgets.QButtonGroup(root)
        mode_group.setExclusive(True)
        mode_group.addButton(self._btn_mode_paint)
        mode_group.addButton(self._btn_mode_size)

        self._btn_mode_paint.setChecked(True)
        self._btn_mode_paint.toggled.connect(lambda on: on and self._set_sub_mode("paint"))
        self._btn_mode_size.toggled.connect(lambda on: on and self._set_sub_mode("size"))

        mode_row.addWidget(self._btn_mode_paint)
        mode_row.addWidget(self._btn_mode_size)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # Stacked content: index 0 = paint controls, index 1 = page size tool UI
        self._stack_modes = QtWidgets.QStackedWidget()
        layout.addWidget(self._stack_modes, 1)

        # ---- Page 0: Paint Controls ----
        page_paint = QtWidgets.QWidget()
        paint_lay = QtWidgets.QHBoxLayout(page_paint)
        paint_lay.setContentsMargins(0, 0, 0, 0)
        paint_lay.setSpacing(12)

        # Palette column
        palette_col = QtWidgets.QVBoxLayout()
        palette_col.setContentsMargins(0, 0, 0, 0)
        palette_col.setSpacing(0)
        self._palette_widget = ToolPaletteWidget(PALETTE, initial_color=self._brush_color)
        self._palette_widget.colorSelected.connect(self._on_palette_color_selected)
        palette_col.addWidget(self._palette_widget, 1)

        # Options column (Pick, Erase)
        options_col = QtWidgets.QVBoxLayout()
        options_col.setContentsMargins(0, 0, 0, 0)
        options_col.setSpacing(6)

        self._btn_pick = QtWidgets.QPushButton("Pick")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setChecked(False)
        self._btn_pick.toggled.connect(self._on_pick_toggled)
        options_col.addWidget(self._btn_pick)

        self._erase_checkbox = QtWidgets.QCheckBox("Erase")
        self._erase_checkbox.setChecked(self._erase)
        self._erase_checkbox.stateChanged.connect(self._on_erase_changed)
        options_col.addWidget(self._erase_checkbox)

        options_col.addStretch(1)

        paint_lay.addLayout(palette_col, 1)
        paint_lay.addLayout(options_col, 2)

        self._stack_modes.addWidget(page_paint)

        # ---- Page 1: Page Size Tool ----
        self._page_size_container = QtWidgets.QWidget()
        ps_lay = QtWidgets.QVBoxLayout(self._page_size_container)
        ps_lay.setContentsMargins(0, 0, 0, 0)
        ps_lay.setSpacing(0)
        # Acquire page size tool widget
        ps_widget = self._page_size_tool.create_options_widget()
        ps_lay.addWidget(ps_widget)
        self._stack_modes.addWidget(self._page_size_container)

        self._options_widget = root
        return root

    # -------- Sub-modes --------
    def _set_sub_mode(self, mode: str):
        if mode == self._sub_mode:
            return
        
        # If leaving 'size' mode, commit the changes.
        if self._sub_mode == "size":
            self._page_size_tool.on_deselected()

        self._sub_mode = mode
        if self._stack_modes:
            if mode == "paint":
                self._stack_modes.setCurrentIndex(0)
            elif mode == "size":
                self._stack_modes.setCurrentIndex(1)
                # Activate embedded page size tool (no brush operations)
                if self._canvas:
                    self._page_size_tool.on_selected(self._canvas, tool_callback=self._tool_callback)
        # Cancel pick mode if switching to size
        if mode == "size" and self._pick_mode:
            self._leave_pick_mode()
        self._update_cursor()

    # -------- Event callbacks (Page Size) --------
    def _on_page_size_changed(self, h: int, w: int):
        if self._tool_callback:
            self._tool_callback("page_size", (h, w))

    # -------- Color / Erase / Pick handlers --------
    def _on_palette_color_selected(self, color: QtGui.QColor):
        self._brush_color = color
        self._update_cursor()
        if self._tool_callback:
            self._tool_callback("color", color)

    def _on_erase_changed(self, state: int):
        self._erase = bool(state)
        self._update_cursor()

    def _on_pick_toggled(self, checked: bool):
        # Only meaningful in paint sub-mode
        if self._sub_mode != "paint":
            # Revert pick toggle if not in paint mode
            if self._btn_pick and self._btn_pick.isChecked():
                block = self._btn_pick.blockSignals(True)
                self._btn_pick.setChecked(False)
                self._btn_pick.blockSignals(block)
            return
        self._pick_mode = bool(checked)
        self._update_cursor()

    # -------- Selection lifecycle --------
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

        # Reset when tool activated
        self._pick_mode = False
        self._sub_mode = "paint"
        if self._btn_pick:
            block = self._btn_pick.blockSignals(True)
            self._btn_pick.setChecked(False)
            self._btn_pick.blockSignals(block)
        if self._btn_mode_paint:
            block2 = self._btn_mode_paint.blockSignals(True)
            self._btn_mode_paint.setChecked(True)
            self._btn_mode_paint.blockSignals(block2)
        if self._stack_modes:
            self._stack_modes.setCurrentIndex(0)

        # Initialize page size tool if user switches later
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
        if self._palette_widget is not None:
            self._palette_widget.set_active_color(self._brush_color)
        self._update_cursor()

    def on_cursor_size_changed(self, size: int):
        self._brush_size = int(size)
        self._update_cursor()

    # -------- Cursor logic --------
    def _update_cursor(self):
        if self._update_cursor_cb is None:
            return
        if self._sub_mode == "size":
            # Use default arrow in size mode
            self._canvas.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            return
        if self._pick_mode:
            self._update_picker_cursor(self._last_mouse)
            return
        size = max(8, self._brush_size)
        if self._erase:
            pm = make_brush_cursor(size, QtCore.Qt.GlobalColor.transparent,
                                   border_color=QtGui.QColor("#888888"), border_width=2)
        else:
            pm = make_brush_cursor(size, self._brush_color)
        self._update_cursor_cb(pm, size // 2, size // 2)

    def _update_picker_cursor(self, pos: Optional[QtCore.QPoint] = None):
        # Only valid in paint mode
        if self._sub_mode != "paint":
            return
        size = max(8, self._brush_size)
        border_w = 1 if self._brush_size < 10 else 3
        border_color = QtGui.QColor("#888888")
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
        pen = QtGui.QPen(border_color); pen.setWidth(border_w)
        p.setPen(pen); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        inset = border_w / 2.0
        p.drawEllipse(QtCore.QRectF(inset, inset, size - border_w, size - border_w))
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
        p.drawLine(size // 2, 0, size // 2, size)
        p.drawLine(0, size // 2, size, size // 2)
        p.end()
        self._update_cursor_cb(pm, size // 2, size // 2)

    # -------- Mouse events --------
    def on_mouse_event(self, event_type: str, pos: QtCore.QPoint, left_down: bool, right_down: bool):
        self._last_mouse = pos

        # Size mode: ignore painting & picking
        if self._sub_mode == "size":
            return

        # Enter pick mode via right-click over image
        if event_type == "press" and right_down and self._canvas:
            if self._canvas._map_widget_to_image(pos) is not None and self._sub_mode == "paint":
                self._enter_pick_mode()
                self._update_picker_cursor(pos)
                return

        if self._pick_mode:
            if event_type == "move":
                self._update_picker_cursor(pos)
            if event_type == "press" and left_down:
                color = self._sample_color_at(pos)
                if color is not None:
                    self._set_brush_color(color)
                self._leave_pick_mode()
                self._update_cursor()
            return

        if event_type in ("press", "move") and left_down:
            self._stamp_at(pos)

    # -------- Pick helpers --------
    def _enter_pick_mode(self):
        if self._btn_pick:
            block = self._btn_pick.blockSignals(True)
            self._btn_pick.setChecked(True)
            self._btn_pick.blockSignals(block)
        self._pick_mode = True

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

    # -------- Painting --------
    def _stamp_at(self, pos: QtCore.QPoint):
        if self._stamp_cb is None or self._canvas is None or pos is None:
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
            erase_color = QtGui.QColor(0, 0, 0, 255)
            p.setBrush(erase_color)
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