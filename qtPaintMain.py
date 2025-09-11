"""
NiceThumb PaintView (Modular, Tool-Driven)

Drop-in replacement for qtPaint.py. Integrates modular tool classes (Paint, Mask, Blur, Clone, Diffuse).
UI, toolbar, canvas, and tool switching logic are preserved. All helper methods are self-contained.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Callable, Any
import time  # needed by next_edit_filename

from PyQt6 import QtCore, QtGui, QtWidgets

# Import modular tools
from qt_paint_tools.qtPaintToolPaint import PaintToolPaint
from qt_paint_tools.qtPaintToolMask import PaintToolMask
from qt_paint_tools.qtPaintToolBlur import PaintToolBlur
from qt_paint_tools.qtPaintToolClone import PaintToolClone
from qt_paint_tools.qtPaintToolD import PaintToolDiffuse
from qt_paint_tools.qtPaintToolSelection import PaintToolSelection

from qtPaintWidgets import BrushControlsWidget, PaletteWidget, ToolsWidget, ActionsWidget
from qtPaintCanvas import PaintCanvas

# --- Self-contained helpers (copied from qtPaint.py) ---
def compose_images(base: QtGui.QImage, overlay: Optional[QtGui.QImage]) -> QtGui.QImage:
    if overlay is None or overlay.isNull():
        return base.copy()
    out = QtGui.QImage(base.size(), base.format())
    out.fill(0)
    painter = QtGui.QPainter(out)
    painter.drawImage(0, 0, base)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.drawImage(0, 0, overlay)
    painter.end()
    return out

def next_edit_filename(src: Path) -> Path:
    stem, ext = src.stem, src.suffix
    parent = src.parent
    for i in range(1, 1000):
        candidate = parent / f"{stem}_edit{i}{ext}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}_edit{int(time.time())}{ext}"

def draw_brush_preview(sprite: QtGui.QPixmap, size: int, color: QtGui.QColor) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.fillRect(pm.rect(), QtGui.QColor("#ffffff"))
    pen = QtGui.QPen(QtGui.QColor("#444444"))
    pen.setWidth(1)
    painter.setPen(pen)
    painter.setBrush(color)
    r = max(2, min(sprite.width(), size - 6))
    cx = cy = size // 2
    painter.drawEllipse(QtCore.QPointF(cx, cy), r / 2.0, r / 2.0)
    painter.end()
    return pm

# --- Constants ---
PAINT_TOOLBAR_HEIGHT_PX = 192
TOOL_OPTIONS_MIN_W = int(320 * 1.6)
BRUSH_PREVIEW_SIZE_PX = 92
BRUSH_MIN = 2
BRUSH_MAX = 92
BRUSH_WHEEL_STEP = 4  # Mouse wheel step size (px) when over the image
PALETTE: List[str] = [
    "#FF0000", "#00FF0000", "#0000FF", "#FFFF00", "#FFFFFF", "#000000", "#808080", "#FFA500",
    "#800080", "#A52A2A", "#008080", "#FFC0CB", "#00FFFF", "#FF00FF", "#40E0D0", "#E6E6FA",
    "#FDFD96", "#FF1493", "#DB7093", "#FFB6C1", "#F08080", "#CD5C5C", "#E0B0FF", "#B4E4B4",
    "#FFB347", "#FFD1DC", "#AEC6CF", "#C3B1E1", "#77DD77", "#836953", "#FAF0E6", "#FFFACD",
    "#EEE8AA", "#F0E68C", "#FFD700", "#DAA520", "#B8860B", "#D2B48C", "#A0522D", "#8B4513",
    "#D2691E", "#CD853F", "#BC8F8F", "#F4A460", "#800000", "#5C4033", "#F0D5BE", "#E0AC69",
    "#C68642", "#8D5524", "#66381A", "#FBEADC", "#F5CBA7", "#E5A07A", "#DDBB99", "#B58A58",
    "#A16B36", "#6F441D", "#4D2A13", "#E0C7B3", "#D5A986", "#C58A6A"
]

DETAILED_LOG = True  # Enable detailed logging

class PaintView(QtWidgets.QWidget):
    saved = QtCore.pyqtSignal(Path)
    canceled = QtCore.pyqtSignal()
    diffused = QtCore.pyqtSignal(Path)

    def __init__(self, root_path: Path, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.root_path = Path(root_path)
        self.current_path: Optional[Path] = None
        self._orig_pixmap: Optional[QtGui.QPixmap] = None
        self._brush_size = 24
        self._brush_color = QtGui.QColor("#ff0000")
        self._active_tool: str = "paint"
        self._active_tool_obj: Optional[Any] = None
        self._tool_callback = self._on_tool_event
        self._build_ui()
        self._tool_map = {
            "paint": PaintToolPaint(),
            "mask": PaintToolMask(),
            "blur": PaintToolBlur(),
            "clone": PaintToolClone(),
            "select": PaintToolSelection(),
            "diffuse": PaintToolDiffuse(),
        }
        # Force default Paint tool active on creation
        self.tools_widget.set_active("Paint")
        self._on_tool_selected("Paint")

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QtWidgets.QWidget()
        toolbar.setFixedHeight(PAINT_TOOLBAR_HEIGHT_PX)
        tlay = QtWidgets.QHBoxLayout(toolbar)
        tlay.setContentsMargins(6, 6, 6, 6)
        tlay.setSpacing(12)

        self.brush_controls = BrushControlsWidget(
            initial_size=self._brush_size,
            min_size=BRUSH_MIN,
            max_size=BRUSH_MAX,
            preview_size=BRUSH_PREVIEW_SIZE_PX,
        )
        self.brush_controls.sizeChanged.connect(self._on_brush_size_changed)

        self.palette_widget = PaletteWidget(PALETTE)
        self.palette_widget.colorSelected.connect(self._on_brush_color_changed)

        self.tools_widget = ToolsWidget(["Paint", "Mask", "Blur", "Clone", "Select", "Diffuse"])
        self.tools_widget.toolSelected.connect(self._on_tool_selected)

        self.tool_options_box = QtWidgets.QGroupBox("Tool Options")
        self.tool_options_box.setMinimumWidth(TOOL_OPTIONS_MIN_W)
        to_lay = QtWidgets.QVBoxLayout(self.tool_options_box)
        to_lay.setContentsMargins(2, 2, 2, 2)
        to_lay.setSpacing(6)
        self.tool_options_stack = QtWidgets.QStackedWidget()
        to_lay.addWidget(self.tool_options_stack)

        self.page_empty = QtWidgets.QWidget()
        self.tool_options_stack.addWidget(self.page_empty)

        self.actions_widget = ActionsWidget()
        self.actions_widget.saveRequested.connect(self._save_current)
        self.actions_widget.cancelRequested.connect(self._cancel_edit)

        tlay.addWidget(self.tools_widget)
        tlay.addWidget(self.brush_controls)
        tlay.addWidget(self.tool_options_box, 1)
        tlay.addWidget(self.palette_widget)
        tlay.addWidget(self.actions_widget)
        tlay.addStretch(1)

        self.image_canvas = PaintCanvas()
        self.image_canvas.setMouseTracking(True)
        self.image_canvas.mouseEventCallback = self._on_canvas_mouse_event
        self.image_canvas.installEventFilter(self)

        root.addWidget(toolbar, 0)
        root.addWidget(self.image_canvas, 1)

        self._update_brush_preview()

    def set_path(self, path: Optional[Path]):
        self.current_path = path if (path and path.exists() and path.is_file()) else None
        self._orig_pixmap = None
        if not self.current_path:
            self.image_canvas.set_pixmap(None)
            return
        pix = QtGui.QPixmap(str(self.current_path))
        if pix.isNull():
            self.image_canvas.set_pixmap(None)
            return
        self._orig_pixmap = pix
        self.image_canvas.set_pixmap(self._orig_pixmap)
        # Force default Paint tool each time edit mode (image) is entered
        self.tools_widget.set_active("Paint")
        self._on_tool_selected("Paint")

    def _on_tool_selected(self, name: str):
        # Auto-commit selection sprite if leaving selection tool
        if self._active_tool_obj and hasattr(self._active_tool_obj, "on_deselected"):
            try:
                self._active_tool_obj.on_deselected()
            except Exception:
                pass
        n = (name or "").lower()
        self._active_tool = n
        self._active_tool_obj = self._tool_map.get(n)
        # Setup tool options UI
        options_widget = self._active_tool_obj.create_options_widget() if self._active_tool_obj else self.page_empty
        if options_widget is None:
            options_widget = self.page_empty  # <-- Ensure a valid QWidget
        idx = self.tool_options_stack.indexOf(options_widget)
        if idx < 0:
            self.tool_options_stack.addWidget(options_widget)
            idx = self.tool_options_stack.indexOf(options_widget)
        self.tool_options_stack.setCurrentIndex(idx)
        self.tool_options_stack.currentWidget().show()
        # Show/hide palette/brush controls
        self.brush_controls.setVisible(getattr(self._active_tool_obj, "display_brush_controls", True))
        self.palette_widget.setVisible(getattr(self._active_tool_obj, "display_palette", True))
        # Activate tool
        self._activate_tool()

    def _activate_tool(self):
        if not self._active_tool_obj:
            print("[PaintView] ERROR: _active_tool_obj is None in _activate_tool")
            return

        params = {}
        if self._active_tool == "paint":
            params = {
                "canvas": self.image_canvas,
                "brush_color": self._brush_color,
                "brush_size": self._brush_size,
                "update_cursor_cb": self._set_cursor_sprite,
                "stamp_cb": self._on_stamp_request,
                "tool_callback": self._tool_callback,
            }
        elif self._active_tool == "blur":
            params = {
                "canvas": self.image_canvas,
                "brush_size": self._brush_size,
                "blur_strength": getattr(self, "_blur_strength", 1.0),
                "update_cursor_cb": self._set_cursor_sprite,
                "stamp_cb": self._on_stamp_request,
                "tool_callback": self._tool_callback,
            }
        elif self._active_tool == "mask":
            params = {
                "canvas": self.image_canvas,
                "brush_size": self._brush_size,
                "update_cursor_cb": self._set_cursor_sprite,
                "stamp_cb": self._on_stamp_request,
                "tool_callback": self._tool_callback,
            }
        elif self._active_tool == "clone":
            params = {
                "canvas": self.image_canvas,
                "brush_size": self._brush_size,
                "update_cursor_cb": self._set_cursor_sprite,
                "stamp_cb": self._on_stamp_request,
                "tool_callback": self._tool_callback,
            }
        elif self._active_tool == "diffuse":
            params = {
                "canvas": self.image_canvas,
                "root_path": self.root_path,
                "current_path": self.current_path,
                "compose_image_provider": self._compose_current_image,
                "mask_image_provider": self._get_mask_image,
                "tool_callback": self._tool_callback,
            }
        elif self._active_tool == "select":
            params = {
                "canvas": self.image_canvas,
                "update_cursor_cb": self._set_cursor_sprite,
                "stamp_cb": self._on_stamp_request,
                "tool_callback": self._tool_callback,
            }

        print(f"[PaintView] Activating tool: {self._active_tool}")
        for k, v in params.items():
            print(f"[PaintView] Param {k}: {type(v)} -> {v}")

        # Wire active tool into canvas so paintOverlay is invoked.
        try:
            self.image_canvas._tool = self._active_tool_obj
            if self._active_tool == "select":
                # Expose selection state for action handlers (fill/blur/etc.)
                self.image_canvas._sel_state = self._active_tool_obj._state
            else:
                # Optional: clear stale selection reference when leaving selection tool
                if hasattr(self.image_canvas, "_sel_state") and self._active_tool != "select":
                    self.image_canvas._sel_state = None
        except Exception as e:
            print(f"[PaintView] Warning: failed to attach tool to canvas: {e}")

        self._active_tool_obj.on_selected(**params)
        self._update_brush_preview()

    def _on_brush_size_changed(self, size: int):
        self._brush_size = int(size)
        if self._active_tool_obj and hasattr(self._active_tool_obj, "on_cursor_size_changed"):
            self._active_tool_obj.on_cursor_size_changed(size)
        self._update_brush_preview()

    def _on_brush_color_changed(self, color: QtGui.QColor):
        # External palette sets global color; persist across tools
        self._brush_color = QtGui.QColor(color)
        if self._active_tool_obj and hasattr(self._active_tool_obj, "on_color_changed"):
            self._active_tool_obj.on_color_changed(color)
        self._update_brush_preview()

    def _set_cursor_sprite(self, pm: QtGui.QPixmap, hot_x: int, hot_y: int):
        self.image_canvas.setCursor(QtGui.QCursor(pm, hot_x, hot_y))
        self.brush_controls.set_preview_pixmap(pm)

    def _on_canvas_mouse_event(self, event_type: str, pos: QtCore.QPoint, left_down: bool, right_down: bool):
        if self._active_tool_obj and hasattr(self._active_tool_obj, "on_mouse_event"):
            self._active_tool_obj.on_mouse_event(event_type, pos, left_down, right_down)

    def _on_stamp_request(self, img: QtGui.QImage, pos: QtCore.QPoint, *args, **kwargs):
        erase = False
        layer = kwargs.get("layer", "paint")
        if args and len(args) > 0:
            erase = bool(args[0])
        if DETAILED_LOG:
            print(f"[PaintView] Stamp request received: pos={pos}, erase={erase}, layer={layer}, img={img}, args={args}")
        if layer == "mask":
            overlay = getattr(self.image_canvas, "_mask_overlay", None)
            if overlay is None or overlay.isNull():
                overlay = QtGui.QImage(self.image_canvas._pixmap.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
                overlay.fill(0)
                self.image_canvas._mask_overlay = overlay
        else:
            overlay = getattr(self.image_canvas, "_overlay", None)
            if overlay is None or overlay.isNull():
                overlay = QtGui.QImage(self.image_canvas._pixmap.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
                overlay.fill(0)
                self.image_canvas._overlay = overlay
        painter = QtGui.QPainter(overlay)
        if erase:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Clear)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.transparent))
            painter.drawEllipse(pos.x(), pos.y(), img.width(), img.height())
        else:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.drawImage(pos, img)
        painter.end()
        self.image_canvas.update()

    def _compose_current_image(self) -> Optional[QtGui.QImage]:
        if not self.current_path:
            return None
        base = QtGui.QImage(str(self.current_path))
        if base.isNull():
            return None
        overlay = getattr(self.image_canvas, "_overlay", None)
        return compose_images(base, overlay)

    def _get_mask_image(self) -> Optional[QtGui.QImage]:
        try:
            return self.image_canvas.get_mask_image()
        except Exception:
            return None

    def _update_brush_preview(self):
        # Show dynamic cursor sprite preview for blur/clone/mask
        if self._active_tool == "blur" and hasattr(self._active_tool_obj, "update_cursor_with_blur"):
            # Use center of canvas for preview if no mouse position
            canvas = self.image_canvas
            if canvas and hasattr(canvas, "rect"):
                center = canvas.rect().center()
                self._active_tool_obj.update_cursor_with_blur(center)
                # The update_cursor_cb will set the preview pixmap
                return
        elif self._active_tool == "mask":
            # Show white brush for mask, black for erase
            erase_mode = False
            if hasattr(self._active_tool_obj, "_erase_mode"):
                erase_mode = getattr(self._active_tool_obj, "_erase_mode")
            color = QtGui.QColor(0, 0, 0) if erase_mode else QtGui.QColor(255, 255, 255)
            pm = QtGui.QPixmap(BRUSH_PREVIEW_SIZE_PX, BRUSH_PREVIEW_SIZE_PX)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(pm.rect(), QtGui.QColor("#ffffff"))
            pen = QtGui.QPen(QtGui.QColor("#444444"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(color)
            r = max(2, min(self._brush_size, BRUSH_PREVIEW_SIZE_PX - 6))
            cx = cy = BRUSH_PREVIEW_SIZE_PX // 2
            painter.drawEllipse(QtCore.QPointF(cx, cy), r / 2.0, r / 2.0)
            painter.end()
            self.brush_controls.set_preview_pixmap(pm)
            return
        elif self._active_tool == "clone":
            sprite = self.image_canvas.get_cursor_sprite()
            if sprite is not None:
                pm = draw_brush_preview(sprite, BRUSH_PREVIEW_SIZE_PX, QtGui.QColor("yellow"))
            else:
                pm = QtGui.QPixmap(BRUSH_PREVIEW_SIZE_PX, BRUSH_PREVIEW_SIZE_PX)
                pm.fill(QtCore.Qt.GlobalColor.transparent)
                painter = QtGui.QPainter(pm)
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
                painter.fillRect(pm.rect(), QtGui.QColor("#ffffff"))
                pen = QtGui.QPen(QtGui.QColor("#444444"))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor("yellow"))
                r = max(2, min(self._brush_size, BRUSH_PREVIEW_SIZE_PX - 6))
                cx = cy = BRUSH_PREVIEW_SIZE_PX // 2
                painter.drawEllipse(QtCore.QPointF(cx, cy), r / 2.0, r / 2.0)
                painter.end()
            self.brush_controls.set_preview_pixmap(pm)
            return
        # Default: solid circle of current brush color
        pm = QtGui.QPixmap(BRUSH_PREVIEW_SIZE_PX, BRUSH_PREVIEW_SIZE_PX)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(pm.rect(), QtGui.QColor("#ffffff"))
        pen = QtGui.QPen(QtGui.QColor("#444444"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(self._brush_color)
        r = max(2, min(self._brush_size, BRUSH_PREVIEW_SIZE_PX - 6))
        cx = cy = BRUSH_PREVIEW_SIZE_PX // 2
        painter.drawEllipse(QtCore.QPointF(cx, cy), r / 2.0, r / 2.0)
        painter.end()
        self.brush_controls.set_preview_pixmap(pm)
        # Solid paint cursor on canvas while in paint mode
        if self._active_tool == "paint":
            size = max(8, self._brush_size)
            cur_pm = QtGui.QPixmap(size, size)
            cur_pm.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(cur_pm)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(self._brush_color)
            p.drawEllipse(QtCore.QPointF(size / 2, size / 2), size / 2 - 0.5, size / 2 - 0.5)
            p.end()
            self.image_canvas.setCursor(QtGui.QCursor(cur_pm, size // 2, size // 2))

    def _save_current(self):
        def show_message(msg: str, title: str = "Save"):
            QtWidgets.QMessageBox.warning(self, title, msg)

        if not self.current_path:
            show_message("No image selected.")
            return
        try:
            out_path = next_edit_filename(self.current_path)
            base = QtGui.QImage(str(self.current_path))
            if base.isNull():
                show_message("Failed to read source image.", "Save Error")
                return
            overlay = getattr(self.image_canvas, "_overlay", None)
            composed = compose_images(base, overlay)
            ok = composed.save(str(out_path))
            if not ok:
                show_message("Failed to write image.", "Save Error")
                return
            self.saved.emit(out_path)
        except Exception as ex:
            show_message(f"Error: {ex}", "Save Error")

    def _cancel_edit(self):
        self.canceled.emit()

    def _on_tool_event(self, event: str, value: str | bool):
        if event == "error":
            QtWidgets.QMessageBox.critical(self, "Tool Error", str(value))
        elif event == "busy":
            if value:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.BusyCursor)
                self.setEnabled(False)
            else:
                try:
                    QtWidgets.QApplication.restoreOverrideCursor()
                except Exception:
                    pass
                self.setEnabled(True)
        elif event == "save":
            out_path = Path(str(value))
            self.saved.emit(out_path)
            if self._active_tool == "diffuse":
                self.diffused.emit(out_path)
        elif event == "color":
            # Persist picked/palette color as global brush color across tool switches & size changes
            if isinstance(value, QtGui.QColor):
                self._brush_color = QtGui.QColor(value)
            else:
                self._brush_color = QtGui.QColor(str(value))
            # Update active paint tool if needed
            if self._active_tool == "paint" and self._active_tool_obj and hasattr(self._active_tool_obj, "on_color_changed"):
                self._active_tool_obj.on_color_changed(self._brush_color)
            self._update_brush_preview()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.image_canvas and event.type() == QtCore.QEvent.Type.Wheel:
            try:
                frame = self.image_canvas._fit_rect()
                posf = event.position()
                if isinstance(posf, QtCore.QPointF) and not frame.contains(posf):
                    return super().eventFilter(obj, event)
            except Exception:
                pass
            dy = 0
            try:
                dy = event.angleDelta().y() or event.pixelDelta().y()
            except Exception:
                dy = 0
            if dy != 0:
                sign = 1 if dy > 0 else -1
                new_size = max(BRUSH_MIN, min(BRUSH_MAX, int(self._brush_size + sign * BRUSH_WHEEL_STEP)))
                if new_size != self._brush_size:
                    self.brush_controls.set_size(new_size)
                event.accept()
                return True
        return super().eventFilter(obj, event)
