from __future__ import annotations
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from qtPaintHelpers import (
    aspect_fit_rect, widget_to_image_point, fast_blur_qimage
)
from qtPaintState import PaintState
from qt_paint_tools.qtPaintToolPaint import PaintToolPaint
from qt_paint_tools.qtPaintToolMask import PaintToolMask
from qt_paint_tools.qtPaintToolBlur import PaintToolBlur
from qt_paint_tools.qtPaintToolClone import PaintToolClone
from qt_paint_tools.qtPaintToolSelection import PaintToolSelection

# Sizing constants mirrored from qtPaint.py (keep in sync if edited there)
BRUSH_MIN = 2
BRUSH_MAX = 92
FRAME_BORDER_PX = 2  # yellow frame border width

class PaintCanvas(QtWidgets.QWidget):
    """Canvas that draws the image fitted into a framed rect with a yellow border.
    Shows a cursor sprite (circle) at mouse position when inside the frame.
    Has a transparent edit layer for paint/erase operations.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._overlay: Optional[QtGui.QImage] = None  # transparent edit canvas (image-space)
        self._mouse_pos: Optional[QtCore.QPoint] = None
        self._mouse_down: bool = False
        self._show_cursor: bool = False
        self._brush_size: int = 24          # display-space diameter (px)
        self._brush_color: QtGui.QColor = QtGui.QColor("#ff0000")
        self._mode: str = "Paint"           # Paint | Erase | Select | Blur | Clone | Diffuse
        self._blur_strength: float = 1.0     # 0.5 .. 4.0
        self._cursor_sprite: Optional[QtGui.QImage] = None  # display-space circular image

        self._state: Optional[PaintState] = None
        self._tool: Optional[object] = None

        # Tool registry (use modular PaintToolSelection)
        self._tools: dict[str, object] = {
            "paint": PaintToolPaint(),
            "blur": PaintToolBlur(),
            "clone": PaintToolClone(),
            "select": PaintToolSelection(),
            "mask": PaintToolMask(),
        }

        # Mask layer (top-most)
        self._mask_overlay: Optional[QtGui.QImage] = None
        self._mask_visible: bool = False
        self._mask_hardness: float = 1.0
        try:
            self._tools["mask"] = PaintToolMask()
        except Exception:
            pass

        # Clone visuals compatibility (used by paintEvent crosshair)
        self._clone_pending_capture: bool = False
        self._clone_source_img: Optional[QtCore.QPointF] = None
        self._clone_offset: Optional[QtCore.QPointF] = None

    # ---------- Public API ----------

    def set_state(self, state: PaintState):
        self._state = state
        # Initial sync
        self._brush_size = state.brush_size()
        self._brush_color = state.brush_color()
        self._blur_strength = state.blur_strength()
        self._on_active_tool_changed(state.active_tool())
        # Wire signals
        state.brushSizeChanged.connect(self._on_brush_size_changed)
        state.brushColorChanged.connect(self._on_brush_color_changed)
        state.blurStrengthChanged.connect(self._on_blur_strength_changed)
        state.activeToolChanged.connect(self._on_active_tool_changed)

    def set_pixmap(self, pm: Optional[QtGui.QPixmap]):
        self._pixmap = pm if pm and not pm.isNull() else None
        self._overlay = None
        self._mask_overlay = None
        self._cursor_sprite = None
        self.update()

    def clear_overlay(self):
        if self._overlay is not None:
            self._overlay.fill(0)
            self.update()

    def set_mode(self, mode: str):
        self._mode = mode or "Paint"
        self.update()

    def set_blur_strength(self, value: float):
        if self._state is not None:
            self._state.set_blur_strength(float(value))
            return
        self._blur_strength = max(0.5, min(4.0, float(value)))
        if self._mouse_pos and isinstance(self._tool, PaintToolBlur):
            evt = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(self._mouse_pos),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier
            )
            self._tool.mouseMove(self, evt, self._state)  # type: ignore
            self.update()

    # ---------- Mask API ----------
    def set_mask_visible(self, on: bool):
        self._mask_visible = bool(on)
        self.update()

    def set_mask_hardness(self, h: float):
        self._mask_hardness = max(0.0, min(1.0, float(h)))
        self.update()

    def get_mask_image(self) -> Optional[QtGui.QImage]:
        if self._mask_visible and self._mask_overlay is not None and not self._mask_overlay.isNull():
            return self._mask_overlay
        return None

    def get_cursor_sprite(self) -> Optional[QtGui.QImage]:
        return self._cursor_sprite

    # ---------- Internal: state sync ----------

    def _on_brush_size_changed(self, v: int):
        self._brush_size = int(v); self.update()

    def _on_brush_color_changed(self, c: QtGui.QColor):
        self._brush_color = QtGui.QColor(c); self.update()

    def _on_blur_strength_changed(self, s: float):
        self._blur_strength = float(s)
        if self._mouse_pos and isinstance(self._tool, PaintToolBlur):
            evt = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(self._mouse_pos),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier
            )
            self._tool.mouseMove(self, evt, self._state)  # type: ignore
        self.update()

    def _on_active_tool_changed(self, name: str):
        n = (name or "").lower()
        if self._tool:
            self._tool.exit(self, self._state)  # type: ignore
        self._tool = self._tools.get(n, None)
        if self._tool:
            self._tool.enter(self, self._state)  # type: ignore

        # Manage cursor via tool if provided, else reset for non-select
        cur = self._tool.cursorFor(self) if self._tool and hasattr(self._tool, "cursorFor") else None
        if cur:
            self.setCursor(cur)
        else:
            self.unsetCursor()

        title = {"paint": "Paint", "erase": "Erase", "select": "Select", "blur": "Blur", "clone": "Clone", "diffuse": "Diffuse"}.get(n, "Paint")
        self.set_mode(title)
        self.update()

    # ---------- Events ----------

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._show_cursor = True
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._show_cursor = False
        self._mouse_pos = None
        self._mouse_down = False
        super().leaveEvent(event)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if hasattr(self, "mouseEventCallback") and self.mouseEventCallback:
            self.mouseEventCallback(
                "press",
                event.position().toPoint(),
                event.button() == QtCore.Qt.MouseButton.LeftButton,
                event.button() == QtCore.Qt.MouseButton.RightButton
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if hasattr(self, "mouseEventCallback") and self.mouseEventCallback:
            self.mouseEventCallback(
                "move",
                event.position().toPoint(),
                event.buttons() & QtCore.Qt.MouseButton.LeftButton,
                event.buttons() & QtCore.Qt.MouseButton.RightButton
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if hasattr(self, "mouseEventCallback") and self.mouseEventCallback:
            self.mouseEventCallback(
                "release",
                event.position().toPoint(),
                event.button() == QtCore.Qt.MouseButton.LeftButton,
                event.button() == QtCore.Qt.MouseButton.RightButton
            )
        super().mouseReleaseEvent(event)

    # ---------- Painting ----------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), self.palette().color(QtGui.QPalette.ColorRole.Base))

        frame = self._fit_rect()

        if self._pixmap:
            painter.drawPixmap(frame.toRect(), self._pixmap)
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor("#666")))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No image")

        if self._overlay is not None and not self._overlay.isNull():
            painter.drawImage(frame.toRect(), self._overlay)

        if self._mask_visible and self._mask_overlay is not None and not self._mask_overlay.isNull():
            painter.setOpacity(1.0)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.drawImage(frame.toRect(), self._mask_overlay)

        # Tool-specific overlays (marching ants, etc.)
        if self._tool and hasattr(self._tool, "paintOverlay"):
            self._tool.paintOverlay(self, painter)

        pen = QtGui.QPen(QtGui.QColor("yellow")); pen.setWidth(FRAME_BORDER_PX)
        painter.setPen(pen); painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(frame)

        # Cursor sprite + ring
        if self._show_cursor and self._mouse_pos and frame.contains(QtCore.QPointF(self._mouse_pos)):
            r = max(2, min(self._brush_size, 256))
            center = QtCore.QPointF(self._mouse_pos)
            top_left = QtCore.QPointF(center.x() - r / 2.0, center.y() - r / 2.0)

            if self._cursor_sprite is not None and not self._cursor_sprite.isNull():
                painter.drawImage(QtCore.QRectF(top_left, QtCore.QSizeF(r, r)), self._cursor_sprite)

            ring_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 200)); ring_pen.setWidth(1)
            painter.setPen(ring_pen); painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, r / 2.0, r / 2.0)

        painter.end()

    # ---------- Core helpers ----------

    def _fit_rect(self) -> QtCore.QRectF:
        r = self.rect()
        if not self._pixmap or self._pixmap.isNull():
            return QtCore.QRectF(r)
        return aspect_fit_rect(r, self._pixmap.width(), self._pixmap.height())

    def _map_widget_to_image(self, pt: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        if not self._pixmap or self._pixmap.isNull():
            return None
        frame = self._fit_rect()
        return widget_to_image_point(pt, frame, self._pixmap.width(), self._pixmap.height())