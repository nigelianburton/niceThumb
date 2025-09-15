from __future__ import annotations
from typing import Optional, Callable

from PyQt6 import QtCore, QtGui, QtWidgets

from qtPaintHelpers import aspect_fit_rect, widget_to_image_point

class PaintCanvas(QtWidgets.QWidget):
    mouseEventCallback: Optional[Callable[[str, QtCore.QPoint, bool, bool], None]] = None

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._overlay: Optional[QtGui.QImage] = None
        self._mask_overlay: Optional[QtGui.QImage] = None
        self._tool = None
        self._sel_state = None
        self._mask_visible = False  # Default is off

    def set_pixmap(self, pixmap: Optional[QtGui.QPixmap]):
        self._pixmap = pixmap.copy() if pixmap else None
        if self._pixmap and not self._pixmap.isNull():
            size = self._pixmap.size()
            self._overlay = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            self._overlay.fill(0)
            self._mask_overlay = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            self._mask_overlay.fill(0)  # Fully transparent
        else:
            self._overlay = None
            self._mask_overlay = None
        self.update()

    def get_composed_image(self) -> Optional[QtGui.QImage]:
        if self._pixmap is None:
            return None
        base = self._pixmap.toImage()
        if self._overlay is None:
            return base
        
        composed = QtGui.QImage(base.size(), base.format())
        composed.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(composed)
        p.drawImage(0, 0, base)
        p.drawImage(0, 0, self._overlay)
        p.end()
        return composed

    def get_mask_image(self) -> Optional[QtGui.QImage]:
        return self._mask_overlay

    def clear_mask(self):
        """Clears the mask overlay."""
        if self._mask_overlay is not None and not self._mask_overlay.isNull():
            self._mask_overlay.fill(0)  # Fully transparent
            self.update()

    def is_mask_visible(self) -> bool:
        return self._mask_visible

    def set_mask_visible(self, visible: bool):
        self._mask_visible = bool(visible)
        self.update()

    def resize_canvas(self, new_w: int, new_h: int, dest_x: int, dest_y: int, fill_color: QtGui.QColor):
        """
        Resizes the canvas and all its layers, preserving existing content.
        """
        if not self._pixmap or self._pixmap.isNull():
            return

        # --- Resize Base Pixmap ---
        old_base = self._pixmap.toImage()
        new_base = QtGui.QImage(new_w, new_h, old_base.format())
        new_base.fill(fill_color)
        p_base = QtGui.QPainter(new_base)
        p_base.drawImage(dest_x, dest_y, old_base)
        p_base.end()
        self._pixmap = QtGui.QPixmap.fromImage(new_base)

        # --- Resize Paint Overlay ---
        if self._overlay and not self._overlay.isNull():
            old_overlay = self._overlay
            self._overlay = QtGui.QImage(new_w, new_h, old_overlay.format())
            self._overlay.fill(0)  # Transparent
            p_over = QtGui.QPainter(self._overlay)
            p_over.drawImage(dest_x, dest_y, old_overlay)
            p_over.end()

        # --- Resize Mask Overlay ---
        if self._mask_overlay and not self._mask_overlay.isNull():
            old_mask = self._mask_overlay
            self._mask_overlay = QtGui.QImage(new_w, new_h, old_mask.format())
            self._mask_overlay.fill(0)  # Fully transparent
            p_mask = QtGui.QPainter(self._mask_overlay)
            p_mask.drawImage(dest_x, dest_y, old_mask)
            p_mask.end()
            
        self.update()

    def commit_canvas_state(self):
        """Flatten the current state to the base pixmap."""
        composed = self.get_composed_image()
        if composed:
            self.set_pixmap(QtGui.QPixmap.fromImage(composed))

    def _fit_rect(self) -> QtCore.QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QtCore.QRectF(self.rect())
        return aspect_fit_rect(self.rect(), self._pixmap.width(), self._pixmap.height())

    def _map_widget_to_image(self, pos: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        frame = self._fit_rect()
        return widget_to_image_point(pos, frame, self._pixmap.width(), self._pixmap.height())

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#444444"))

        if self._pixmap is None or self._pixmap.isNull():
            p.end()
            return

        frame = self._fit_rect()
        p.drawPixmap(frame, self._pixmap, QtCore.QRectF(self._pixmap.rect()))

        if self._overlay is not None and not self._overlay.isNull():
            p.drawImage(frame, self._overlay)

        if self._mask_visible and self._mask_overlay is not None and not self._mask_overlay.isNull():
            print("[PaintCanvas] Drawing mask overlay")  # Add this line
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            p.drawImage(frame, self._mask_overlay)
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)

        if self._tool and hasattr(self._tool, "paintOverlay"):
            self._tool.paintOverlay(self, p)

        p.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.mouseEventCallback:
            self.mouseEventCallback("press", event.pos(),
                                    bool(event.buttons() & QtCore.Qt.MouseButton.LeftButton),
                                    bool(event.buttons() & QtCore.Qt.MouseButton.RightButton))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.mouseEventCallback:
            self.mouseEventCallback("move", event.pos(),
                                    bool(event.buttons() & QtCore.Qt.MouseButton.LeftButton),
                                    bool(event.buttons() & QtCore.Qt.MouseButton.RightButton))

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self.mouseEventCallback:
            self.mouseEventCallback("release", event.pos(),
                                    bool(event.buttons() & QtCore.Qt.MouseButton.LeftButton),
                                    bool(event.buttons() & QtCore.Qt.MouseButton.RightButton))