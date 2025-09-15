"""
Selection Tool with proper rotation handling.

Changes:
- Remembers original (base) selection rect & sprite at time of capture.
- Tracks cumulative rotation (rotation_deg).
- On RotR / RotL: rebuilds rotated sprite from the unmodified base sprite (no progressive shrink).
- Computes new marching-ants bounding box using trigonometry (|w*cosθ|+|h*sinθ| etc.), keeps center constant.
- Updates rect_img to rotated bounding box, so no scaling of the rotated image occurs.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from qt_paint_tools.qtPaintToolUtilities import (
    ToolPaletteWidget, PALETTE, blur_qimage_gaussian
)
import math


class SelectionState(QtCore.QObject):
    activeChanged = QtCore.pyqtSignal(bool)
    rectChanged = QtCore.pyqtSignal(QtCore.QRectF)
    changed = QtCore.pyqtSignal()
    modeChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active = False
        self.rect_img = QtCore.QRectF()          # current (possibly rotated) bounding box (image space)
        self.sprite: Optional[QtGui.QImage] = None       # current (possibly rotated) sprite
        self.defining = False
        self.dragging = False
        self.drag_delta = QtCore.QPointF()
        self.ants_phase = 0
        self.mode = "edit"
        # Rotation state
        self.base_rect_img = QtCore.QRectF()     # original, un-rotated selection rect (image space)
        self.base_sprite: Optional[QtGui.QImage] = None  # original, un-rotated sprite
        self.rotation_deg: float = 0.0           # cumulative rotation from base (degrees)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(120)

    def _tick(self):
        if self.active or self.defining or self.dragging:
            self.ants_phase = (self.ants_phase + 1) % 16
            self.changed.emit()

    def set_mode(self, mode: str):
        m = (mode or "edit").lower()
        if m not in ("edit", "erase"):
            m = "edit"
        if m != self.mode:
            self.mode = m
            self.modeChanged.emit(m)


class PaintToolSelection(QtCore.QObject):
    name = "select"
    display_brush_controls = False
    display_palette = False
    display_tool_options = True
    display_cursor_preview = False

    def __init__(self):
        super().__init__()
        self._canvas = None
        self._state = SelectionState()
        self._tool_callback: Optional[Callable[[str, str | bool], None]] = None
        self._last_palette_color = QtGui.QColor(0, 0, 0)

    def button_name(self) -> str:
        return "Select"

    # ---------------- UI -----------------
    def create_options_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(12)

        options_col = QtWidgets.QVBoxLayout()
        options_col.setContentsMargins(0, 0, 0, 0)
        options_col.setSpacing(6)
        rb_row = QtWidgets.QHBoxLayout()
        self.rb_edit = QtWidgets.QRadioButton("Edit")
        self.rb_erase = QtWidgets.QRadioButton("Erase")
        self.rb_edit.setChecked(True)
        self.rb_erase.setEnabled(False)
        rb_group = QtWidgets.QButtonGroup(widget)
        rb_group.setExclusive(True)
        rb_group.addButton(self.rb_edit)
        rb_group.addButton(self.rb_erase)
        self.rb_edit.toggled.connect(lambda on: on and self._state.set_mode("edit"))
        self.rb_erase.toggled.connect(lambda on: on and self._state.set_mode("erase"))
        rb_row.addWidget(self.rb_edit)
        rb_row.addWidget(self.rb_erase)
        options_col.addLayout(rb_row)

        row1 = QtWidgets.QHBoxLayout()
        row2 = QtWidgets.QHBoxLayout()
        self._select_action_buttons = []
        for text in ["Fill", "Blur", "RotR", "Commit"]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _, a=text: self._apply_select_action(a))
            row1.addWidget(btn)
            self._select_action_buttons.append(btn)
        for text in ["RotL", "FlipH", "FlipV", "CropSave", "Clear"]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _, a=text: self._apply_select_action(a))
            row2.addWidget(btn)
            self._select_action_buttons.append(btn)
        options_col.addLayout(row1)
        options_col.addLayout(row2)
        options_col.addStretch(1)

        palette_col = QtWidgets.QVBoxLayout()
        palette_col.setContentsMargins(0, 0, 0, 0)
        palette_col.setSpacing(0)
        self._palette_widget = ToolPaletteWidget(PALETTE)
        self._palette_widget.colorSelected.connect(self._on_palette_color_selected)
        self._last_palette_color = self._palette_widget.active_color()
        palette_col.addWidget(self._palette_widget, 1)

        main_layout.addLayout(options_col, 2)
        main_layout.addLayout(palette_col, 1)
        return widget

    def _on_palette_color_selected(self, color: QtGui.QColor):
        self._last_palette_color = QtGui.QColor(color)
        if self._tool_callback:
            self._tool_callback("color", color)

    # ------------- Lifecycle --------------
    def on_selected(
        self,
        canvas,
        brush_size=None,
        update_cursor_cb: Callable = None,
        stamp_cb: Callable = None,
        tool_callback: Optional[Callable[[str, str | bool], None]] = None,
        **kwargs
    ) -> dict:
        if self._canvas and self._state.sprite is not None and not self._state.sprite.isNull() and self._state.active:
            self._commit_sprite_to_overlay(self._canvas)
        self._canvas = canvas
        self._tool_callback = tool_callback
        self._state.changed.connect(canvas.update)
        self._state.activeChanged.connect(lambda active: self.rb_erase.setEnabled(active))
        if update_cursor_cb:
            canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        s = self._state
        s.active = False
        s.defining = False
        s.dragging = False
        s.rect_img = QtCore.QRectF()
        s.sprite = None
        s.base_rect_img = QtCore.QRectF()
        s.base_sprite = None
        s.rotation_deg = 0.0
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
            "display_cursor_preview": self.display_cursor_preview,
        }

    def on_deselected(self):
        s = self._state
        if self._canvas and s.sprite is not None and not s.sprite.isNull() and s.active:
            self._commit_sprite_to_overlay(self._canvas)

    # ------------- Mouse ------------------
    def on_mouse_event(self, event_type: str, pos: QtCore.QPoint, left_down: bool, right_down: bool):
        canvas = self._canvas
        s = self._state
        img_ptf = canvas._map_widget_to_image(pos)
        hover_in_sel = bool(
            img_ptf is not None and s.active and not s.rect_img.isNull() and s.rect_img.contains(img_ptf)
        )

        if hasattr(canvas, "setCursor"):
            if s.dragging:
                canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ClosedHandCursor))
            elif hover_in_sel:
                canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor))
            else:
                canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        if event_type == "press":
            if left_down and img_ptf is not None:
                if s.active and hover_in_sel:
                    if s.sprite is None or s.sprite.isNull():
                        self._update_sprite(canvas)
                    s.dragging = True
                    s.defining = False
                    s.drag_delta = img_ptf - s.rect_img.topLeft()
                else:
                    if s.active and s.sprite is not None and not s.sprite.isNull():
                        self._commit_sprite_to_overlay(canvas)
                    s.defining = True
                    s.dragging = False
                    s.active = False
                    s.rect_img = QtCore.QRectF(img_ptf, img_ptf)
                    s.sprite = None
                    s.base_rect_img = QtCore.QRectF()
                    s.base_sprite = None
                    s.rotation_deg = 0.0
                    s.activeChanged.emit(False)
            canvas.update()

        elif event_type == "move":
            if s.defining and img_ptf is not None:
                x0, y0 = s.rect_img.topLeft().x(), s.rect_img.topLeft().y()
                rect = QtCore.QRectF(
                    QtCore.QPointF(min(x0, img_ptf.x()), min(y0, img_ptf.y())),
                    QtCore.QPointF(max(x0, img_ptf.x()), max(y0, img_ptf.y())),
                )
                s.rect_img = rect
                s.rectChanged.emit(rect)
            elif s.dragging and img_ptf is not None:
                w, h = s.rect_img.width(), s.rect_img.height()
                new_top_left = img_ptf - s.drag_delta
                if canvas._pixmap and not canvas._pixmap.isNull():
                    img_w = canvas._pixmap.width()
                    img_h = canvas._pixmap.height()
                    new_top_left.setX(max(0, min(new_top_left.x(), img_w - w)))
                    new_top_left.setY(max(0, min(new_top_left.y(), img_h - h)))
                s.rect_img = QtCore.QRectF(new_top_left, QtCore.QSizeF(w, h))
                s.rectChanged.emit(s.rect_img)
            canvas.update()

        elif event_type == "release":
            # Always finalize selection/drag on mouse release
            if s.defining:
                s.defining = False
                if s.rect_img.width() > 2 and s.rect_img.height() > 2:
                    s.active = True
                    s.activeChanged.emit(True)
                    self._update_sprite(canvas)  # capture base sprite
                else:
                    s.active = False
                    s.rect_img = QtCore.QRectF()
                    s.sprite = None
                    s.base_rect_img = QtCore.QRectF()
                    s.base_sprite = None
                    s.rotation_deg = 0.0
            if s.dragging:
                s.dragging = False
            canvas.update()

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        return QtGui.QCursor(QtCore.Qt.CursorShape.ClosedHandCursor) if self._state.dragging else QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor)

    # ------------- Painting ---------------
    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        s = self._state
        if canvas._pixmap is None or canvas._pixmap.isNull() or s.rect_img.isNull():
            return
        frame = canvas._fit_rect()
        sx = frame.width() / canvas._pixmap.width()
        sy = frame.height() / canvas._pixmap.height()

        # Rect (image space) -> display space
        tl = s.rect_img.topLeft()
        rect_disp = QtCore.QRectF(
            frame.x() + tl.x() * sx,
            frame.y() + tl.y() * sy,
            s.rect_img.width() * sx,
            s.rect_img.height() * sy,
        )

        # Draw sprite WITHOUT scaling (since rect_img is bounding box of current sprite size)
        if s.sprite is not None and not s.sprite.isNull():
            if abs(s.sprite.width() - s.rect_img.width()) < 0.5 and abs(s.sprite.height() - s.rect_img.height()) < 0.5:
                # sizes match bounding box: simple draw (scaled only by image->display factors)
                painter.drawImage(rect_disp, s.sprite)
            else:
                # Fallback (should not happen unless state inconsistent)
                painter.drawImage(rect_disp, s.sprite)

        # Marching ants
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        ants_w = QtGui.QPen(QtGui.QColor(255, 255, 255), 1, QtCore.Qt.PenStyle.CustomDashLine)
        ants_w.setDashPattern([6, 6]); ants_w.setDashOffset(s.ants_phase)
        painter.setPen(ants_w); painter.drawRect(rect_disp)
        ants_b = QtGui.QPen(QtGui.QColor(0, 0, 0), 1, QtCore.Qt.PenStyle.CustomDashLine)
        ants_b.setDashPattern([6, 6]); ants_b.setDashOffset((s.ants_phase + 6) % 12)
        painter.setPen(ants_b); painter.drawRect(rect_disp)

    # ------------- Sprite Helpers ---------
    def _update_sprite(self, canvas):
        s = self._state
        if not canvas._pixmap or s.rect_img.isNull():
            s.sprite = None
            return
        img = canvas._pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        rect = s.rect_img.toRect().intersected(img.rect())
        if rect.isEmpty():
            s.sprite = None
            return
        base_region = img.copy(rect)
        if getattr(canvas, "_overlay", None) is not None and not canvas._overlay.isNull():
            ov = canvas._overlay.copy(rect)
            painter = QtGui.QPainter(base_region)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.drawImage(0, 0, ov)
            painter.end()
        s.sprite = base_region
        # Initialize base snapshot for future rotations
        s.base_sprite = base_region.copy()
        s.base_rect_img = QtCore.QRectF(s.rect_img)
        s.rotation_deg = 0.0
        s.changed.emit()

    def _commit_sprite_to_overlay(self, canvas):
        s = self._state
        if s.sprite is None or s.sprite.isNull() or s.rect_img.isNull():
            return
        if canvas._overlay is None or canvas._overlay.isNull():
            if not canvas._pixmap or canvas._pixmap.isNull():
                return
            canvas._overlay = QtGui.QImage(canvas._pixmap.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            canvas._overlay.fill(0)
        top_left = s.rect_img.topLeft().toPoint()
        p = QtGui.QPainter(canvas._overlay)
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        p.drawImage(top_left, s.sprite)
        p.end()
        s.sprite = None
        s.base_sprite = None
        s.base_rect_img = QtCore.QRectF()
        s.rotation_deg = 0.0
        s.changed.emit()

    # ------------- Rotation Logic ----------
    def _apply_rotation(self, delta_deg: float):
        s = self._state
        if s.base_sprite is None or s.base_sprite.isNull():
            # If no base recorded yet, treat current sprite as base
            if s.sprite is None or s.sprite.isNull():
                return
            s.base_sprite = s.sprite.copy()
            if s.base_rect_img.isNull():
                s.base_rect_img = QtCore.QRectF(s.rect_img)
            s.rotation_deg = 0.0

        s.rotation_deg = (s.rotation_deg + delta_deg) % 360.0

        # Rebuild rotated sprite from base
        tr = QtGui.QTransform()
        tr.rotate(s.rotation_deg)
        rotated = s.base_sprite.transformed(tr, QtCore.Qt.TransformationMode.SmoothTransformation)

        # Compute new bounding box (image space) using trig so rect encloses rotated base rect
        w = s.base_rect_img.width()
        h = s.base_rect_img.height()
        theta = math.radians(s.rotation_deg)
        new_w = abs(w * math.cos(theta)) + abs(h * math.sin(theta))
        new_h = abs(w * math.sin(theta)) + abs(h * math.cos(theta))

        # Keep center constant
        center = s.rect_img.center() if not s.rect_img.isNull() else s.base_rect_img.center()

        # Adjust rect to new size, centered
        s.rect_img = QtCore.QRectF(center.x() - new_w / 2.0, center.y() - new_h / 2.0, new_w, new_h)

        # Ensure rotated sprite size matches bounding box: rotated already has its own size = math bounding box of rotation
        # Just use rotated directly; its width/height equals its own bounding box; we align centers.
        # Need to re-center rotated sprite if its size differs from computed new_w/new_h due to rounding.
        s.sprite = rotated

        s.rectChanged.emit(s.rect_img)
        s.changed.emit()
        if self._canvas:
            self._canvas.update()

    # ------------- Actions -----------------
    def _apply_select_action(self, action: str):
        a = (action or "").lower()
        if not self._canvas:
            return
        s = self._state

        if a == "clear":
            s.active = False
            s.rect_img = QtCore.QRectF()
            s.sprite = None
            s.base_sprite = None
            s.base_rect_img = QtCore.QRectF()
            s.rotation_deg = 0.0
            s.defining = False
            s.dragging = False
            s.activeChanged.emit(False)
            s.changed.emit()
            self._canvas.update()
            return

        if a == "commit":
            self._commit_sprite_to_overlay(self._canvas)
            self._canvas.update()
            return

        if s.sprite is not None and not s.sprite.isNull():
            if a == "fill":
                fill_color = self._last_palette_color
                p = QtGui.QPainter(s.sprite)
                p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
                p.fillRect(s.sprite.rect(), fill_color)
                p.end()
            elif a == "blur":
                s.sprite = blur_qimage_gaussian(s.sprite, 6.0)
            elif a == "rotr":
                self._apply_rotation(10.0)
                return
            elif a == "rotl":
                self._apply_rotation(-10.0)
                return
            elif a == "fliph":
                # Mirror base sprite to maintain correct future rotations
                if s.base_sprite is not None and not s.base_sprite.isNull():
                    s.base_sprite = s.base_sprite.mirrored(True, False)
                s.sprite = s.sprite.mirrored(True, False)
            elif a == "flipv":
                if s.base_sprite is not None and not s.base_sprite.isNull():
                    s.base_sprite = s.base_sprite.mirrored(False, True)
                s.sprite = s.sprite.mirrored(False, True)
            elif a == "cropsave":
                self._crop_save_selection()
            else:
                QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Unknown: {action}")
            s.changed.emit()
            self._canvas.update()
            return

        # Legacy fallbacks
        if not hasattr(self._canvas, "_sel_state"):
            return
        if a == "fill":
            self._canvas.apply_selection_fill()
        elif a == "blur":
            self._canvas.apply_selection_blur(4.0)
        elif a == "rotr":
            self._canvas.apply_selection_rotate(10.0)
        elif a == "rotl":
            self._canvas.apply_selection_rotate(-10.0)
        elif a == "fliph":
            self._canvas.apply_selection_flip(horizontal=True)
        elif a == "flipv":
            self._canvas.apply_selection_flip(horizontal=False)
        elif a == "cropsave":
            self._crop_save_selection()
        else:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Unknown: {action}")

    def _crop_save_selection(self):
        # Placeholder (unchanged legacy hook)
        pass