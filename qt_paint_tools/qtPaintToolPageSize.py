from __future__ import annotations
from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable, List, Tuple
import math

from .qtPaintToolUtilities import get_composed_image

EXPANSION_AMOUNT_PX = 128
PREVIEW_FRAME_PX = 92
PREVIEW_INSET = 4  # margin inside frame for drawing
FILL_COLOR = QtGui.QColor(128, 128, 128)  # Neutral gray for new areas
TARGET_AREA = 1024 * 1024

class PaintToolPageSize(QtCore.QObject):
    """
    Page Size Tool

    UI:
      [ 92x92 framed preview ]
      [H,W size display]
      [ 3x3 expansion grid with directional expand buttons ]

    Signals:
      sizeChanged(int height, int width)
    """
    name = "pagesize"
    display_brush_controls = False
    display_palette = False
    display_tool_options = True

    sizeChanged = QtCore.pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self._canvas = None
        self._tool_callback: Optional[Callable[[str, str | bool], None]] = None
        self._options_widget: Optional[QtWidgets.QWidget] = None
        self._height, self._width = 1024, 1024

        # UI refs
        self._lbl_size: Optional[QtWidgets.QLabel] = None
        self._preview_label: Optional[QtWidgets.QLabel] = None

    # --- Tool System API ---
    def button_name(self) -> str:
        return "PgSz"

    def create_options_widget(self) -> QtWidgets.QWidget:
        if self._options_widget is None:
            self._build_ui()
        return self._options_widget

    def on_selected(
        self,
        canvas,
        tool_callback: Optional[Callable[[str, str | bool], None]] = None,
        **_
    ) -> dict:
        self._canvas = canvas
        self._tool_callback = tool_callback
        
        # Initialize size from current canvas
        if self._canvas and hasattr(self._canvas, "_pixmap") and self._canvas._pixmap:
            pixmap = self._canvas._pixmap
            self._width = pixmap.width()
            self._height = pixmap.height()

        self._update_size_label()
        self._update_preview()
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
        }

    def on_deselected(self):
        """Called when the tool is switched away from."""
        if self._canvas:
            self._canvas.commit_canvas_state()

    # --- UI Construction ---
    def _build_ui(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Preview frame
        preview_frame = QtWidgets.QFrame()
        preview_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        preview_frame.setLineWidth(1)
        preview_frame.setFixedSize(PREVIEW_FRAME_PX, PREVIEW_FRAME_PX)
        pv_lay = QtWidgets.QVBoxLayout(preview_frame)
        pv_lay.setContentsMargins(0, 0, 0, 0)
        pv_lay.setSpacing(0)
        self._preview_label = QtWidgets.QLabel()
        self._preview_label.setFixedSize(PREVIEW_FRAME_PX, PREVIEW_FRAME_PX)
        self._preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pv_lay.addWidget(self._preview_label)

        # Size display column
        col_display = QtWidgets.QVBoxLayout()
        col_display.setContentsMargins(0, 0, 0, 0)
        col_display.setSpacing(4)
        
        self._lbl_size = QtWidgets.QLabel("")
        self._lbl_size.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lbl_size.setMinimumWidth(88)
        font = self._lbl_size.font()
        font.setBold(True)
        self._lbl_size.setFont(font)
        
        col_display.addWidget(self._lbl_size)
        col_display.addStretch(1)

        # Expansion grid column
        expand_col = QtWidgets.QVBoxLayout()
        expand_col.setContentsMargins(0, 0, 0, 0)
        expand_col.setSpacing(0)

        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        # Buttons for directional expansion
        self._btn_expand_up = QtWidgets.QToolButton()
        self._btn_expand_up.setText("↑")
        self._btn_expand_up.clicked.connect(lambda: self._expand_dir("up"))

        self._btn_expand_left = QtWidgets.QToolButton()
        self._btn_expand_left.setText("←")
        self._btn_expand_left.clicked.connect(lambda: self._expand_dir("left"))

        self._btn_expand_right = QtWidgets.QToolButton()
        self._btn_expand_right.setText("→")
        self._btn_expand_right.clicked.connect(lambda: self._expand_dir("right"))

        self._btn_expand_down = QtWidgets.QToolButton()
        self._btn_expand_down.setText("↓")
        self._btn_expand_down.clicked.connect(lambda: self._expand_dir("down"))

        center_label = QtWidgets.QLabel(str(EXPANSION_AMOUNT_PX))
        center_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        center_label.setFixedSize(32, 32)
        center_label.setStyleSheet("border:1px solid #444;")

        grid.addWidget(self._btn_expand_up, 0, 1)
        grid.addWidget(self._btn_expand_left, 1, 0)
        grid.addWidget(center_label, 1, 1)
        grid.addWidget(self._btn_expand_right, 1, 2)
        grid.addWidget(self._btn_expand_down, 2, 1)

        expand_col.addWidget(grid_w, 0)
        expand_col.addStretch(1)

        layout.addWidget(preview_frame, 0)
        layout.addLayout(col_display, 0)
        layout.addLayout(expand_col, 0)
        layout.addStretch(1)

        self._options_widget = root
        self._update_size_label()
        self._update_preview()

    # --- Actions ---
    def _expand_dir(self, direction: str):
        if not self._canvas:
            return

        old_w, old_h = self._width, self._height
        new_w, new_h = old_w, old_h
        dest_x, dest_y = 0, 0

        if direction == "up":
            new_h += EXPANSION_AMOUNT_PX
            dest_y = EXPANSION_AMOUNT_PX
        elif direction == "down":
            new_h += EXPANSION_AMOUNT_PX
            dest_y = 0
        elif direction == "left":
            new_w += EXPANSION_AMOUNT_PX
            dest_x = EXPANSION_AMOUNT_PX
        elif direction == "right":
            new_w += EXPANSION_AMOUNT_PX
            dest_x = 0
        
        # Round final dimensions to nearest multiple of 8
        self._width = round(new_w / 8) * 8
        self._height = round(new_h / 8) * 8
        
        # Instruct the canvas to perform the resize of all its layers
        self._canvas.resize_canvas(self._width, self._height, dest_x, dest_y, FILL_COLOR)

        self._update_size_label()
        self._update_preview()
        self.sizeChanged.emit(self._height, self._width)

    def _rescale_to_target_area(self, w: int, h: int) -> Tuple[int, int]:
        """
        Rescales dimensions to match TARGET_AREA while preserving aspect ratio,
        rounding to the nearest multiple of 8.
        """
        if w <= 0 or h <= 0:
            return 0, 0
        
        aspect = w / h
        target_w = math.sqrt(TARGET_AREA * aspect)
        target_h = TARGET_AREA / target_w
        
        # Round to nearest multiple of 8
        new_w = round(target_w / 8) * 8
        new_h = round(target_h / 8) * 8
        
        return int(new_w), int(new_h)

    # --- UI updates ---
    def _update_size_label(self):
        if self._lbl_size:
            self._lbl_size.setText(f"{self._height}x{self._width}")

    def _update_preview(self):
        if not self._preview_label:
            return
        pm = QtGui.QPixmap(PREVIEW_FRAME_PX, PREVIEW_FRAME_PX)
        pm.fill(QtGui.QColor("#222222"))  # background
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        avail = PREVIEW_FRAME_PX - PREVIEW_INSET * 2
        h, w = self._height, self._width
        if h <= 0 or w <= 0:
            painter.end()
            self._preview_label.setPixmap(pm)
            return
        
        if h >= w:
            rect_h = avail
            rect_w = int(avail * (w / h)) if h > 0 else 0
        else:
            rect_w = avail
            rect_h = int(avail * (h / w)) if w > 0 else 0
            
        x = (PREVIEW_FRAME_PX - rect_w) // 2
        y = (PREVIEW_FRAME_PX - rect_h) // 2

        painter.setPen(QtGui.QPen(QtGui.QColor("#555555"), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
        painter.drawRect(x, y, rect_w, rect_h)
        painter.end()
        self._preview_label.setPixmap(pm)

    # --- Optional canvas integrations (currently unused) ---
    def on_mouse_event(self, event_type: str, pos: QtCore.QPoint, left_down: bool, right_down: bool):
        pass

    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        pass

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        return None