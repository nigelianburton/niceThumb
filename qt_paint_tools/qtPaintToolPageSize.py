from __future__ import annotations
from PyQt6 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable, List, Tuple

ALLOWED_PAGE_SIZES: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

PREVIEW_FRAME_PX = 92
PREVIEW_INSET = 4  # margin inside frame for drawing

class PaintToolPageSize(QtCore.QObject):
    """
    Page Size Tool

    UI:
      [ 92x92 framed preview ]
      [Up]
      [Sz  H,W]
      [Down]
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
        # Start at 1024x1024
        self._index = next((i for i, hw in enumerate(ALLOWED_PAGE_SIZES) if hw == (1024, 1024)), 8)
        self._height, self._width = ALLOWED_PAGE_SIZES[self._index]

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
        # Ensure UI reflects current size
        self._update_size_label()
        self._update_preview()
        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
        }

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

        # Size selector column
        col_sizes = QtWidgets.QVBoxLayout()
        col_sizes.setContentsMargins(0, 0, 0, 0)
        col_sizes.setSpacing(4)

        btn_up = QtWidgets.QToolButton()
        btn_up.setText("▲")
        btn_up.clicked.connect(self._on_increase_aspect)

        self._lbl_size = QtWidgets.QLabel("")
        self._lbl_size.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lbl_size.setMinimumWidth(88)
        font = self._lbl_size.font()
        font.setBold(True)
        self._lbl_size.setFont(font)

        lbl_cap = QtWidgets.QLabel("Sz")
        lbl_cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        size_row = QtWidgets.QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(4)
        size_row.addWidget(lbl_cap)
        size_row.addWidget(self._lbl_size, 1)

        btn_down = QtWidgets.QToolButton()
        btn_down.setText("▼")
        btn_down.clicked.connect(self._on_decrease_aspect)

        col_sizes.addWidget(btn_up)
        col_sizes.addLayout(size_row)
        col_sizes.addWidget(btn_down)
        col_sizes.addStretch(1)

        # Expansion grid column
        expand_col = QtWidgets.QVBoxLayout()
        expand_col.setContentsMargins(0, 0, 0, 0)
        expand_col.setSpacing(0)

        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        # Helper to place empty fillers
        def ensure_cell(r, c):
            if grid.itemAtPosition(r, c) is None:
                spacer = QtWidgets.QLabel("")
                spacer.setFixedSize(22, 22)
                grid.addWidget(spacer, r, c)

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

        center_label = QtWidgets.QLabel("64")
        center_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        center_label.setFixedSize(32, 32)
        center_label.setStyleSheet("border:1px solid #444;")

        # Place per spec (cell numbers: 0..8)
        # Row0: cells 0 1 2
        # Row1: cells 3 4 5
        # Row2: cells 6 7 8
        ensure_cell(0, 0)
        grid.addWidget(self._btn_expand_up, 0, 1)
        ensure_cell(0, 2)

        grid.addWidget(self._btn_expand_left, 1, 0)
        grid.addWidget(center_label, 1, 1)
        grid.addWidget(self._btn_expand_right, 1, 2)

        ensure_cell(2, 0)
        ensure_cell(2, 1)
        grid.addWidget(self._btn_expand_down, 2, 2)  # per spec: cell 8 (row2,col2)

        expand_col.addWidget(grid_w, 0)
        expand_col.addStretch(1)

        layout.addWidget(preview_frame, 0)
        layout.addLayout(col_sizes, 0)
        layout.addLayout(expand_col, 0)
        layout.addStretch(1)

        self._options_widget = root
        self._update_size_label()
        self._update_preview()

    # --- Actions ---
    def _on_increase_aspect(self):
        # Move "away" from square toward more extreme aspect (higher index if available)
        if self._index < len(ALLOWED_PAGE_SIZES) - 1:
            self._index += 1
            self._apply_index()

    def _on_decrease_aspect(self):
        if self._index > 0:
            self._index -= 1
            self._apply_index()

    def _apply_index(self):
        self._height, self._width = ALLOWED_PAGE_SIZES[self._index]
        self._update_size_label()
        self._update_preview()
        self.sizeChanged.emit(self._height, self._width)

    def _expand_dir(self, direction: str):
        # Simple expansion: try to step one index toward larger area preserving direction bias.
        # For now just print/log; can be replaced with real logic later.
        if self._tool_callback:
            self._tool_callback("info", f"expand_{direction}")
        # Placeholder: no dimension change (spec did not define transformation)
        # Could extend later to adjust width/height with a +64 increment respecting allowed list.

    # --- UI updates ---
    def _update_size_label(self):
        if self._lbl_size:
            self._lbl_size.setText(f"{self._height},{self._width}")

    def _update_preview(self):
        if not self._preview_label:
            return
        pm = QtGui.QPixmap(PREVIEW_FRAME_PX, PREVIEW_FRAME_PX)
        pm.fill(QtGui.QColor("#222222"))  # background
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # Compute rectangle representing aspect ratio inside inset bounds
        avail = PREVIEW_FRAME_PX - PREVIEW_INSET * 2
        h, w = self._height, self._width
        if h <= 0 or w <= 0:
            painter.end()
            self._preview_label.setPixmap(pm)
            return
        # Fit by max dimension
        if h >= w:
            rect_h = avail
            rect_w = int(avail * (w / h))
        else:
            rect_w = avail
            rect_h = int(avail * (h / w))
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