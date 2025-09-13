"""
NiceThumb BrowserView (grid/file browser)

This module builds a scrollable, responsive thumbnail grid for a filesystem path and provides
single/multi-selection, range-selection (Shift), and callbacks for selection and edit actions.

Primary components:
- BrowserView (QtWidgets.QWidget)
  Role: Orchestrates header controls, grid of _ItemCard widgets, async thumbnailing, and selection.
  External callbacks:
    - on_item_selected(Path | None): called with a file when a single file is selected, else None.
    - on_edit_request(): invoked from the header “Edit” button.
  Public API:
    - render(): (re)scan current folder and rebuild grid
    - set_interactive(bool): temporarily disable interactions (header + grid)
    - refresh_and_select(path: Path): re-render parent folder, then select the item
    - get_thumbnail_path(file_path: Path) -> Path
    - delete_thumbnail(file_path: Path) -> None

- _ItemCard (QtWidgets.QFrame)
  Role: Visual card for a file/folder with a square thumbnail box and caption; emits clicked/doubleClicked.

- Thumbnail pipeline
  Behavior:
    - Images (suffix in IMAGE_SUFFIXES): loaded directly to QPixmap (no background worker).
    - Videos/others: thumbnail path is computed and generation is delegated to helpers in a background
      _ThumbTask. Results are returned via _ThumbSignals.finished and applied to the grid item.
  Threading:
    - Uses QtCore.QThreadPool and QRunnable to avoid blocking UI. Epoch guards drop stale results.

Integration with qtBrowseHelpers
- qtBrowseHelpers.safe_name(root: Path, file_path: Path) -> str
  Role: Stable, filesystem-safe filename for the thumbnail, derived from path relative to root.
- qtBrowseHelpers.thumbnail_path(thumbnail_dir: Path, root: Path, file_path: Path) -> Path
  Role: Compute the on-disk JPEG path for a thumbnail; used by BrowserView.get_thumbnail_path().
- qtBrowseHelpers.resolve_folder_proxy(file_path: Path) -> Optional[Path]
  Role: For folders, returns folder.jpg/png if present, else None. Used inside ensure_thumbnail.
- qtBrowseHelpers.ensure_thumbnail(root: Path, thumbnail_dir: Path, file_path: Path,
                                   thumb_max: int = 256, force: bool = False) -> Optional[Path]
  Role: Ensure a thumbnail file exists and return its path. Internally:
    - For images: generates a downscaled JPEG via Pillow.
    - For videos: captures a representative frame via OpenCV (cv2) then saves as JPEG.
  Use in this module:
    - Called asynchronously via _ThumbTask when a cached or prebuilt thumbnail is not present.

Notes
- Header includes path display, Up/Edit/Delete/All/LAN/Stories buttons, and S/M/L radios to choose
  the grid thumbnail box size (THUMB_S/M/L_PX). The radios update state['thumb_size'] and trigger render().
- Selection visuals are styled with QSS; selection state drives on_item_selected callback.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Callable, Optional, Dict, List, Set

from PyQt6 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFilter

from qtBrowseHelpers import (
    thumbnail_path as helper_thumbnail_path,
    ensure_thumbnail as helper_ensure_thumbnail
)
from qt_paint_tools.qtPaintToolUtilities import (
    blur_qimage_gaussian, pil_to_qimage, qimage_to_pil
)

DEBUG_BORDER_PX = 3

IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
VIDEO_SUFFIXES = ['.mp4', '.mov', '.avi', '.mkv']
THUMB_S_PX = 50
THUMB_M_PX = 100
THUMB_L_PX = 150
THUMB_SIZES = [THUMB_S_PX, THUMB_M_PX, THUMB_L_PX]
DEFAULT_THUMB_SIZE = THUMB_M_PX
THUMB_MAX = 256
THUMB_LABEL_FONT_SIZE_PT = 9

HEADER_PADDING_PX = 1       # was 5
HEADER_BUTTON_SIZE_PX = 37  # was 32
HEADER_SLIDER_WIDTH_PX = 128
NEW_IMAGE_BLUR_STRENGTH = 100
NEW_IMAGE_BLUR_ITERATIONS = 5
NEW_IMAGE_BG_COLOR = (245, 245, 245)  # Very light gray
NEW_IMAGE_RECT_COLOR = (180, 180, 180) # Light gray


class _ResizeAwareScrollArea(QtWidgets.QScrollArea):
    viewportResized = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        # Also watch the actual viewport so we get a signal when it first shows
        self.viewport().installEventFilter(self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.viewportResized.emit()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.viewport() and event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self.viewportResized.emit()
        return super().eventFilter(obj, event)


class _ThumbSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)  # (item_path: Path, result: object)


class _ThumbTask(QtCore.QRunnable):
    def __init__(self, item: Path, work_fn: Callable[[Path], Optional[Path]], signals: _ThumbSignals):
        super().__init__()
        self.item = item
        self.work_fn = work_fn
        self.signals = signals
        self.setAutoDelete(True)

    def run(self):
        try:
            res = self.work_fn(self.item)
        except Exception as ex:
            res = ex
        self.signals.finished.emit(self.item, res)


class _ItemCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(object, object)       # (event: QMouseEvent, path: Path)
    doubleClicked = QtCore.pyqtSignal(object, object) # (event: QMouseEvent, path: Path)

    def __init__(self, path: Path, thumb_size: int, parent=None):
        super().__init__(parent)
        self.path = path
        self.thumb_size = int(thumb_size)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setObjectName("itemCard")
        self.setAutoFillBackground(True)
        self._selected = False

        self._bg_normal = QtGui.QColor("#ffffff")
        self._bg_selected = QtGui.QColor("#bfdbfe")

        fixed_h = self.thumb_size + 40
        self.setMinimumHeight(fixed_h)
        self.setMaximumHeight(fixed_h)
        self.setFixedWidth(self.thumb_size + 20)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.thumb_label = QtWidgets.QLabel(self)
        # Make the thumbnail area a square box equal to thumb_size in both dimensions
        self.thumb_label.setFixedSize(self.thumb_size, self.thumb_size)
        self.thumb_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setText("…")
        self.thumb_label.setStyleSheet("QLabel { background: transparent; }")
        self.thumb_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

        self.name_label = QtWidgets.QLabel(self)
        f = self.name_label.font()
        f.setPointSize(THUMB_LABEL_FONT_SIZE_PT)
        self.name_label.setFont(f)
        self.name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.name_label.setText(self._caption_text())
        self.name_label.setWordWrap(True)
        fm = QtGui.QFontMetrics(self.name_label.font())
        self.name_label.setMaximumHeight(fm.lineSpacing() * 2)  # allow up to 2 lines
        self.name_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        self.name_label.setStyleSheet("color: #000000;")

        layout.addWidget(self.thumb_label, 1)
        layout.addWidget(self.name_label, 0)

        self._apply_bg()

        self.thumb_label.installEventFilter(self)
        self.name_label.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            self.doubleClicked.emit(event, self.path)
            return True
        elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
            self.clicked.emit(event, self.path)
            return True
        return super().eventFilter(obj, event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.doubleClicked.emit(event, self.path)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.clicked.emit(event, self.path)

    def set_selected(self, selected: bool):
        self._selected = selected
        # Drive QSS selector with a dynamic property
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _apply_bg(self):
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, self._bg_selected if self._selected else self._bg_normal)
        pal.setColor(QtGui.QPalette.ColorRole.Base, self._bg_selected if self._selected else self._bg_normal)
        self.setPalette(pal)
        self.setStyleSheet("QFrame#itemCard { border: none; }")

    def _set_placeholder_icon(self, is_dir: bool):
        # Render a large standard icon into a pixmap of thumb_size
        target = QtCore.QSize(self.thumb_size, self.thumb_size)
        std = QtWidgets.QStyle.StandardPixmap.SP_DirIcon if is_dir else QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        icon = self.style().standardIcon(std)
        pm = icon.pixmap(target)
        if pm.isNull():
            # Fallback: simple colored pixmap if style returns null (rare)
            pm = QtGui.QPixmap(self.thumb_size, self.thumb_size)
            pm.fill(QtGui.QColor("#e5e7eb"))  # light gray
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtGui.QPen(QtGui.QColor("#9ca3af")))
            painter.drawRect(2, 2, self.thumb_size - 4, self.thumb_size - 4)
            painter.end()
        self.thumb_label.setText("")
        self.thumb_label.setPixmap(pm)

    def set_pixmap(self, pix: Optional[QtGui.QPixmap]):
        if pix is None:
            self._set_placeholder_icon(self.path.is_dir())
            return

        target = QtCore.QSize(self.thumb_size, self.thumb_size)
        scaled = pix.scaled(target,
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation)
        self.thumb_label.setText("")
        self.thumb_label.setPixmap(scaled)

    def _caption_text(self) -> str:
        name = self.path.name
        return (name[:45] + "...") if len(name) > 48 else name


class BrowserView(QtWidgets.QWidget):
    llm_mode_requested = QtCore.pyqtSignal()

    def __init__(self,
                 state: dict,
                 on_item_selected: Callable[[Path | None], None],
                 root_path: Path,
                 thumbnail_dir: Path,
                 on_edit_request: Callable[[], None],
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.state = state
        self.on_item_selected = on_item_selected
        self.root_path = Path(root_path)
        self.thumbnail_dir = Path(thumbnail_dir)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.on_edit_request = on_edit_request

        self.items: List[Path] = []
        self.selected: Set[Path] = set()
        self.last_click: Optional[Path] = None
        self.card_map: Dict[Path, _ItemCard] = {}
        self.pix_cache: Dict[Path, QtGui.QPixmap] = {}
        self._interactive: bool = True

        self._pool = QtCore.QThreadPool.globalInstance()
        # Debug aid: serialize thumbnail workers to make native crashes reproducible
        try:
            self._pool.setMaxThreadCount(1)
        except Exception:
            pass

        # Re-entrancy guards and thumbnail epoching
        self._suspend_layout: bool = False
        self._pending_layout: bool = False
        self._render_epoch: int = 0
        self._thumb_epoch: Dict[Path, int] = {}

        self._build_ui()

        if 'current_path' not in self.state:
            self.state['current_path'] = str(self.root_path)
        if 'thumb_size' not in self.state:
            self.state['thumb_size'] = DEFAULT_THUMB_SIZE

        self.render()

    def get_folder_contents(self, path: Path) -> List[Path]:
        try:
            items = [p for p in path.iterdir()
                     if '$RECYCLE.BIN' not in p.name and self.thumbnail_dir.name not in p.name]
            items.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
            return items
        except (PermissionError, FileNotFoundError):
            return []

    def get_thumbnail_path(self, file_path: Path) -> Path:
        return helper_thumbnail_path(self.thumbnail_dir, self.root_path, file_path)

    def _create_thumbnail_worker(self, file_path: Path, force: bool = False) -> Optional[Path]:
        try:
            return helper_ensure_thumbnail(self.root_path, self.thumbnail_dir, file_path, thumb_max=THUMB_MAX, force=force)
        except Exception as e:
            print(f'[thumb] Failed for {file_path.name}: {e}')
            return None

    def delete_thumbnail(self, file_path: Path):
        try:
            thumb_path = self.get_thumbnail_path(file_path)
            if thumb_path.exists():
                thumb_path.unlink()
        except Exception as e:
            print(f'[thumb] Delete failed for {file_path.name}: {e}')

    def set_interactive(self, enabled: bool):
        self._interactive = enabled
        self._set_header_enabled(enabled)
        self.grid_widget.setEnabled(enabled)

    def refresh_and_select(self, path: Path):
        self.delete_thumbnail(path)
        parent = path.parent
        if Path(self.state.get('current_path', self.root_path)) != parent:
            self.state['current_path'] = str(parent)
        self.render()
        QtCore.QTimer.singleShot(50, lambda p=path: self._select_programmatically(p))

    def mount(self, parent: QtWidgets.QWidget):
        if parent.layout() is None:
            lay = QtWidgets.QVBoxLayout(parent)
            lay.setContentsMargins(0, 0, 0, 0)
        parent.layout().addWidget(self)

    def _build_ui(self):
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        border = QtWidgets.QFrame(self)
        border.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        border.setStyleSheet(f"QFrame {{ border: {DEBUG_BORDER_PX}px solid purple; }}")
        border_layout = QtWidgets.QVBoxLayout(border)
        border_layout.setContentsMargins(HEADER_PADDING_PX, HEADER_PADDING_PX, HEADER_PADDING_PX, HEADER_PADDING_PX)
        border_layout.setSpacing(0)

        header = QtWidgets.QWidget(border)
        header.setObjectName("headerRow")
        header.setStyleSheet("QWidget#headerRow { background: #f3f4f6; }")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(HEADER_PADDING_PX, HEADER_PADDING_PX, HEADER_PADDING_PX, HEADER_PADDING_PX)
        header_layout.setSpacing(1)  # 1px padding between buttons

        self.path_label = QtWidgets.QLabel("")
        self.path_label.setStyleSheet("QLabel { background: white; padding: 2px; border-radius: 2px; }")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addWidget(self.path_label, 1)  # Stretch factor 1

        # --- Button container ---
        button_container = QtWidgets.QWidget(header)
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(1)  # 1px padding between buttons

        self.btn_up = QtWidgets.QToolButton()
        self.btn_up.setText("Up")
        self.btn_up.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_up)

        self.btn_new = QtWidgets.QToolButton()
        self.btn_new.setText("New")
        self.btn_new.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_new)

        self.btn_edit = QtWidgets.QToolButton()
        self.btn_edit.setText("Edit")
        self.btn_edit.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_edit)

        self.btn_delete = QtWidgets.QToolButton()
        self.btn_delete.setText("Del")
        self.btn_delete.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_delete)

        self.btn_select_all = QtWidgets.QToolButton()
        self.btn_select_all.setText("All")
        self.btn_select_all.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_select_all)

        self.btn_auto = QtWidgets.QToolButton()
        self.btn_auto.setText("AUTO")
        self.btn_auto.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_auto)

        self.btn_llm = QtWidgets.QToolButton()
        self.btn_llm.setText("LLM")
        self.btn_llm.setFixedSize(HEADER_BUTTON_SIZE_PX, HEADER_BUTTON_SIZE_PX)
        button_layout.addWidget(self.btn_llm)

        button_container.setFixedWidth(7 * HEADER_BUTTON_SIZE_PX + 7 * 1)  # 7 buttons, 1px spacing each
        header_layout.addWidget(button_container, 0)  # Stretch factor 0

        # S/M/L size buttons
        size_box = QtWidgets.QWidget(header)
        size_layout = QtWidgets.QHBoxLayout(size_box)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(2)
        self.btn_s = QtWidgets.QPushButton("S")
        self.btn_m = QtWidgets.QPushButton("M")
        self.btn_l = QtWidgets.QPushButton("L")
        for btn in (self.btn_s, self.btn_m, self.btn_l):
            btn.setCheckable(True)
            btn.setMinimumWidth(32)
            btn.setMaximumWidth(38)
            btn.setStyleSheet("""
                QPushButton {
                    background: #f3f4f6;
                    color: #222;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                }
                QPushButton:hover:!checked {
                    background: #e5e7eb;
                    color: #111;
                }
                QPushButton:checked {
                    background: #3b82f6;
                    color: #fff;
                    border: 2px solid #2563eb;
                }
            """)
            size_layout.addWidget(btn)
        self.size_group = QtWidgets.QButtonGroup(self)
        self.size_group.setExclusive(True)
        self.size_group.addButton(self.btn_s)
        self.size_group.addButton(self.btn_m)
        self.size_group.addButton(self.btn_l)
        self.btn_m.setChecked(True)  # Default selection

        self.btn_s.clicked.connect(lambda: self._on_size_radio(THUMB_S_PX))
        self.btn_m.clicked.connect(lambda: self._on_size_radio(THUMB_M_PX))
        self.btn_l.clicked.connect(lambda: self._on_size_radio(THUMB_L_PX))
        header_layout.addWidget(size_box, 0)

        border_layout.addWidget(header)
        root_layout.addWidget(border, 0)

        self.scroll = _ResizeAwareScrollArea(self)
        self.scroll.viewportResized.connect(self._relayout)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root_layout.addWidget(self.scroll, 1)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_layout.setHorizontalSpacing(8)
        self.grid_layout.setVerticalSpacing(8)
        self.scroll.setWidget(self.grid_widget)

        # Selection visuals: thin base border; 2px accent when selected
        self.setStyleSheet("""
            QFrame#itemCard {
                border: 1px solid transparent;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QFrame#itemCard[selected="true"] {
                border: 2px solid #3b82f6;     /* accent blue */
                background-color: #e0f2fe;     /* subtle tint */
            }
        """)

        self.btn_up.clicked.connect(self._go_up)
        self.btn_new.clicked.connect(self._on_new_clicked)
        self.btn_edit.clicked.connect(self._on_edit_clicked)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_select_all.clicked.connect(self._select_all_files)
        self.btn_auto.clicked.connect(self._on_auto_clicked)
        self.btn_llm.clicked.connect(self.llm_mode_requested.emit)

    def _on_edit_clicked(self):
        if self._edit_enabled():
            self.on_edit_request()

    def _edit_enabled(self):
        if len(self.selected) == 1:
            selected_item = next(iter(self.selected))
            return selected_item.is_file() and selected_item.suffix.lower() in IMAGE_SUFFIXES
        return False

    def _on_auto_clicked(self):
        QtWidgets.QMessageBox.information(self, "Auto", "AUTO action triggered")

    def _on_new_clicked(self):
        """Creates a new blank (blurred rectangle) image in the current directory."""
        try:
            # 1. Generate the image content
            img = self._generate_new_base_image(1024, 1024)

            # 2. Determine a unique filename
            current_dir = Path(self.state.get('current_path', self.root_path))
            timestamp = int(time.time())
            i = 0
            while True:
                suffix = f"_{i}" if i > 0 else ""
                new_path = current_dir / f"new_{timestamp}{suffix}.png"
                if not new_path.exists():
                    break
                i += 1

            # 3. Save the image
            img.save(str(new_path), "PNG")

            # 4. Refresh the browser and select the new file
            self.refresh_and_select(new_path)

        except Exception as e:
            print(f"[browse] Failed to create new image: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not create new image:\n{e}")

    def _generate_new_base_image(self, width: int, height: int) -> Image.Image:
        """
        Creates a new image of the specified size with a blurred rectangle.
        """
        W, H = width, height
        # Scale the rectangle to be ~80% of the image dimensions
        RW, RH = int(W * 0.8), int(H * 0.8)
        rx = (W - RW) // 2
        ry = H - RH

        img = Image.new("RGB", (W, H), NEW_IMAGE_BG_COLOR)
        dr = ImageDraw.Draw(img)
        dr.rectangle([rx, ry, rx + RW, ry + RH], fill=NEW_IMAGE_RECT_COLOR)

        # Convert to QImage, blur, and convert back to PIL. No fallback.
        qimg = pil_to_qimage(img)
        blurred_q = qimg
        for _ in range(NEW_IMAGE_BLUR_ITERATIONS):
            blurred_q = blur_qimage_gaussian(blurred_q, strength=NEW_IMAGE_BLUR_STRENGTH)
        out = qimage_to_pil(blurred_q)
        return out.convert("RGB")

    def _update_header_buttons(self):
        self.btn_edit.setEnabled(self._edit_enabled())
        self.btn_auto.setEnabled(len(self.selected) > 1 and all(p.is_file() for p in self.selected))
        self.btn_delete.setEnabled(bool(self.selected))

    def render(self):
        # Prevent re-entrant relayout while we rebuild the grid
        self._suspend_layout = True
        self._render_epoch += 1
        epoch = self._render_epoch

        cur = Path(self.state.get('current_path', self.root_path))
        if not cur.exists():
            cur = self.root_path
            self.state['current_path'] = str(cur)

        self.items = self.get_folder_contents(cur)
        self.selected.clear()
        self.last_click = None

        self.path_label.setText(str(cur))

        # Keep S/M/L radios in sync with current state
        self._sync_size_radios()

        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.card_map.clear()

        size = int(self.state.get('thumb_size', DEFAULT_THUMB_SIZE))

        for item_path in self.items:
            card = _ItemCard(item_path, size, parent=self.grid_widget)
            card.clicked.connect(self._on_card_clicked)
            card.doubleClicked.connect(self._on_card_double_clicked)
            self.card_map[item_path] = card
            # Record epoch for this item and queue thumbnail
            self._thumb_epoch[item_path] = epoch
            self._queue_thumb(item_path)

        # Resume layout and perform a single relayout now
        self._suspend_layout = False
        self._relayout()
        # If any relayouts were requested while suspended, do one more pass
        if self._pending_layout:
            self._pending_layout = False
            self._relayout()

    def _relayout(self):
        if self._suspend_layout:
            # Defer; a render() is in progress
            self._pending_layout = True
            return
        width = self.scroll.viewport().width()
        size = int(self.state.get('thumb_size', DEFAULT_THUMB_SIZE))
        grid_min_width = size + 20
        if grid_min_width <= 0:
            grid_min_width = 64
        if width <= 0:
            width = 1
        spacing = self.grid_layout.horizontalSpacing() or 8
        margins = self.grid_layout.contentsMargins()
        cols = max(1, (width - margins.left() - margins.right() + spacing) // (grid_min_width + spacing))

        while self.grid_layout.count():
            _ = self.grid_layout.takeAt(0)

        idx = 0
        for path in self.items:
            card = self.card_map.get(path)
            if not card:
                continue
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)
            idx += 1

    def _on_slider_change(self, value: int):
        pass  # deprecated: slider removed in favor of S/M/L radios

    def _queue_thumb(self, item: Path):
        suffix = item.suffix.lower()
        # For supported images, avoid background decoding; let Qt load directly.
        if suffix in IMAGE_SUFFIXES:
            pix = self._load_pixmap(item, int(self.state.get('thumb_size', DEFAULT_THUMB_SIZE)))
            self.pix_cache[item] = pix if pix is not None else QtGui.QPixmap()
            self._apply_thumb_to_card(item, self.pix_cache.get(item))
            return
        # For supported videos, try to use cached or generated thumbnails.
        if suffix in VIDEO_SUFFIXES:
            thumb_path = self.get_thumbnail_path(item)
            if thumb_path.exists():
                pix = self._load_pixmap(thumb_path, int(self.state.get('thumb_size', DEFAULT_THUMB_SIZE)))
                self.pix_cache[item] = pix if pix is not None else QtGui.QPixmap()
                self._apply_thumb_to_card(item, self.pix_cache.get(item))
                return
            signals = _ThumbSignals()
            signals.finished.connect(self._on_thumb_ready)
            task = _ThumbTask(item, lambda p=item: self._create_thumbnail_worker(p, False), signals)
            this = self._pool.start(task)
            return

        # For unknown/unsupported types: show generic document icon; don't try to thumb.
        self._apply_thumb_to_card(item, None)
        return

    def _on_thumb_ready(self, item: Path, result: object):
        # Ignore stale completions from previous renders
        if self._thumb_epoch.get(item) != self._render_epoch:
            return
        if isinstance(result, Exception):
            print(f'[thumb] Error for {item.name}: {result}')
            self._apply_thumb_to_card(item, None)
            return

        if isinstance(result, Path) and result.exists():
            pix = self._load_pixmap(result, int(self.state.get('thumb_size', DEFAULT_THUMB_SIZE)))
            self.pix_cache[item] = pix if pix is not None else QtGui.QPixmap()
            self._apply_thumb_to_card(item, self.pix_cache.get(item))
        else:
            self._apply_thumb_to_card(item, None)

    def _load_pixmap(self, path: Path, size: int) -> Optional[QtGui.QPixmap]:
        try:
            pm = QtGui.QPixmap(str(path))
            if pm.isNull():
                return None
            # Let _ItemCard.set_pixmap() handle scaling to the target box
            return pm
        except Exception as ex:
            print(f'[thumb] Render failed for {path.name}: {ex}')
            return None

    def _apply_thumb_to_card(self, item: Path, pix: Optional[QtGui.QPixmap]):
        card = self.card_map.get(item)
        if not card:
            return
        card.set_pixmap(pix)

    def _on_card_double_clicked(self, event: QtGui.QMouseEvent, item: Path):
        if not self._interactive:
            return
        if item.is_dir():
            self._set_path(item)

    def _on_card_clicked(self, event: QtGui.QMouseEvent, item: Path):
        if not self._interactive:
            return

        mods = event.modifiers()
        is_ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier) or bool(mods & QtCore.Qt.KeyboardModifier.MetaModifier)
        is_shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)

        if is_shift and self.last_click and self.last_click in self.items:
            try:
                start_idx = self.items.index(self.last_click)
                end_idx = self.items.index(item)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                self.selected.clear()
                for i in range(start_idx, end_idx + 1):
                    self.selected.add(self.items[i])
            except ValueError:
                self.selected.clear()
                self.selected.add(item)
        elif is_ctrl:
            if item in self.selected:
                self.selected.remove(item)
            else:
                self.selected.add(item)
            self.last_click = item
        else:
            self.selected.clear()
            self.selected.add(item)
            self.last_click = item

        self._update_selection_styles()

        if len(self.selected) == 1:
            selected_item = next(iter(self.selected))
            print(f'[browse] on_item_selected -> {selected_item} (is_file={selected_item.is_file()})')
            if selected_item.is_file():
                self.on_item_selected(selected_item)
            else:
                self.on_item_selected(None)
        else:
            self.on_item_selected(None)

        self._update_header_buttons()

    def _update_selection_styles(self):
        for it, card in self.card_map.items():
            card.set_selected(it in self.selected)

    def _select_programmatically(self, path: Path):
        if path in self.card_map:
            self.selected.clear()
            self.selected.add(path)
            self.last_click = path
            self._update_selection_styles()
            if path.is_file():
                self.on_item_selected(path)
            else:
                self.on_item_selected(None)
            # Ensure the newly selected item is visible
            card = self.card_map.get(path)
            if card:
                self.scroll.ensureWidgetVisible(card)

    def _set_path(self, path: Path):
        if not self._interactive:
            return
        self.state['current_path'] = str(path)
        self.render()

    def _go_up(self):
        if not self._interactive:
            return
        cur = Path(self.state.get('current_path', self.root_path))
        if cur != self.root_path and cur.parent.exists():
            self.state['current_path'] = str(cur.parent)
            self.render()

    def _select_all_files(self):
        self.selected.clear()
        for item in self.items:
            if item.is_file():
                self.selected.add(item)
        self._update_selection_styles()
        self.on_item_selected(None)

    def _delete_selected(self):
        if not self.selected:
            return

        deleted_count = 0
        for item in list(self.selected):
            if item.is_file():
                try:
                    item.unlink()
                    self.delete_thumbnail(item)
                    deleted_count += 1
                except Exception as e:
                    print(f'[browse] Delete error for {item.name}: {e}')

        # Always refresh silently; selection might contain non-files
        if deleted_count > 0:
            self.render()

    def _thumbsize_to_index(self, size: int) -> int:
        try:
            return THUMB_SIZES.index(size)
        except ValueError:
            return THUMB_SIZES.index(DEFAULT_THUMB_SIZE)

    def _on_size_radio(self, size: int):
        if self.state.get('thumb_size') != size:
            self.state['thumb_size'] = size
            self.render()

    def _sync_size_radios(self):
        size = int(self.state.get('thumb_size', DEFAULT_THUMB_SIZE))
        self.btn_s.setChecked(size == THUMB_S_PX)
        self.btn_m.setChecked(size == THUMB_M_PX)
        self.btn_l.setChecked(size == THUMB_L_PX)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        # Defer a relayout until after the widget is polished and sized by its parent/splitter
        QtCore.QTimer.singleShot(0, self._relayout)

    def _set_header_enabled(self, enabled: bool):
        # Enable/disable all header buttons
        for btn in [self.btn_up, self.btn_new, self.btn_edit, self.btn_delete, self.btn_select_all, self.btn_auto, self.btn_s, self.btn_m, self.btn_l, self.btn_llm]:
            btn.setEnabled(enabled)


def mount_browse(parent: QtWidgets.QWidget,
                 state: dict,
                 on_item_selected: Callable[[Path | None], None],
                 root_path: Path,
                 thumbnail_dir: Path,
                 on_edit_request: Callable[[], None]) -> BrowserView:
    view = BrowserView(state, on_item_selected, root_path, thumbnail_dir, on_edit_request, parent=parent)  # was parent=None
    if parent.layout() is None:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
    parent.layout().addWidget(view)
    # Ensure an initial relayout after the view is actually attached and sized
    QtCore.QTimer.singleShot(0, view._relayout)
    return view


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("qtBrowse (PyQt6)")

    start_path = Path("T:/")
    thumbs = start_path / "_thumbnails"
    app_state = {
        "current_path": str(start_path),
        "thumb_size": DEFAULT_THUMB_SIZE,
    }

    def on_sel(p: Optional[Path]):
        print("Selected:", p)

    def on_edit():
        QtWidgets.QMessageBox.information(None, "Edit", "Edit requested")

    window = QtWidgets.QMainWindow()
    central = QtWidgets.QWidget()
    window.setCentralWidget(central)

    mount_browse(central, app_state, on_sel, start_path, thumbs, on_edit)

    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())