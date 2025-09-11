from PyQt6 import QtCore, QtGui, QtWidgets
from typing import List, Optional, Dict, Tuple

# Updated palette definition as a dictionary with categories, names, and RGB tuples
PALETTE: Dict[str, List[Tuple[str, Tuple[int, int, int]]]] = {
    "statement": [
        ("Crimson", (200, 0, 40)),
        ("Sapphire", (15, 80, 180)),
        ("Emerald", (0, 150, 90)),
        ("Tangerine", (255, 120, 0)),
        ("Electric Blue", (0, 180, 255)),
        ("Lime Zest", (180, 255, 60)),
        ("Orchid", (190, 80, 190)),
        ("Magenta", (255, 0, 150)),
    ],
    "basics": [
        ("Jet Black", (0, 0, 0)),
        ("Pure White", (255, 255, 255)),
        ("Charcoal", (60, 60, 60)),
        ("Steel Grey", (120, 120, 130)),
        ("Ink Blue", (30, 40, 80)),
        ("Teal", (0, 128, 128)),
        ("Canary", (255, 255, 100)),
        ("Ultramarine", (0, 0, 200)),
    ],
    "porcelain": [
        ("Alabaster", (255, 250, 240)),
        ("Ivory Veil", (245, 235, 220)),
        ("Rose Porcelain", (250, 230, 225)),
        ("Bisque", (240, 220, 200)),
        ("Linen Glow", (235, 225, 210)),
        ("Soft Sand", (220, 200, 180)),
        ("Pale Almond", (230, 210, 190)),
        ("Vanilla Frost", (250, 240, 220)),
    ],
    "sunlit": [
        ("Golden Tan", (210, 170, 130)),
        ("Bronze Kiss", (180, 130, 90)),
        ("Caramel Drizzle", (200, 140, 100)),
        ("Terracotta", (190, 110, 80)),
        ("Amber Glow", (220, 150, 90)),
        ("Cinnamon Spice", (160, 90, 60)),
        ("Tawny Buff", (170, 120, 90)),
        ("Desert Clay", (200, 160, 130)),
    ],
    "petals": [
        ("Ballet Slipper", (255, 220, 230)),
        ("Blush Bloom", (250, 200, 210)),
        ("Flamingo", (255, 160, 180)),
        ("Rose Quartz", (240, 180, 190)),
        ("Peony Pop", (255, 140, 160)),
        ("Dusty Pink", (220, 170, 180)),
        ("Coral Mist", (255, 180, 160)),
        ("Bubblegum", (255, 105, 180)),
    ],
    "powder": [
        ("Mint Cream", (200, 255, 230)),
        ("Lavender Fog", (220, 200, 255)),
        ("Baby Blue", (180, 220, 255)),
        ("Peach Sorbet", (255, 210, 180)),
        ("Lilac Haze", (200, 180, 230)),
        ("Seafoam", (180, 240, 220)),
        ("Powder Green", (190, 230, 200)),
        ("Sky Whisper", (210, 230, 250)),
    ],
    "golden": [
        ("Platinum Blonde", (230, 225, 210)),
        ("Champagne Mist", (240, 220, 180)),
        ("Honey Glaze", (215, 180, 90)),
        ("Buttermilk", (255, 245, 190)),
        ("Straw Gold", (235, 200, 100)),
        ("Marigold", (255, 185, 60)),
        ("Dijon Whisper", (200, 160, 60)),
        ("Saffron Silk", (255, 200, 90)),
    ],
    "earth": [
        ("Chestnut", (150, 90, 60)),
        ("Cocoa Bean", (120, 70, 50)),
        ("Espresso", (80, 50, 40)),
        ("Walnut Bark", (100, 70, 50)),
        ("Mocha Dust", (170, 120, 90)),
        ("Burnt Umber", (110, 60, 40)),
        ("Hazelnut Cream", (190, 140, 100)),
        ("Smoky Taupe", (160, 130, 110)),
    ]
}

class ToolPaletteWidget(QtWidgets.QGroupBox):
    colorSelected = QtCore.pyqtSignal(QtGui.QColor)

    def __init__(self, palette: Dict[str, List[Tuple[str, Tuple[int, int, int]]]], initial_color: Optional[QtGui.QColor] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Palette", parent)
        self.setStyleSheet("QGroupBox { border: none; }")  # <-- Remove border
        self._palette = palette
        self._color_buttons: List[QtWidgets.QPushButton] = []
        self._color_map: Dict[QtWidgets.QPushButton, QtGui.QColor] = {}
        self._active_color = initial_color
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        row = 0
        col_count = max(len(colors) for colors in palette.values())
        for category, colors in palette.items():
            # Add category label at the start of the row
            label = QtWidgets.QLabel(category.capitalize())
            label.setStyleSheet("font-size:11px; color:#444; padding-right:8px;")
            grid.addWidget(label, row, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)
            for col, (name, rgb) in enumerate(colors):
                color = QtGui.QColor(*rgb)
                btn = QtWidgets.QPushButton()
                btn.setFixedSize(18, 18)
                btn.setCheckable(True)
                btn.setStyleSheet(f"background:{color.name()}; border:1px solid #888;")
                btn.setToolTip(name)
                btn.clicked.connect(lambda _, c=color: self._on_color_clicked(c))
                grid.addWidget(btn, row, col + 1)
                self._color_buttons.append(btn)
                self._color_map[btn] = color
            row += 1

        # Set initial color
        if initial_color is not None:
            self.set_active_color(initial_color)
        else:
            # Default to first color in first category
            first_cat = next(iter(palette.values()))
            first_color = QtGui.QColor(*first_cat[0][1])
            self.set_active_color(first_color)

    def _on_color_clicked(self, color: QtGui.QColor):
        self.set_active_color(color)
        self.colorSelected.emit(color)

    def set_active_color(self, color: QtGui.QColor):
        self._active_color = color
        for btn, c in self._color_map.items():
            btn.setChecked(c.name().lower() == color.name().lower())
            if btn.isChecked():
                btn.setStyleSheet(f"background:{c.name()}; border:2px solid #3b82f6;")
            else:
                btn.setStyleSheet(f"background:{c.name()}; border:1px solid #888;")
    def active_color(self) -> QtGui.QColor:
        return self._active_color

def make_brush_cursor(size, color, border_color=None, border_width=2, cross=False):
    """
    Returns a QPixmap for the cursor preview.
    - size: int, diameter in pixels
    - color: QtGui.QColor
    - border_color: QtGui.QColor or None
    - border_width: int
    - cross: bool, if True, draws a cross over the circle
    """
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    if border_color:
        pen = QtGui.QPen(border_color)
        pen.setWidth(border_width)
        painter.setPen(pen)
    else:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(1, 1, size - 2, size - 2)
    if cross:
        pen_cross = QtGui.QPen(QtGui.QColor(0, 0, 0), 2)
        painter.setPen(pen_cross)
        painter.drawLine(size // 2, 0, size // 2, size)
        painter.drawLine(0, size // 2, size, size // 2)
    painter.end()
    return pm

def make_circular_patch(image, center, diameter, border_color=None, border_width=2):
    """
    Returns a QImage masked to a circle, optionally with border.
    - image: QtGui.QImage
    - center: QtCore.QPointF or QtCore.QPoint
    - diameter: int
    - border_color: QtGui.QColor or None
    - border_width: int
    """
    patch = image.copy(
        int(center.x() - diameter / 2),
        int(center.y() - diameter / 2),
        int(diameter),
        int(diameter)
    )
    circ = QtGui.QImage(int(diameter), int(diameter), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    circ.fill(0)
    painter = QtGui.QPainter(circ)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QtGui.QBrush(QtGui.QPixmap.fromImage(patch.scaled(int(diameter), int(diameter)))))
    if border_color:
        painter.setPen(QtGui.QPen(border_color, border_width))
        painter.drawEllipse(border_width // 2, border_width // 2, int(diameter) - border_width, int(diameter) - border_width)
    else:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, int(diameter), int(diameter))
    painter.end()
    return circ

def get_composed_image(canvas):
    """
    Returns the composed image (background + overlay) from a canvas.
    """
    if hasattr(canvas, "_pixmap") and hasattr(canvas, "_overlay") and canvas._pixmap and canvas._overlay:
        base = canvas._pixmap.toImage()
        overlay = canvas._overlay
        composed = QtGui.QImage(base.size(), base.format())
        composed.fill(0)
        painter = QtGui.QPainter(composed)
        painter.drawImage(0, 0, base)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(0, 0, overlay)
        painter.end()
        return composed
    elif hasattr(canvas, "_pixmap") and canvas._pixmap:
        return canvas._pixmap.toImage()
    return None

def get_brush_geometry(canvas, pos, brush_size):
    """
    Returns scale, diameter, radius, and top_left for brush operations.
    - canvas: PaintCanvas
    - pos: QtCore.QPoint
    - brush_size: int
    """
    frame = canvas._fit_rect() if hasattr(canvas, "_fit_rect") else None
    pixmap = getattr(canvas, "_pixmap", None)
    if frame is None or pixmap is None or pixmap.isNull():
        return None, None, None, None
    scale = pixmap.width() / frame.width() if frame.width() > 0 else 1.0
    dia_img = max(1.0, brush_size * scale)
    rad_img = dia_img / 2.0
    img_pt = canvas._map_widget_to_image(pos) if hasattr(canvas, "_map_widget_to_image") else None
    if img_pt is None:
        return scale, dia_img, rad_img, None
    top_left = QtCore.QPoint(int(img_pt.x() - rad_img), int(img_pt.y() - rad_img))
    return scale, dia_img, rad_img, top_left

# --- New / Shared Blur Utilities -------------------------------------------------

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def blur_qimage_gaussian(img: QtGui.QImage, strength: float) -> QtGui.QImage:
    """
    Shared blur utility:
    - Uses OpenCV Gaussian blur if available.
    - Falls back to QGraphicsBlurEffect otherwise.
    Strength meaning:
      * Interpreted as a "radius" (>=1) for kernel sizing or Qt blur radius.
    """
    if img is None or img.isNull():
        return img

    strength = max(1.0, float(strength))

    if _HAS_CV2:
        # Convert QImage -> numpy BGRA
        src = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        ptr = src.bits()
        ptr.setsize(src.sizeInBytes())
        import numpy as np  # local import; existing dependency through blur tool
        arr = np.array(ptr, dtype=np.uint8).reshape(src.height(), src.width(), 4)

        ksize = int(strength)
        # Ensure odd and >=3
        ksize = max(3, ksize * 2 + 1)
        blurred = cv2.GaussianBlur(arr, (ksize, ksize), 0)

        out = QtGui.QImage(
            blurred.data, blurred.shape[1], blurred.shape[0],
            src.bytesPerLine(), QtGui.QImage.Format.Format_ARGB32_Premultiplied
        ).copy()
        return out

    # Fallback: Qt blur effect
    pm = QtGui.QPixmap.fromImage(img)
    scene = QtWidgets.QGraphicsScene()
    item = QtWidgets.QGraphicsPixmapItem(pm)
    blur = QtWidgets.QGraphicsBlurEffect()
    blur.setBlurRadius(strength)
    item.setGraphicsEffect(blur)
    scene.addItem(item)
    result = QtGui.QImage(pm.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    result.fill(0)
    painter = QtGui.QPainter(result)
    scene.render(painter)
    painter.end()
    return result