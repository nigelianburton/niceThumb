from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui


def aspect_fit_rect(container: QtCore.QRect, img_w: int, img_h: int) -> QtCore.QRectF:
    """Compute a QRectF inside container that preserves image aspect and fills as much as possible."""
    if img_w <= 0 or img_h <= 0 or container.isEmpty():
        return QtCore.QRectF(container)
    w = container.width()
    h = container.height()
    scale = min(w / img_w, h / img_h)
    dw = img_w * scale
    dh = img_h * scale
    x = container.x() + (w - dw) / 2.0
    y = container.y() + (h - dh) / 2.0
    return QtCore.QRectF(x, y, dw, dh)


def widget_to_image_point(widget_pt: QtCore.QPoint, frame_rect: QtCore.QRectF, img_w: int, img_h: int) -> Optional[QtCore.QPointF]:
    """Map a point from widget space into image pixel space using the fitted frame rect."""
    if not frame_rect.contains(QtCore.QPointF(widget_pt)) or img_w <= 0 or img_h <= 0:
        return None
    sx = img_w / frame_rect.width() if frame_rect.width() > 0 else 1.0
    sy = img_h / frame_rect.height() if frame_rect.height() > 0 else 1.0
    x_img = (widget_pt.x() - frame_rect.x()) * sx
    y_img = (widget_pt.y() - frame_rect.y()) * sy
    return QtCore.QPointF(x_img, y_img)


def merged_patch(base_pixmap: QtGui.QPixmap, overlay: Optional[QtGui.QImage], center_img: QtCore.QPointF, diameter_img: float) -> QtGui.QImage:
    """Return merged base(image)+overlay patch (image space) centered at center_img with given diameter."""
    if base_pixmap.isNull() or diameter_img <= 0:
        return QtGui.QImage()
    base = base_pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    r = int(max(1.0, diameter_img))
    x = int(center_img.x() - r / 2)
    y = int(center_img.y() - r / 2)
    rect = QtCore.QRect(x, y, r, r).intersected(base.rect())
    if rect.isEmpty():
        return QtGui.QImage()
    base_roi = base.copy(rect)
    if overlay is not None and not overlay.isNull():
        ov_roi = overlay.copy(rect)
        p = QtGui.QPainter(base_roi)
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        p.drawImage(0, 0, ov_roi)
        p.end()
    return base_roi


# High-quality blur with OpenCV (falls back if not available)
try:
    import numpy as _np  # type: ignore
    import cv2 as _cv2   # type: ignore
except Exception:
    _np = None
    _cv2 = None

def fast_blur_qimage(img: QtGui.QImage, strength: float) -> QtGui.QImage:
    """High-quality Gaussian blur. Uses OpenCV; falls back to identity if unavailable.
    strength range expected ~0.5..4.0 from UI.
    """
    if img is None or img.isNull():
        return img
    if _np is None or _cv2 is None:
        return img  # fallback: no-op to avoid over-blur in absence of OpenCV

    src = img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h, bpl = src.width(), src.height(), src.bytesPerLine()
    ptr = src.bits(); ptr.setsize(h * bpl)
    arr = _np.frombuffer(ptr, _np.uint8).reshape((h, bpl // 4, 4))[:, :w, :]
    rgb = arr[:, :, :3]
    a = arr[:, :, 3:4]

    # Keep sigma gentle so min feels small (~0.5 px). Empirical curve:
    # slider strength s in [0.5..4.0] -> sigma ~= 0.12 + 0.38*(s-0.5)
    s = max(0.5, float(strength))
    sigma = 0.12 + 0.38 * (s - 0.5)  # min ~0.12, max ~1.52
    blurred = _cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=_cv2.BORDER_DEFAULT)
    out = _np.concatenate([blurred, a], axis=2)
    qimg = QtGui.QImage(out.data, w, h, w * 4, QtGui.QImage.Format.Format_RGBA8888)
    return qimg.copy()


def circular_mask(img: QtGui.QImage) -> QtGui.QImage:
    """Apply a circular alpha mask to the given image and return it."""
    if img.isNull():
        return img
    size = img.size()
    mask = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    mask.fill(0)
    p = QtGui.QPainter(mask)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 255)))
    p.drawEllipse(QtCore.QRectF(0, 0, size.width(), size.height()))
    p.end()
    res = img.copy()
    p2 = QtGui.QPainter(res)
    p2.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_DestinationIn)
    p2.drawImage(0, 0, mask)
    p2.end()
    return res


def compose_images(base: QtGui.QImage, overlay: Optional[QtGui.QImage]) -> QtGui.QImage:
    """Return base composited with overlay (resized if needed). Does not mutate inputs."""
    if base.isNull():
        return base
    result = base.copy()
    if overlay is None or overlay.isNull():
        return result
    if overlay.size() != base.size():
        overlay_use = overlay.scaled(base.size(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                     QtCore.Qt.TransformationMode.SmoothTransformation)
    else:
        overlay_use = overlay
    p = QtGui.QPainter(result)
    p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
    p.drawImage(0, 0, overlay_use)
    p.end()
    return result


def draw_brush_preview(sprite: Optional[QtGui.QImage], preview_size: int, ring_color: QtGui.QColor = QtGui.QColor("yellow")) -> QtGui.QPixmap:
    """Build a preview pixmap with optional sprite centered and a ring overlay."""
    pm = QtGui.QPixmap(preview_size, preview_size)
    pm.fill(QtGui.QColor("#ffffff"))
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
    if sprite is not None and not sprite.isNull():
        pad = 6
        size = max(1, preview_size - pad)
        target = QtCore.QRect((preview_size - size) // 2, (preview_size - size) // 2, size, size)
        p.drawImage(target, sprite)
    pen = QtGui.QPen(ring_color)
    pen.setWidth(2)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawEllipse(2, 2, preview_size - 4, preview_size - 4)
    p.end()
    return pm


def next_edit_filename(src: Path) -> Path:
    """Return a unique sibling filename with _editN suffix."""
    parent = src.parent
    stem = src.stem
    suffix = src.suffix
    import re
    m = re.search(r'(?:_edit(\d*)$)', stem)
    if m:
        base = stem[:m.start()]
        num_str = m.group(1)
        start = int(num_str) + 1 if num_str else 1
    else:
        base = stem
        start = 1
    n = start
    while True:
        candidate = parent / f"{base}_edit{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1