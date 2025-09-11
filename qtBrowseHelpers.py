from __future__ import annotations
from pathlib import Path
import os


def safe_name(root: Path, file_path: Path) -> str:
    """Create a filesystem-safe name for a thumbnail file based on path relative to root."""
    try:
        rel = file_path.relative_to(root)
        rel_str = str(rel)
    except Exception:
        # Fallback if item is outside root or on different drive
        rel_str = file_path.name
    return rel_str.replace(os.sep, '~_')


def thumbnail_path(thumbnail_dir: Path, root: Path, file_path: Path) -> Path:
    """Return the destination path of the thumbnail file for the given item."""
    return thumbnail_dir / f"{safe_name(root, file_path)}.jpg"


def resolve_folder_proxy(file_path: Path) -> Optional[Path]:
    """If the path is a directory, return folder.jpg/png inside it when available."""
    if file_path.is_dir():
        for name in ('folder.jpg', 'folder.png'):
            p = file_path / name
            if p.exists():
                return p
        return None
    return file_path


def _render_image_thumbnail(src: Path, dst: Path, thumb_max: int) -> None:
    with Image.open(src) as img:
        img.thumbnail((thumb_max, thumb_max))
        img.convert('RGB').save(dst, 'JPEG')


def _render_video_thumbnail(src: Path, dst: Path, thumb_max: int) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {src}')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        target_frame = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    if not ok or frame is None:
        dur_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
        if dur_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, dur_ms / 2.0)
            ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f'Failed to read frame: {src}')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img.thumbnail((thumb_max, thumb_max))
    img.convert('RGB').save(dst, 'JPEG')
    cap.release()


def ensure_thumbnail(root: Path,
                     thumbnail_dir: Path,
                     file_path: Path,
                     thumb_max: int = 256,
                     force: bool = False) -> Optional[Path]:
    """
    Ensure a thumbnail exists for file_path and return its Path, or None if unsupported.
    - Directories use folder.jpg/png when present; otherwise return None.
    - Images are thumbnailed via Pillow; videos via OpenCV.
    """
    proxy = resolve_folder_proxy(file_path)
    if proxy is None:
        return None

    suffix = proxy.suffix.lower()
    dst = thumbnail_path(thumbnail_dir, root, file_path)

    if dst.exists() and not force:
        return dst

    if suffix in IMAGE_SUFFIXES:
        _render_image_thumbnail(proxy, dst, thumb_max)
        return dst

    if suffix in VIDEO_SUFFIXES:
        _render_video_thumbnail(proxy, dst, thumb_max)
        return dst

    return None