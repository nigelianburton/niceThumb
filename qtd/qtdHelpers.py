from typing import Any, Dict, List, Optional, Tuple, Callable
import os
import base64
import math
from io import BytesIO
from PIL import Image

# Generic utilities shared by SDXL and Qwen backends

def list_safetensors(dir_path: str) -> List[str]:
    try:
        if not os.path.isdir(dir_path):
            return []
        return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".safetensors")])
    except Exception:
        return []

def resolve_in_dir(name_or_path: str, base_dir: str, default_ext: Optional[str] = ".safetensors") -> Optional[str]:
    if not isinstance(name_or_path, str):
        return None
    s = name_or_path.strip()
    if not s:
        return None
    if os.path.isabs(s) and os.path.isfile(s):
        return s
    if not isinstance(base_dir, str) or not base_dir:
        return None
    cand = os.path.join(base_dir, s)
    if os.path.isfile(cand):
        return cand
    if default_ext and not s.lower().endswith(default_ext.lower()):
        cand2 = os.path.join(base_dir, s + default_ext)
        if os.path.isfile(cand2):
            return cand2
    return None

def resolve_lora_list(loras: List[str], base_dir: str) -> List[str]:
    out: List[str] = []
    for n in (loras or []):
        p = resolve_in_dir(n, base_dir, default_ext=".safetensors")
        if isinstance(p, str) and os.path.isfile(p):
            out.append(p)
    return out

def seed_from_inputs(p: Dict[str, Any]) -> Optional[int]:
    try:
        if "generator" in p and p["generator"] is not None:
            return int(p["generator"])
    except Exception:
        pass
    try:
        if "seed" in p and p["seed"] is not None:
            return int(p["seed"])
    except Exception:
        pass
    return None

def decode_data_url_to_pil(data_url: str) -> Image.Image:
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("image is empty")
    if os.path.exists(data_url):
        return Image.open(data_url).convert("RGB")
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("must be a data URL or file path")
    try:
        _, b64 = data_url.split(",", 1)
        b64 = b64.strip()
        img = Image.open(BytesIO(base64.b64decode(b64)))
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot decode data URL: {e}")

def decode_and_resize_image(data_url: str, width: int, height: int) -> Image.Image:
    img = decode_data_url_to_pil(data_url)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img

def decode_and_resize_mask(mask_url: Optional[str], width: int, height: int) -> Optional[Image.Image]:
    if isinstance(mask_url, str) and mask_url:
        img = decode_data_url_to_pil(mask_url)
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img
    return None

def pil_to_data_url(img: Image.Image, fmt: str = "PNG", quality: int = 92) -> str:
    fmt = (fmt or "PNG").upper()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    buf = BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        save_kwargs.update(dict(quality=int(quality or 92), optimize=True))
    img.save(buf, format=fmt, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def compute_sdxl_size(src_w: int, src_h: int, target_area: int = 1024 * 1024, multiple: int = 8) -> Tuple[int, int]:
    w = max(1, int(src_w))
    h = max(1, int(src_h))
    ar = float(w) / float(h)
    tgt_h = math.sqrt(target_area / ar)
    tgt_w = ar * tgt_h
    out_w = max(multiple, int(round(tgt_w / multiple)) * multiple)
    out_h = max(multiple, int(round(tgt_h / multiple)) * multiple)
    return int(out_w), int(out_h)

def make_diffusers_progress_callback(set_progress: Callable[[int], None], steps: int):
    steps = max(1, int(steps or 1))
    def _cb(step, timestep, latents):
        try:
            pct = int((int(step) + 1) * 100 / steps)
            set_progress(max(0, min(100, pct)))
        except Exception:
            pass
    return _cb

def sanitize_adapter_name(name: str, idx: int) -> str:
    base = os.path.splitext(os.path.basename(name or f"lora{idx}"))[0]
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in base)
    return f"lora_{idx}_{safe or 'adapter'}"

def bf16_supported(torch_mod) -> bool:
    try:
        return bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def empty_cuda_cache(torch_mod) -> None:
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
    except Exception:
        pass

# ------------------------------------------------------------------
# Shared diffusion helpers (new)
# ------------------------------------------------------------------
def resize_to_area_preserve_aspect(image: Image.Image, target_area: int = 1024 * 1024, multiple: int = 4) -> Image.Image:
    """
    Deterministically resize an image so its area is ~ target_area while preserving aspect.
    Both width and height are rounded to the nearest 'multiple'.
    """
    if image is None:
        return image
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    area = w * h
    if area == target_area and w % multiple == 0 and h % multiple == 0:
        return image
    aspect = w / h
    import math
    scale = math.sqrt(target_area / float(area))
    ideal_w = w * scale
    def round_mult(v: float) -> int:
        return max(multiple, int(round(v / multiple)) * multiple)
    # Try a few candidates around ideal width
    candidates = []
    for off in (-multiple, 0, multiple):
        cw = round_mult(ideal_w + off)
        if cw <= 0:
            continue
        ch = round_mult(cw / aspect)
        if ch <= 0:
            continue
        cand_area = cw * ch
        candidates.append((abs(cand_area - target_area), abs((cw / ch) - aspect), cand_area, cw, ch))
    if not candidates:
        return image
    candidates.sort()
    _, _, _, nw, nh = candidates[0]
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.Resampling.LANCZOS)

def ensure_rgb(img: Image.Image) -> Image.Image:
    """Guarantee a 3‑channel RGB image."""
    if img is None:
        return img
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def patch_scheduler_progress(pipe, user_steps: int, set_progress: Callable[[int], None], base: int = 10):
    """
    Monkey-patch a diffusers scheduler.step to emit mapped progress.
    Returns a restore() function.
    Safe no-op if scheduler absent.
    """
    try:
        if pipe is None or not hasattr(pipe, "scheduler"):
            return lambda: None
        sched = getattr(pipe, "scheduler", None)
        if sched is None or not hasattr(sched, "step"):
            return lambda: None
        original_step = sched.step
        state = {"i": 0}
        def patched_step(model_output, timestep, *args, **kwargs):
            state["i"] += 1
            if user_steps > 0:
                frac = min(1.0, max(0.0, state["i"] / float(user_steps)))
                pct = base + int(frac * 89)
                if pct >= 100:
                    pct = 99
                set_progress(pct)
            return original_step(model_output, timestep, *args, **kwargs)
        sched.step = patched_step
        def restore():
            try:
                sched.step = original_step
            except Exception:
                pass
        return restore
    except Exception:
        return lambda: None