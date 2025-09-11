"""
qtDiffusion.py
Headless HTTP service for image generation/editing used by NiceThumb.

Purpose
- Serve Stable Diffusion XL (SDXL) text-to-image (T2I), image-to-image (I2I, including inpaint) and Qwen Image Edit over localhost.
- Run generations asynchronously with job progress and Server-Sent Events (SSE).
- Lazy-load and switch between SDXL and Qwen models, managing GPU/CPU memory.

Key Endpoints (all generation is async)
- GET  /ping                      : Health and device status.
- GET  /models                    : List available SDXL model files (.safetensors).
- GET  /loras                     : List available LoRA files (.safetensors).
- POST /switch_character          : Build or set a reference image used as character anchor.
- POST /t2i_async                 : SDXL text-to-image job; returns job_id.
- POST /i2i_async                 : SDXL image-to-image/inpaint job; returns job_id.
- POST /qwen_edit_async           : Qwen Image Edit job; returns job_id.
- GET  /progress/<job_id>         : Poll job status and result when done.
- GET  /events/<job_id>           : Progress/result via SSE stream.

Notes
- Synchronous /t2i and /i2i endpoints are intentionally disabled; use the async variants.
- Models/paths are configurable via environment variables (see NT6_* variables below).
"""

# nt6diffusion.py - Headless SDXL + IP-Adapter HTTP service (no Tk UI)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import os
import time
import threading
import queue
import base64
import math
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, Response, stream_with_context

import uuid
import sys

# Toggle for verbose per-request logging
detailed_logging: bool = False

# Track currently attached LoRA names for SDXL text and img2img
_current_sdxl_lora_names_text: List[str] = []
_current_sdxl_lora_names_i2i: List[str] = []

def _sanitize_adapter_name(name: str, idx: int) -> str:
    """Make a safe, unique adapter name from a file base name."""
    base = os.path.splitext(os.path.basename(name or f"lora{idx}"))[0]
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in base)
    return f"lora_{idx}_{safe or 'adapter'}"

def _apply_sdxl_loras(pipeline, lora_names: Optional[List[str]], *, which: str) -> None:
    """
    Load and activate multiple LoRAs on the given SDXL pipeline.
    - lora_names: list of file names or absolute paths; resolved via LORAS_DIR if not absolute.
    - which: 'text' for StableDiffusionXLPipeline, 'i2i' for StableDiffusionXLImg2ImgPipeline.
    """
    global _current_sdxl_lora_names_text, _current_sdxl_lora_names_i2i

    def _set_current(names: List[str]):
        if which == "text":
            _current_sdxl_lora_names_text = names
        else:
            _current_sdxl_lora_names_i2i = names

    # Unload current adapters first
    try:
        if hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()
    except Exception:
        pass

    if not lora_names:
        _set_current([])
        return

    resolved_paths: List[str] = []
    resolved_basenames: List[str] = []
    for name in lora_names:
        if not isinstance(name, str) or not name.strip():
            continue
        path = _resolve_lora_path(name)
        if path and os.path.isfile(path):
            resolved_paths.append(path)
            resolved_basenames.append(os.path.basename(path))

    if not resolved_paths:
        _set_current([])
        return

    adapter_names: List[str] = []
    for idx, path in enumerate(resolved_paths):
        adapter = _sanitize_adapter_name(path, idx)
        try:
            pipeline.load_lora_weights(path, adapter_name=adapter)
            adapter_names.append(adapter)
        except Exception as e:
            print(f"[qtDiffusion][sdxl_lora] load failed ({which}): {path} -> {e}")

    if not adapter_names:
        _set_current([])
        return

    try:
        weights = [1.0] * len(adapter_names)
        pipeline.set_adapters(adapter_names, adapter_weights=weights)
    except Exception as e:
        print(f"[qtDiffusion][sdxl_lora] set_adapters failed ({which}): {e}")
        try:
            pipeline.set_adapters(adapter_names[0], adapter_weights=[1.0])
            resolved_basenames = [resolved_basenames[0]]
        except Exception:
            resolved_basenames = []

    _set_current(resolved_basenames)

def _log_generation(kind: str,
                    model_file: Optional[str],
                    prompt: str,
                    *,
                    steps: Optional[int] = None,
                    guidance: Optional[float] = None,
                    strength: Optional[float] = None,
                    seed: Optional[int] = None,
                    true_cfg_scale: Optional[float] = None) -> None:
    """Structured logging for generation requests (guarded by 'detailed_logging')."""
    if not detailed_logging:
        return
    def _fmt(v):
        return "-" if v is None or v == "" else v
    try:
        q = repr(prompt)
    except Exception:
        q = "<unprintable>"
    print(
        f"[qtDiffusion][gen:{kind}] "
        f"model={_fmt(model_file)} steps={_fmt(steps)} guidance={_fmt(guidance)} strength={_fmt(strength)} "
        f"true_cfg_scale={_fmt(true_cfg_scale)} seed={_fmt(seed)} query={q}"
    )
    # New line: indicate active LoRA(s)
    try:
        if ACTIVE_KIND == 'qwen':
            p = _current_qwen_lora_path
            names = [os.path.basename(p)] if isinstance(p, str) and p.strip() else []
        elif ACTIVE_KIND == 'sdxl':
            if kind == "i2i_async":
                names = list(_current_sdxl_lora_names_i2i)
            else:
                names = list(_current_sdxl_lora_names_text)
        else:
            names = []
    except Exception:
        names = []
    print(f"[qtDiffusion][gen:{kind}] LoRA(s): {', '.join(names) if names else 'NO LORA'}")

FILE_PATH = os.path.abspath(__file__)
print(f"[qtDiffusion] starting: {FILE_PATH}")

# Global state for pipelines and libs (lazy-loaded)
pipe = None
pipe_i2i = None
pipe_inpaint = None
scheduler_cls = None
torch = None
diffusers = None
qwen_pipe = None

# Track currently attached LoRA for SDXL text and img2img
_current_sdxl_lora_path_text: Optional[str] = None
_current_sdxl_lora_path_i2i: Optional[str] = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(BASE_DIR, "storybook_pages")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Configuration via env vars
MODEL_PATH = os.environ.get("NT6_SDXL_MODEL", r"C:\_MODELS-SD\StableDiffusion\juggernautXL_ragnarokBy.safetensors")
IP_ADAPTER_PATH = os.environ.get("NT6_IPADAPTER", r"C:\_MODELS-SD\IpAdaptersXl\ip-adapter-plus-face_sdxl_vit-h.safetensors")
IMAGE_ENCODER_DIR = os.environ.get("NT6_IPADAPTER_ENCODER", r"C:\_MODELS-SD\ClipVision\ip_adapter_sdxl_image_encoder")
FONT_PATH = os.environ.get("NT6_FONT_PATH", r"C:\Windows\Fonts\arial.ttf")
MAX_PAGES = int(os.environ.get("NT6_MAX_PAGES", "5"))
PORT = int(os.environ.get("NT6DIFF_PORT", "5015"))

# Fixed Qwen model config (ignore model_file for Qwen; use LoRA if provided)
QWEN_MODEL_NAME = os.environ.get("NT6_QWEN_MODEL", r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit")
QWEN_LORA_PATH = os.environ.get("NT6_QWEN_LORA", "").strip()

CANVAS = {
    "image_width": int(os.environ.get("NT6_IMG_W", "768")),
    "image_height": int(os.environ.get("NT6_IMG_H", "768")),
    "text_panel_width": int(os.environ.get("NT6_TEXT_W", "600")),
    "padding": 20,
    "line_spacing": 6,
    "min_font": 24,
    "max_font": 250,
}

SD_PARAMS = {
    "steps": int(os.environ.get("NT6_STEPS", "30")),
    "cfg_scale": float(os.environ.get("NT6_CFG", "7.5")),
    "negative_prompt": os.environ.get("NT6_NEG_PROMPT", "blurry, lowres, deformed, extra limbs, bad anatomy, watermark, text"),
}

device = "cpu"
ip_adapter_enabled = False
current_character: Optional[Dict[str, Any]] = None
_current_qwen_lora_path: Optional[str] = None
ACTIVE_KIND: Optional[str] = None  # 'sdxl' or 'qwen'

jobs_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
app = Flask(__name__)
_init_lock = threading.Lock()
_current_model_path_text: Optional[str] = None
_current_model_path_i2i: Optional[str] = None

SD_MODELS_DIR = os.environ.get("NT6_SDXL_DIR", r"C:\_MODELS-SD\StableDiffusion")
LORAS_DIR = os.environ.get("NT6_LORAS_DIR", r"C:\_MODELS-SD\Lora")

def _list_safetensors(dir_path: str) -> list[str]:
    """List all .safetensors files in a directory (returns empty list on error)."""
    try:
        if not os.path.isdir(dir_path):
            return []
        return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".safetensors")])
    except Exception:
        return []

def _load_heavy():
    """Lazy import torch, diffusers and the default scheduler class."""
    global torch, diffusers, scheduler_cls
    if torch is None:
        import torch as _torch; globals()["torch"] = _torch
    if diffusers is None:
        import diffusers as _diff; globals()["diffusers"] = _diff
    if scheduler_cls is None:
        from diffusers import DPMSolverMultistepScheduler as _DPM; globals()["scheduler_cls"] = _DPM

def _cuda_empty_cache():
    """Empty CUDA cache if available; ignore any errors."""
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _unload_sdxl():
    """Release SDXL pipelines and mark SDXL as inactive to free VRAM."""
    global pipe, pipe_i2i, _current_model_path_text, _current_model_path_i2i, ACTIVE_KIND
    with _init_lock:
        pipe = None
        pipe_i2i = None
        _current_model_path_text = None
        _current_model_path_i2i = None
        if ACTIVE_KIND == 'sdxl':
            ACTIVE_KIND = None
        _cuda_empty_cache()

def _unload_qwen():
    """Release Qwen pipeline and mark Qwen as inactive to free VRAM."""
    global qwen_pipe, _current_qwen_lora_path, ACTIVE_KIND
    with _init_lock:
        qwen_pipe = None
        _current_qwen_lora_path = None
        if ACTIVE_KIND == 'qwen':
            ACTIVE_KIND = None
        _cuda_empty_cache()

def _bf16_supported() -> bool:
    """Return True if CUDA bfloat16 is supported by the current device."""
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def _model_dir():
    """Return the directory that contains the default SDXL model."""
    return os.path.dirname(MODEL_PATH) if MODEL_PATH else r"C:\_MODELS-SD\StableDiffusion"

def _resolve_model_path(model_file: str) -> str:
    """Resolve a model filename to an absolute path using the SDXL model directory as fallback."""
    if not model_file:
        return MODEL_PATH
    if os.path.isabs(model_file) and os.path.exists(model_file):
        return model_file
    cand = os.path.join(_model_dir(), model_file)
    return cand

def _resolve_lora_path(lora_name: Optional[str]) -> Optional[str]:
    """Resolve a LoRA name or path to an absolute file path under LORAS_DIR (or absolute if given)."""
    if not isinstance(lora_name, str):
        return None
    name = lora_name.strip()
    if not name:
        return None
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    cand = os.path.join(LORAS_DIR, name)
    if os.path.isfile(cand):
        return cand
    if not name.lower().endswith(".safetensors"):
        cand2 = os.path.join(LORAS_DIR, name + ".safetensors")
        if os.path.isfile(cand2):
            return cand2
    return None

def _apply_or_unload_sdxl_lora(pipeline, new_lora_path: Optional[str], *, which: str) -> None:
    """
    Apply (or unload) SDXL LoRA on a given pipeline.
    which: 'text' for StableDiffusionXLPipeline, 'i2i' for StableDiffusionXLImg2ImgPipeline.
    """
    global _current_sdxl_lora_path_text, _current_sdxl_lora_path_i2i
    current = _current_sdxl_lora_path_text if which == "text" else _current_sdxl_lora_path_i2i
    try:
        if new_lora_path and os.path.isfile(new_lora_path):
            if current != new_lora_path:
                try:
                    if hasattr(pipeline, "unload_lora_weights"):
                        pipeline.unload_lora_weights()
                except Exception:
                    pass
                pipeline.load_lora_weights(new_lora_path)
                try:
                    pipeline.set_adapters("default", 1.0)
                except Exception:
                    pass
                if which == "text":
                    _current_sdxl_lora_path_text = new_lora_path
                else:
                    _current_sdxl_lora_path_i2i = new_lora_path
        else:
            # No LoRA specified: unload existing if any
            if current:
                try:
                    if hasattr(pipeline, "unload_lora_weights"):
                        pipeline.unload_lora_weights()
                except Exception:
                    pass
                if which == "text":
                    _current_sdxl_lora_path_text = None
                else:
                    _current_sdxl_lora_path_i2i = None
    except Exception as e:
        print(f"[qtDiffusion][sdxl_lora] apply/unload failed ({which}): {e}")

# ---------------- Shared Helper Methods ----------------

def _init_pipeline(model_path: str, pipeline_cls, global_pipe_name: str, global_model_path_name: str):
    """
    Load/initialize a diffusers pipeline (by file) if not already loaded for the same model.
    - model_path: absolute path to a .safetensors (SDXL) model file.
    - pipeline_cls: diffusers pipeline class (or its name).
    - global_pipe_name/global_model_path_name: names of globals to update.
    Ensures only one of SDXL/Qwen is active at a time.
    """
    global device, scheduler_cls, ACTIVE_KIND
    with _init_lock:
        _load_heavy()  # Must run first (imports + scheduler)
        if isinstance(pipeline_cls, str):
            pipeline_cls = getattr(diffusers, pipeline_cls)
        pipe_obj = globals().get(global_pipe_name)
        current_model_path = globals().get(global_model_path_name)
        if pipe_obj is not None and current_model_path == model_path and ACTIVE_KIND == 'sdxl':
            return
        if qwen_pipe is not None or ACTIVE_KIND == 'qwen':
            _unload_qwen()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[qtDiffusion] Loading new model: {model_path} (pipeline: {pipeline_cls.__name__}, device: {device})")
        p = pipeline_cls.from_single_file(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        p.scheduler = scheduler_cls.from_config(p.scheduler.config)
        diffusers.utils.logging.set_verbosity_error()
        p = p.to(device)
        p.set_progress_bar_config(disable=True)
        globals()[global_pipe_name] = p
        globals()[global_model_path_name] = model_path
        ACTIVE_KIND = 'sdxl'


def log_prompt_request(kind: str, prompt: str, params: Optional[dict] = None):
    """Lightweight, always-on prompt logging for key parameters."""
    try:
        q = repr(prompt)
    except Exception:
        q = "<unprintable>"
    print(f"[qtDiffusion][{kind}] prompt={q} params={params or {}}")

def _decode_data_url_to_pil(data_url: str) -> Image.Image:
    """Decode a data URL or file path to a PIL RGB image."""
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("init_image is empty")
    if os.path.exists(data_url):
        return Image.open(data_url).convert("RGB")
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("init_image must be a data URL or file path")
    try:
        header, b64 = data_url.split(",", 1)
        b64 = b64.strip()
        img = Image.open(BytesIO(base64.b64decode(b64)))
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot decode data URL: {e}")

def _decode_and_resize_image(data_url: str, width: int, height: int) -> Image.Image:
    """Decode a data URL or file path to a PIL image and resize to (width, height)."""
    img = _decode_data_url_to_pil(data_url)
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img

def _decode_and_resize_mask(mask_url: str, width: int, height: int) -> Optional[Image.Image]:
    """Decode and resize mask image if provided; return None if mask_url is falsy."""
    if isinstance(mask_url, str) and mask_url:
        mask_img = _decode_data_url_to_pil(mask_url)
        if mask_img.size != (width, height):
            mask_img = mask_img.resize((width, height), Image.Resampling.LANCZOS)
        return mask_img
    return None

def _make_progress_callback(job_id: str, steps: int):
    """Create a callback that updates job progress based on pipeline callback invocations."""
    def _cb(step, timestep, latents):
        _set_progress(job_id, int((step + 1) * 100 / steps))
    return _cb

def _handle_job_result(job_id: str, result: Any, error: Exception = None):
    """Store job result as a data URL or record an error."""
    if error is not None:
        _set_error(job_id, f"{type(error).__name__}: {error}")
    else:
        _set_result(job_id, _pil_to_data_url(result))

def _seed_from_payload(p: Dict[str, Any]) -> Optional[int]:
    """Extract seed from 'generator' (preferred) or 'seed' if present."""
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

# ---------------- Qwen Pipeline ----------------

def _init_qwen_pipe(lora_path: Optional[str]):
    """
    Initialize Qwen Image Edit pipeline; unload SDXL if needed; optionally attach a LoRA.
    Uses dtype based on device (bf16/float16/float32), enables memory optimizations where possible.
    """
    global qwen_pipe, device, _current_qwen_lora_path, ACTIVE_KIND
    with _init_lock:
        if pipe is not None or pipe_i2i is not None or ACTIVE_KIND == 'sdxl':
            _unload_sdxl()
        _load_heavy()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            dtype = torch.bfloat16 if _bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        model_loc = QWEN_MODEL_NAME  # Expected to be defined/configured elsewhere
        local_only = os.path.isdir(model_loc)
        if qwen_pipe is None:
            from diffusers import QwenImageEditPipeline
            print(f"[nt6diffusion] Loading Qwen Image Edit '{model_loc}' on {device} (bf16_supported={_bf16_supported()}, torch_dtype={dtype}, local_only={local_only})...")
            qwen_pipe = QwenImageEditPipeline.from_pretrained(
                model_loc if local_only else "Qwen/Qwen-Image-Edit",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
            )
            try:
                qwen_pipe = qwen_pipe.to(device)
            except Exception as e:
                print(f"[nt6diffusion] to({device}) failed: {e}")
            try:
                qwen_pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            try:
                if hasattr(qwen_pipe, "vae") and qwen_pipe.vae is not None:
                    try: qwen_pipe.vae.enable_tiling()
                    except Exception: pass
                    try: qwen_pipe.vae.enable_slicing()
                    except Exception: pass
            except Exception:
                pass
            try:
                from diffusers.utils import is_xformers_available
                if is_xformers_available():
                    qwen_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                qwen_pipe.enable_model_cpu_offload()
            except Exception:
                try: qwen_pipe.enable_sequential_cpu_offload()
                except Exception: pass
            ACTIVE_KIND = 'qwen'
        lora_path = (lora_path or QWEN_LORA_PATH or "").strip()
        if lora_path and os.path.isfile(lora_path) and lora_path != _current_qwen_lora_path:
            print(f"[nt6diffusion] Attaching Qwen Lightning LoRA: {lora_path}")
            try:
                try:
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception: pass
                qwen_pipe.load_lora_weights(lora_path)
                try: qwen_pipe.set_adapters("default", 1.0)
                except Exception: pass
                _current_qwen_lora_path = lora_path
            except Exception as e:
                print(f"[nt6diffusion] load_lora_weights failed: {e}")

# ---------------- Routes (all generation is async) ----------------

@app.route("/ping")
def ping():
    """Health endpoint; returns device info and whether SDXL pipe is ready."""
    ok = True; msg = "ok"
    try: _ = pipe is not None
    except Exception as e: ok = False; msg = f"init_check_failed: {e}"
    return jsonify({"status": "ok" if ok else "error", "device": device, "ip_adapter_enabled": ip_adapter_enabled, "pipe_ready": pipe is not None, "message": msg})

@app.route("/models", methods=["GET"])
def api_list_models():
    """List SDXL models available in NT6_SDXL_DIR."""
    return jsonify({"models": _list_safetensors(SD_MODELS_DIR)})

@app.route("/loras", methods=["GET"])
def api_list_loras():
    """List LoRA files available in NT6_LORAS_DIR."""
    return jsonify({"loras": _list_safetensors(LORAS_DIR)})

# /switch_character
@app.route("/switch_character", methods=["POST"])
def api_switch_character():
    try:
        data = request.get_json(force=True)
    except Exception:
        return _json_error("Invalid JSON", 400)
    tokens = data.get("tokens")
    ref_path = data.get("reference_image_path") or ""
    try:
        weight = float(data.get("weight", 0.6))
    except Exception:
        return _json_error("'weight' must be a number", 400)
    seed = data.get("seed")
    if not (isinstance(tokens, list) and all(isinstance(t, str) and t.strip() for t in tokens)):
        return _json_error("tokens must be non-empty list[str]", 400)
    _init_pipeline(_resolve_model_path(os.path.basename(MODEL_PATH)),
                   pipeline_cls=diffusers.StableDiffusionXLPipeline,
                   global_pipe_name='pipe',
                   global_model_path_name='_current_model_path_text')
    if ref_path:
        if not os.path.exists(ref_path):
            return _json_error(f"reference_image_path not found: {ref_path}", 400)
        ref_img = Image.open(ref_path).convert("RGB")
    else:
        gen = torch.Generator(device=device).manual_seed(seed) if isinstance(seed, int) else None
        ref_img = pipe(
            prompt=", ".join(tokens),
            negative_prompt=SD_PARAMS["negative_prompt"],
            width=CANVAS["image_width"],
            height=CANVAS["image_height"],
            num_inference_steps=SD_PARAMS["steps"],
            guidance_scale=SD_PARAMS["cfg_scale"],
            generator=gen,
        ).images[0]
        try:
            ref_img.save(os.path.join(SAVE_FOLDER, f"char_ref_{int(time.time())}.png"), "PNG")
        except Exception:
            pass
    global current_character
    current_character = {"tokens": tokens, "image": ref_img, "weight": weight}
    return jsonify({"status": "ok", "tokens": tokens, "weight": weight, "seed": seed})

# /t2i_async
@app.route("/t2i_async", methods=["POST"])
def api_t2i_async():
    try:
        data = request.get_json(force=True)
    except Exception:
        return _json_error("Invalid JSON", 400)
    jid = _new_job()
    threading.Thread(target=_run_t2i_job, args=(jid, data), daemon=True).start()
    return jsonify({"status": "accepted", "job_id": jid}), 202

# /i2i_async
@app.route("/i2i_async", methods=["POST"])
def api_i2i_async():
    try:
        data = request.get_json(force=True)
    except Exception:
        return _json_error("Invalid JSON", 400)
    jid = _new_job()
    threading.Thread(target=_run_i2i_job, args=(jid, data), daemon=True).start()
    return jsonify({"status": "accepted", "job_id": jid}), 202

# /qwen_edit_async
@app.route("/qwen_edit_async", methods=["POST"])
def api_qwen_edit_async():
    try:
        data = request.get_json(force=True)
    except Exception:
        return _json_error("Invalid JSON", 400)
    jid = _new_job()
    threading.Thread(target=_run_qwen_edit_job, args=(jid, data), daemon=True).start()
    return jsonify({"status": "accepted", "job_id": jid}), 202

@app.route("/progress/<job_id>", methods=["GET"])
def api_progress(job_id: str):
    """Return JSON progress for a job; includes the base64 data URL when done."""
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    body = {k: j.get(k, "") for k in ("status", "percent", "status_text")}
    if j.get("status") == "done":
        body["image_data_url"] = j.get("result", "")
    if j.get("status") == "error":
        body["error"] = j.get("error", "")
    return jsonify(body)

@app.route("/events/<job_id>", methods=["GET"])
def api_events(job_id: str):
    """SSE stream of progress and final result/error for a job."""
    def _gen():
        last = -1
        while True:
            j = JOBS.get(job_id)
            if not j:
                yield "event: error\ndata: {\"error\":\"unknown job\"}\n\n"; return
            pct = int(j.get("percent", 0))
            if pct != last:
                last = pct
                yield f"event: progress\ndata: {{\"percent\": {pct}}}\n\n"
            if j["status"] == "done":
                yield f"event: result\ndata: {{\"image_data_url\": \"{j['result']}\"}}\n\n"
                return
            if j["status"] == "error":
                err = (j.get("error") or "").replace('"', '\\"')
                yield f"event: error\ndata: {{\"error\": \"{err}\"}}\n\n"
                return
            time.sleep(0.2)
    return Response(stream_with_context(_gen()), mimetype="text/event-stream")

# ---------------- Progress/job registry ----------------

JOBS: Dict[str, Dict[str, Any]] = {}

def _new_job() -> str:
    """Create and register a new job; return its id."""
    jid = uuid.uuid4().hex
    JOBS[jid] = {"status": "running", "percent": 0, "result": None, "error": None, "started_at": time.time()}
    return jid

def _set_progress(job_id: str, percent: int):
    """Update job progress (clamped to 0..100)."""
    j = JOBS.get(job_id)
    if j:
        j["percent"] = max(0, min(100, int(percent)))

def _set_result(job_id: str, data_url: str):
    """Mark job as done and store the base64 image data URL."""
    j = JOBS.get(job_id)
    if j:
        j["status"] = "done"; j["percent"] = 100; j["result"] = data_url; JOBS[job_id]["status_text"] = "Done"

def _set_error(job_id: str, message: str):
    """Mark job as error and store the message."""
    j = JOBS.get(job_id)
    if j:
        j["status"] = "error"
        j["error"] = str(message)
        JOBS[job_id]["status_text"] = "Error"
    # Console log for any job error
    try:
        print(f"[qtDiffusion][job:{job_id}] ERROR: {message}")
    except Exception:
        pass

# ---------------- Miscellaneous Helpers ----------------

def _round_to_multiple(x: float, m: int) -> int:
    """Round 'x' to the nearest multiple of 'm' (minimum of 'm')."""
    if m <= 0:
        return max(1, int(round(x)))
    return max(m, int(round(float(x) / m)) * m)

def _compute_sdxl_size(src_w: int, src_h: int, target_area: int = 1024 * 1024, multiple: int = 8) -> Tuple[int, int]:
    """
    Compute SDXL-friendly output size that:
    - preserves aspect ratio of (src_w x src_h),
    - targets 'target_area' pixels overall,
    - and is divisible by 'multiple' (default 8).
    """
    w = max(1, int(src_w))
    h = max(1, int(src_h))
    ar = float(w) / float(h)
    tgt_h = math.sqrt(target_area / ar)
    tgt_w = ar * tgt_h
    out_w = max(multiple, int(round(tgt_w / multiple)) * multiple)
    out_h = max(multiple, int(round(tgt_h / multiple)) * multiple)
    return int(out_w), int(out_h)

def _pil_to_data_url(img: Image.Image, fmt: str = "PNG", quality: int = 92) -> str:
    """Encode a PIL image as a data URL (PNG by default; JPEG with quality when requested)."""
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

# ---------------- Async worker functions ----------------
def _run_t2i_job(job_id: str, payload: Dict[str, Any]):
    """Worker: SDXL text-to-image. Writes progress via callback and posts result/error to JOBS."""
    try:
        model_file = payload.get("model_file") or os.path.basename(MODEL_PATH)
        prompt = payload.get("prompt") or ""
        req_w = int(payload.get("width") or 1024)
        req_h = int(payload.get("height") or 1024)
        width, height = _compute_sdxl_size(req_w, req_h)
        steps = int(payload.get("num_inference_steps") or SD_PARAMS["steps"])
        guidance = float(payload.get("guidance_scale") or SD_PARAMS["cfg_scale"])
        seed = _seed_from_payload(payload)

        # Gather LoRAs: prefer 'loras' list; accept single 'lora' or list fallback
        loras_req = payload.get("loras")
        if not loras_req:
            lora_single = payload.get("lora")
            if isinstance(lora_single, list):
                loras_req = [x for x in lora_single if isinstance(x, str)]
            elif isinstance(lora_single, str):
                loras_req = [lora_single]
        lora_names = [x.strip() for x in (loras_req or []) if isinstance(x, str) and x.strip()]

        model_path = _resolve_model_path(model_file)
        if not os.path.exists(model_path):
            _set_error(job_id, f"model not found: {model_path}"); return

        _init_pipeline(model_path, diffusers.StableDiffusionXLPipeline, 'pipe', '_current_model_path_text')

        # Apply multiple SDXL LoRAs if requested
        _apply_sdxl_loras(pipe, lora_names, which="text")

        cb = _make_progress_callback(job_id, steps)
        gen = torch.Generator(device=device).manual_seed(seed) if isinstance(seed, int) else None

        log_prompt_request("t2i_async", prompt, {
            "steps": steps, "guidance_scale": guidance, "seed": seed,
            "loras": lora_names if lora_names else None
        })
        _log_generation("t2i_async", model_file, prompt, steps=steps, guidance=guidance, seed=seed)

        img = pipe(
            prompt=prompt,
            negative_prompt=SD_PARAMS["negative_prompt"],
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            callback=cb,
            callback_steps=1,
        ).images[0]
        _handle_job_result(job_id, img)
    except Exception as e:
        _handle_job_result(job_id, None, error=e)

def _run_i2i_job(job_id: str, payload: Dict[str, Any]):
    """Worker: SDXL image-to-image. Supports optional mask via AutoPipelineForInpainting."""
    try:
        model_file = payload.get("model_file") or os.path.basename(MODEL_PATH)
        prompt = payload.get("prompt") or ""
        req_w = int(payload.get("width") or 1024)
        req_h = int(payload.get("height") or 1024)
        strength = float(payload.get("strength") or 0.7)
        steps = int(payload.get("num_inference_steps") or SD_PARAMS["steps"])
        guidance = float(payload.get("guidance_scale") or SD_PARAMS["cfg_scale"])
        seed = _seed_from_payload(payload)
        init_image_url = payload.get("init_image") or ""

        # Gather LoRAs (same rules as T2I)
        loras_req = payload.get("loras")
        if not loras_req:
            lora_single = payload.get("lora")
            if isinstance(lora_single, list):
                loras_req = [x for x in lora_single if isinstance(x, str)]
            elif isinstance(lora_single, str):
                loras_req = [lora_single]
        lora_names = [x.strip() for x in (loras_req or []) if isinstance(x, str) and x.strip()]

        width, height = _compute_sdxl_size(req_w, req_h)

        model_path = _resolve_model_path(model_file)
        if not os.path.exists(model_path):
            _set_error(job_id, f"model not found: {model_path}"); return

        _init_pipeline(model_path, "StableDiffusionXLImg2ImgPipeline", 'pipe_i2i', '_current_model_path_i2i')

        # Apply multiple SDXL LoRAs if requested
        _apply_sdxl_loras(pipe_i2i, lora_names, which="i2i")

        cb = _make_progress_callback(job_id, steps)
        try:
            init_img = _decode_and_resize_image(init_image_url, width, height)
        except Exception as e:
            _set_error(job_id, f"invalid init_image: {e}"); return

        gen = torch.Generator(device=device).manual_seed(seed) if isinstance(seed, int) else None

        log_prompt_request("i2i_async", prompt, {
            "steps": steps, "guidance_scale": guidance, "strength": strength, "seed": seed,
            "loras": lora_names if lora_names else None
        })
        _log_generation("i2i_async", model_file, prompt, steps=steps, guidance=guidance, strength=strength, seed=seed)

        out = None
        mask_img = _decode_and_resize_mask(payload.get("mask_image"), width, height)
        if mask_img:
            try:
                from diffusers import AutoPipelineForInpainting
                pin = AutoPipelineForInpainting.from_pipe(pipe_i2i).to(device)
                pin.set_progress_bar_config(disable=True)
                out = pin(
                    prompt=prompt,
                    negative_prompt=SD_PARAMS["negative_prompt"],
                    image=init_img,
                    mask_image=mask_img,
                    strength=max(0.0, min(1.0, strength)),
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                    callback=cb,
                    callback_steps=1,
                ).images[0]
            except Exception as e:
                print(f"[nt6diffusion][i2i_async+mask] error: {e} (fallback to i2i)")

        if out is None:
            out = pipe_i2i(
                prompt=prompt,
                negative_prompt=SD_PARAMS["negative_prompt"],
                image=init_img,
                strength=max(0.0, min(1.0, strength)),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
                callback=cb,
                callback_steps=1,
            ).images[0]

        _handle_job_result(job_id, out)
    except Exception as e:
        _handle_job_result(job_id, None, error=e)

def _run_qwen_edit_job(job_id: str, payload: Dict[str, Any]):
    """Worker: Qwen image edit pipeline (optionally conditioned on an input image)."""
    try:
        prompt = payload.get("prompt") or ""
        true_cfg_scale = float(payload.get("true_cfg_scale") or 4.0)
        steps = int(payload.get("num_inference_steps") or 50)
        seed = _seed_from_payload(payload)
        lora = payload.get("lora")
        model_file = payload.get("model_file") or None
        init_image_url = payload.get("init_image") or None

        log_prompt_request("qwen_async", prompt, {"true_cfg_scale": true_cfg_scale, "steps": steps, "seed": seed})
        _log_generation("qwen_async", model_file, prompt, steps=steps, true_cfg_scale=true_cfg_scale, seed=seed)

        _init_qwen_pipe(lora)
        gen = torch.Generator(device=device).manual_seed(seed) if (seed is not None and isinstance(seed, int)) else None

        image = None
        if init_image_url:
            try:
                image = _decode_data_url_to_pil(init_image_url)
            except Exception as e:
                _set_error(job_id, f"invalid init_image: {e}")
                return

        qwen_args = {
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "num_inference_steps": steps,
            "generator": gen,
        }
        if image is not None:
            qwen_args["image"] = image

        result = qwen_pipe(**qwen_args)
        img = result.images[0] if hasattr(result, "images") and result.images else result
        _handle_job_result(job_id, img)
    except Exception as e:
        _handle_job_result(job_id, None, error=e)

# ---------------- Entrypoint ----------------
def start_http_server():
    """Start the Flask HTTP server on localhost."""
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    start_http_server()

