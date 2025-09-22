# -*- coding: utf-8 -*-
"""
Qwen Image Edit Backend (qwen1) - polishTest.py feature parity for server use (no Gradio).

 - Local preloaded model directory override (with environment variable fallbacks)
 - Optional mixed precision: vision (4-bit NF4) + text encoder (bf16) or full precision via flags
 - Lightning LoRA (local-first, HF fallback) + additional LoRAs (multi-adapter activation)
 - Deterministic image resizing to ~1024x1024 area (aspect preserved, dims %4)
 - Verbose GPU/time logging & per-step diffusion progress (scheduler.step monkey patch)
 - Lazy, thread-safe pipeline initialization with reuse
 - Progress callbacks integrated with server job system (set_progress)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import os, threading, time, traceback, math
from PIL import Image
import sys

# -------------------------------------------------------------------
# (NEW) Optional import of shared Qt blur utility
# Defines _HAS_QT_BLUR / _qt_blur used later in _generate_t2i_base_image.
# Safe even if PyQt or the utilities module are unavailable (graceful fallback).
# -------------------------------------------------------------------
_HAS_QT_BLUR = False
_qt_blur = None
try:
    # Direct import (when project root already on sys.path)
    from qt_paint_tools.qtPaintToolUtilities import blur_qimage_gaussian as _qt_blur  # type: ignore
    _HAS_QT_BLUR = True
except Exception:
    # Second attempt: prepend repo root (parent of 'qtd') then retry
    try:
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from qt_paint_tools.qtPaintToolUtilities import blur_qimage_gaussian as _qt_blur  # type: ignore
        _HAS_QT_BLUR = True
    except Exception:
        _HAS_QT_BLUR = False
        _qt_blur = None

# -------------------------------------------------------------------
# Protocol Types (server expects this shape)
# -------------------------------------------------------------------
ProgressFn = Callable[[int], None]
SetResultFn = Callable[[Any], None]
SetErrorFn = Callable[[str], None]

class Backend:
    id: str = "backend"
    def describe_models(self) -> List[Dict[str, Any]]: raise NotImplementedError
    def list_loras(self, model_id: str) -> List[Dict[str, Any]]: return []
    def submit(self, *, model_id: str, operation: str, inputs: Dict[str, Any],
               loras: Optional[List[str]], set_progress: ProgressFn,
               set_result: SetResultFn, set_error: SetErrorFn) -> None: raise NotImplementedError
    def unload(self) -> None: pass

# -------------------------------------------------------------------
# Helper Imports (dual-path like other backends)
# -------------------------------------------------------------------
try:
    from .qtdHelpers import (  # type: ignore
        list_safetensors, resolve_in_dir, seed_from_inputs, decode_data_url_to_pil,
        pil_to_data_url, bf16_supported, empty_cuda_cache,
        resize_to_area_preserve_aspect, ensure_rgb, patch_scheduler_progress
    )
    from .qtdConstants import (  # type: ignore
        QWEN_USE_PRELOADED,
        QWEN_PRELOADED_PATH,
        QWEN_USE_HIGH_PRECISION_VISION,
        QWEN_USE_HIGH_PRECISION_TEXT,
        QWEN_LIGHTNING_LORA_DIR,
        QWEN_LIGHTNING_LORA_FILENAME,
        QWEN_ADDITIONAL_LORA_DIR,
        QWEN_ADDITIONAL_LORAS,
        QWEN_VERBOSE,
        QWEN_TARGET_AREA,
        QWEN_DEFAULT_STEPS,
        QWEN_HF_MODEL_ID,
    )
except ImportError:
    from qtdHelpers import (  # type: ignore
        list_safetensors, resolve_in_dir, seed_from_inputs, decode_data_url_to_pil,
        pil_to_data_url, bf16_supported, empty_cuda_cache,
        resize_to_area_preserve_aspect, ensure_rgb, patch_scheduler_progress
    )
    from qtdConstants import (  # type: ignore
        QWEN_USE_PRELOADED,
        QWEN_PRELOADED_PATH,
        QWEN_USE_HIGH_PRECISION_VISION,
        QWEN_USE_HIGH_PRECISION_TEXT,
        QWEN_LIGHTNING_LORA_DIR,
        QWEN_LIGHTNING_LORA_FILENAME,
        QWEN_ADDITIONAL_LORA_DIR,
        QWEN_ADDITIONAL_LORAS,
        QWEN_VERBOSE,
        QWEN_TARGET_AREA,
        QWEN_DEFAULT_STEPS,
        QWEN_HF_MODEL_ID,
    )

# -------------------------------------------------------------------
# Logging utilities
# -------------------------------------------------------------------

def _gpu_mem(torch_mod) -> str:
    if not QWEN_VERBOSE: return ""
    try:
        if torch_mod is None or not torch_mod.cuda.is_available():
            return "CUDA not available"
        free_b, total_b = torch_mod.cuda.mem_get_info(torch_mod.cuda.current_device())
        used_b = total_b - free_b
        gb = 1024 ** 3
        return f"used {used_b/gb:.2f}GB / total {total_b/gb:.2f}GB (free {free_b/gb:.2f}GB)"
    except Exception as e:
        return f"mem? ({e})"

def _vprint(msg: str):
    if QWEN_VERBOSE:
        print(msg)

def _log_step(label: str, start_time: float, torch_mod):
    if not QWEN_VERBOSE: return
    elapsed = time.time() - start_time
    print(f"[backend][qwen1][load] {label} | {elapsed:.2f}s | GPU: {_gpu_mem(torch_mod)}")

def _resize_to_target(image: Image.Image) -> Image.Image:
    width, height = image.size
    orig_area = width * height
    if width % 4 == 0 and height % 4 == 0 and orig_area == QWEN_TARGET_AREA:
        return image
    aspect = width / height
    scale = math.sqrt(QWEN_TARGET_AREA / orig_area)
    ideal_w = width * scale
    def mult4(x: float) -> int: return max(4, int(round(x / 4.0)) * 4)
    w_floor = max(4, (int(ideal_w) // 4) * 4)
    w_ceil = w_floor + 4
    w_candidates = {w_floor, w_ceil, mult4(ideal_w)}
    candidates = []
    for w_c in w_candidates:
        if w_c <= 0: continue
        h_ideal = w_c / aspect
        h_floor = max(4, (int(h_ideal) // 4) * 4)
        h_ceil = h_floor + 4
        for h_c in {h_floor, h_ceil, mult4(h_ideal)}:
            if h_c <= 0: continue
            area = w_c * h_c
            candidates.append((abs(area - QWEN_TARGET_AREA), abs((w_c / h_c) - aspect), area, w_c, h_c))
    candidates.sort()
    _, _, _, new_w, new_h = candidates[0]
    if new_w == width and new_h == height:
        return image
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

# -------------------------------------------------------------------
# Backend Implementation
# -------------------------------------------------------------------

class QwenEdit1Backend(Backend):
    id = "qwen1"
    _torch = None
    _diffusers = None
    _pipe = None
    _device = "cpu"
    _init_lock = threading.Lock()
    _current_lightning: Optional[str] = None
    _loaded_additional: List[str] = []

    def describe_models(self) -> List[Dict[str, Any]]:
        tags = ["qwen", "edit"]
        if not QWEN_USE_HIGH_PRECISION_VISION: tags.append("vision-4bit")
        if not QWEN_USE_HIGH_PRECISION_TEXT: tags.append("text-4bit")
        if QWEN_USE_HIGH_PRECISION_VISION or QWEN_USE_HIGH_PRECISION_TEXT: tags.append("mixed-precision")
        if QWEN_LIGHTNING_LORA_FILENAME: tags.append("lightning")
        # Expose both "t2i" (implemented via generated + blurred base image) and "edit"
        return [{
            "id": "qwen1:image-edit",
            "backend": self.id,
            "displayName": "Qwen Image Edit (qwen1)",
            "tags": tags,
            "supportsMultiLoRA": True,
            "operations": [
                {
                    "id": "t2i",
                    "displayName": "Text to Image",
                    "inputs": [
                        {"name": "prompt", "type": "string", "required": True},
                        {"name": "true_cfg_scale", "type": "float", "min": 0.0, "max": 20.0, "default": 1.0},
                        {"name": "num_inference_steps", "type": "int", "min": 1, "max": 200, "default": QWEN_DEFAULT_STEPS},
                        {"name": "generator", "type": "int", "required": False},
                    ],
                },
                {
                    "id": "edit",
                    "displayName": "Edit",
                    "inputs": [
                        {"name": "prompt", "type": "string", "required": True},
                        {"name": "true_cfg_scale", "type": "float", "min": 0.0, "max": 20.0, "default": 1.0},
                        {"name": "num_inference_steps", "type": "int", "min": 1, "max": 200, "default": QWEN_DEFAULT_STEPS},
                        {"name": "generator", "type": "int", "required": False},
                        {"name": "init_image", "type": "image", "required": False},
                    ],
                },
            ],
        }]


    def list_loras(self, model_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if os.path.isdir(QWEN_LIGHTNING_LORA_DIR):
                for f in list_safetensors(QWEN_LIGHTNING_LORA_DIR):
                    out.append({"name": f, "path": os.path.join(QWEN_LIGHTNING_LORA_DIR, f), "tags": ["lightning"]})
        except Exception:
            print("[backend][qwen1][loras] lightning list failed"); traceback.print_exc()
        try:
            if os.path.isdir(QWEN_ADDITIONAL_LORA_DIR):
                for f in list_safetensors(QWEN_ADDITIONAL_LORA_DIR):
                    out.append({"name": f, "path": os.path.join(QWEN_ADDITIONAL_LORA_DIR, f), "tags": ["additional"]})
        except Exception:
            print("[backend][qwen1][loras] additional list failed"); traceback.print_exc()
        return out

    # ------------------------------------------------------------------
    # Public uniform loader (called by server prior to submit)
    # ------------------------------------------------------------------
    def ensure_pipeline(self, *, model_id: str, operation: str,
                        progress_cb: Optional[ProgressFn] = None,
                        status_cb: Optional[Callable[[str], None]] = None) -> None:
        """
        Ensure the Qwen pipeline is loaded (idempotent).
        Added status_cb for richer textual phase updates.
        """
        if self._pipe is not None:
            if status_cb: status_cb("Pipeline ready")
            return
        if status_cb: status_cb("Initializing pipeline")
        self._init_pipeline(progress_cb=progress_cb, status_cb=status_cb)
        if status_cb: status_cb("Pipeline ready")

    # ------------------------------------------------------------------
    # Submit (extended with set_status)
    # ------------------------------------------------------------------
    def submit(
        self,
        *,
        model_id: str,
        operation: str,
        inputs: Dict[str, Any],
        loras: Optional[List[str]],
        set_progress: ProgressFn,
        set_result: SetResultFn,
        set_error: SetErrorFn,
        set_status: Optional[Callable[[str], None]] = None
    ) -> None:
        op = (operation or "").strip().lower()
        if op not in ("edit", "t2i"):
            set_error(f"Unsupported operation: {operation}")
            return

        prompt = (inputs or {}).get("prompt") or ""
        steps = int((inputs or {}).get("num_inference_steps") or QWEN_DEFAULT_STEPS)
        if steps < 1:
            steps = 1
        true_cfg_scale = float((inputs or {}).get("true_cfg_scale") or 1.0)
        seed = seed_from_inputs(inputs or {})
        init_image_data = (inputs or {}).get("init_image") or None

        # Cold init path
        if self._pipe is None:
            try:
                set_progress(1)
                if set_status: set_status("Initializing")
                self.ensure_pipeline(model_id=model_id, operation=op,
                                     progress_cb=set_progress, status_cb=set_status)
                set_progress(10)
                if set_status: set_status("Model ready")
            except Exception as e:
                set_error(f"InitError: {e}")
                traceback.print_exc()
                return
        else:
            set_progress(10)
            if set_status: set_status("Model ready")

        torch_mod = self._torch
        gen = None
        if seed is not None and isinstance(seed, int):
            try:
                gen = torch_mod.Generator(device=self._device).manual_seed(seed)
            except Exception:
                pass

        image: Optional[Image.Image] = None
        if op == "t2i":
            try:
                if set_status: set_status("Preparing base image")
                image = self._generate_t2i_base_image()
            except Exception as e:
                set_error(f"T2I base image generation failed: {e}")
                traceback.print_exc()
                return
        else:
            if isinstance(init_image_data, str) and init_image_data.strip():
                try:
                    if set_status: set_status("Decoding init image")
                    image = decode_data_url_to_pil(init_image_data)
                except Exception as e:
                    set_error(f"init_image decode failed: {e}")
                    traceback.print_exc()
                    return
            if image is not None:
                if set_status: set_status("Resizing image")
                image = resize_to_area_preserve_aspect(image, QWEN_TARGET_AREA, multiple=4)

        if image is not None:
            if set_status: set_status("Normalizing channels")
            image = ensure_rgb(image)

        if set_status: set_status("Denoising")
        start_time = time.time()

        restore_fn = self._patch_scheduler_for_progress(
            steps, set_progress, base=10, status_cb=set_status
        )

        try:
            call_kwargs: Dict[str, Any] = {"prompt": prompt, "num_inference_steps": steps}
            if image is not None:
                call_kwargs["image"] = image
            if gen is not None:
                call_kwargs["generator"] = gen
            if "true_cfg_scale" in self._pipe.__call__.__code__.co_varnames:
                call_kwargs["true_cfg_scale"] = true_cfg_scale

            result = self._pipe(**call_kwargs)
            out_img = result.images[0] if hasattr(result, "images") and result.images else result
            data_url = pil_to_data_url(out_img)
            set_progress(100)
            if set_status: set_status("Finalizing")
            set_result(data_url)
            elapsed = time.time() - start_time
            _vprint(f"[backend][qwen1][gen] done in {elapsed:.2f}s | GPU: {_gpu_mem(torch_mod)}")
        except Exception as e:
            set_error(f"{type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            try:
                restore_fn()
            except Exception:
                pass

    def unload(self) -> None:
        with self._init_lock:
            self._pipe = None
            self._current_lightning = None
            self._loaded_additional = []
            try: empty_cuda_cache(self._torch)
            except Exception: pass
            _vprint("[backend][qwen1] unloaded pipeline & cleared cache")

    # ------------------- T2I synthetic image helpers -------------------
    def _generate_t2i_base_image(self):
        """
        Creates a 1024x1024 white canvas with a dark (black) 800x800 rectangle
        horizontally centered and flush to the bottom. The entire composed image
        is then blurred using the shared Qt blur utility (blur_qimage_gaussian).
        """
        from PIL import Image, ImageDraw
        W, H = 1024, 1024
        RW, RH = 800, 800
        # Center horizontally; flush to bottom vertically.
        rx = (W - RW) // 2
        ry = H - RH
        # Start as RGB (no alpha) to align with pipeline 3-channel expectation
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        # Rectangle fill: black (spec wording ambiguous: "gray 800x800 black filled rectangle";
        # using solid black fill, could add gray border if desired).
        dr.rectangle([rx, ry, rx + RW, ry + RH], fill=(0, 0, 0))

        # Convert to QImage -> blur -> back to PIL using shared utility if available.
        if _HAS_QT_BLUR and _qt_blur:
            try:
                qimg = self._pil_to_qimage(img)
                blurred_q = _qt_blur(qimg, strength=9.0)
                out = self._qimage_to_pil(blurred_q)
                return out.convert("RGB")
            except Exception as e:
                _vprint(f"[backend][qwen1][t2i] qt blur path failed: {e} (falling back)")
        # Fallback: use PIL GaussianBlur (no PyQt dependency)
        try:
            from PIL import ImageFilter
            return img.filter(ImageFilter.GaussianBlur(radius=9)).convert("RGB")
        except Exception as e:
            _vprint(f"[backend][qwen1][t2i] fallback blur failed: {e} (returning unblurred)")
            return img.convert("RGB")

    @staticmethod
    def _pil_to_qimage(pil_img):
        from PyQt6 import QtGui
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", "RGBA")
        qimg = QtGui.QImage(data, pil_img.width, pil_img.height, QtGui.QImage.Format.Format_RGBA8888)
        return qimg

    @staticmethod
    def _qimage_to_pil(qimg):
        from PyQt6 import QtGui
        from PIL import Image
        if qimg.format() != QtGui.QImage.Format.Format_RGBA8888:
            qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        arr = bytes(ptr)
        img = Image.frombuffer("RGBA", (qimg.width(), qimg.height()), arr, "raw", "RGBA", 0, 1)
        return img

    # ---------- Internal helpers ----------
    def _ensure_rgb(self, img: Image.Image) -> Image.Image:
        """
        Guarantee a 3-channel RGB image (Qwen VAE expects 3, error showed 4).
        Also guards accidental conversion to mode 'LA' or others.
        """
        if img is None:
            return img
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    # ------------------- Pipeline init (add status_cb) -------------------
    def _init_pipeline(self, progress_cb: Optional[ProgressFn] = None,
                       status_cb: Optional[Callable[[str], None]] = None):
        if self._pipe is not None:
            return
        with self._init_lock:
            if self._pipe is not None:
                return
            self._load_libs()
            torch_mod = self._torch
            diff_mod = self._diffusers
            start_time = time.time()
            self._device = "cuda" if torch_mod.cuda.is_available() else "cpu"
            desired_dtype = (
                (torch_mod.bfloat16 if bf16_supported(torch_mod) else torch_mod.float16)
                if self._device == "cuda" else torch_mod.float32
            )
            model_root = self._resolve_model_id()
            local_only = os.path.isdir(model_root)

            if progress_cb: progress_cb(2)
            if status_cb: status_cb("Loading vision transformer")
            if QWEN_USE_HIGH_PRECISION_VISION:
                transformer = self._load_transformer(diff_mod, model_root, None, desired_dtype, local_only)
            else:
                from diffusers import BitsAndBytesConfig as DiffBNB
                vision_quant = DiffBNB(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=desired_dtype,
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                )
                transformer = self._load_transformer(diff_mod, model_root, vision_quant, desired_dtype, local_only)
            if progress_cb: progress_cb(4)
            _log_step("Vision loaded", start_time, torch_mod)
            if status_cb: status_cb("Vision loaded")

            if status_cb: status_cb("Loading text encoder")
            if QWEN_USE_HIGH_PRECISION_TEXT:
                text_encoder = self._load_text_encoder(model_root, None, desired_dtype, local_only)
            else:
                from transformers import BitsAndBytesConfig as HFBNB
                text_quant = HFBNB(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=desired_dtype,
                )
                text_encoder = self._load_text_encoder(model_root, text_quant, desired_dtype, local_only)
            if progress_cb: progress_cb(6)
            _log_step("Text encoder loaded", start_time, torch_mod)
            if status_cb: status_cb("Text encoder loaded")

            if status_cb: status_cb("Assembling pipeline")
            from diffusers import QwenImageEditPipeline
            pipe = QwenImageEditPipeline.from_pretrained(
                model_root,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=desired_dtype,
                local_files_only=local_only,
                use_safetensors=True,
            )
            if progress_cb: progress_cb(7)
            _log_step("Pipeline assembled", start_time, torch_mod)
            if status_cb: status_cb("Pipeline assembled")

            if status_cb: status_cb("Loading Lightning LoRA")
            self._load_lightning_lora(pipe, start_time, torch_mod)
            if status_cb: status_cb("Loading additional LoRAs")
            self._load_additional_loras(pipe, start_time, torch_mod)
            if progress_cb: progress_cb(8)

            try:
                if status_cb: status_cb("Enabling CPU offload")
                pipe.enable_model_cpu_offload()
                _log_step("CPU offload enabled", start_time, torch_mod)
                if status_cb: status_cb("CPU offload enabled")
            except Exception:
                _vprint("[backend][qwen1][init] CPU offload enabling failed")
                if status_cb: status_cb("CPU offload skipped")

            if progress_cb: progress_cb(9)
            self._pipe = pipe
            _vprint(f"[backend][qwen1][init] COMPLETE in {time.time()-start_time:.2f}s | GPU: {_gpu_mem(torch_mod)}")
            if progress_cb: progress_cb(10)
            if status_cb: status_cb("Pipeline ready")

    # ------------------- Scheduler progress patch (extended) -------------
    def _patch_scheduler_for_progress(self, user_steps: int, set_progress: ProgressFn,
                                      base: int = 10,
                                      status_cb: Optional[Callable[[str], None]] = None):
        """
        Keeps original scheduler monkey patch logic; now optionally emits
        periodic status text (every ~1/8 of total steps).
        """
        if self._pipe is None or not hasattr(self._pipe, "scheduler"):
            return lambda: None
        sched = getattr(self._pipe, "scheduler", None)
        if sched is None or not hasattr(sched, "step"):
            return lambda: None

        original_step = sched.step
        state = {"i": 0}
        interval = max(1, user_steps // 8) if user_steps > 0 else 1

        def patched_step(model_output, timestep, *args, **kwargs):
            state["i"] += 1
            if user_steps > 0:
                frac = min(1.0, max(0.0, state["i"] / float(user_steps)))
                pct = base + int(frac * 89)
                if pct >= 100:
                    pct = 99
                set_progress(pct)
                if status_cb and (state["i"] % interval == 0 or state["i"] == user_steps):
                    status_cb(f"Denoising {state['i']}/{user_steps}")
            return original_step(model_output, timestep, *args, **kwargs)

        sched.step = patched_step

        def restore():
            try:
                sched.step = original_step
            except Exception:
                pass
        return restore

    # ------------------------------------------------------------------
    # INTERNAL MODEL / COMPONENT LOADING HELPERS (restored)
    # These were referenced by _init_pipeline but got removed, causing
    # AttributeError ('_load_libs', etc.). Reintroducing verbatim logic.
    # ------------------------------------------------------------------
    def _load_libs(self):
        if self._torch is None:
            import torch as _torch
            self._torch = _torch
        if self._diffusers is None:
            import diffusers as _diff
            self._diffusers = _diff

    def _resolve_model_id(self) -> str:
        if QWEN_USE_PRELOADED and os.path.isdir(QWEN_PRELOADED_PATH):
            return QWEN_PRELOADED_PATH
        return QWEN_HF_MODEL_ID

    def _load_transformer(self, diff_mod, model_root: str, quant_config, dtype, local_only: bool):
        from diffusers import QwenImageTransformer2DModel
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_root,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=dtype,
            local_files_only=local_only,
            use_safetensors=True,
        )
        return transformer.to("cpu")

    def _load_text_encoder(self, model_root: str, quant_config, dtype, local_only: bool):
        from transformers import Qwen2_5_VLForConditionalGeneration
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_root,
            subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        return te.to("cpu")

    def _load_lightning_lora(self, pipe, start_time, torch_mod):
        if not QWEN_LIGHTNING_LORA_FILENAME:
            return
        local_path = os.path.join(QWEN_LIGHTNING_LORA_DIR, QWEN_LIGHTNING_LORA_FILENAME)
        loaded = False
        if os.path.isfile(local_path):
            try:
                pipe.load_lora_weights(local_path)
                loaded = True
                self._current_lightning = local_path
            except Exception as e:
                _vprint(f"[backend][qwen1][lora][warn] local lightning load failed: {e}")
        if not loaded:
            try:
                pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name=QWEN_LIGHTNING_LORA_FILENAME)
                self._current_lightning = f"hf:{QWEN_LIGHTNING_LORA_FILENAME}"
                loaded = True
            except Exception as e:
                _vprint(f"[backend][qwen1][lora][warn] HF fallback failed: {e}")
        if loaded:
            _log_step("Lightning LoRA loaded", start_time, torch_mod)

    def _load_additional_loras(self, pipe, start_time, torch_mod):
        if not QWEN_ADDITIONAL_LORAS:
            return
        loaded_names: List[str] = []
        for fname in QWEN_ADDITIONAL_LORAS:
            full = os.path.join(QWEN_ADDITIONAL_LORA_DIR, fname)
            if not os.path.isfile(full):
                _vprint(f"[backend][qwen1][lora][extra][miss] {full}")
                continue
            adapter_name = os.path.splitext(os.path.basename(fname))[0]
            try:
                pipe.load_lora_weights(full, adapter_name=adapter_name)
                loaded_names.append(adapter_name)
                _vprint(f"[backend][qwen1][lora][extra] Loaded {full} (adapter={adapter_name})")
            except Exception as e:
                _vprint(f"[backend][qwen1][lora][extra][warn] Failed {full}: {e}")
        if loaded_names:
            try:
                if hasattr(pipe, "set_adapters"):
                    weights = [1.0] * len(loaded_names)
                    pipe.set_adapters(loaded_names, adapter_weights=weights)
                    self._loaded_additional = loaded_names
                    _log_step(f"Additional LoRAs activated ({len(loaded_names)})", start_time, torch_mod)
            except Exception as e:
                _vprint(f"[backend][qwen1][lora][extra][warn] activation failed: {e}")