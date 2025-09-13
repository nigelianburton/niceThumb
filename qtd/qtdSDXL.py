from typing import Any, Dict, List, Optional, Callable
import os
import threading

from PIL import Image  # type hints only

ProgressFn = Callable[[int], None]

class Backend:
    id: str = "backend"
    def describe_models(self) -> List[Dict[str, Any]]: raise NotImplementedError
    def list_loras(self, model_id: str) -> List[Dict[str, Any]]: return []
    def submit(self, *, model_id: str, operation: str, inputs: Dict[str, Any],
               loras: Optional[List[str]], set_progress: ProgressFn,
               set_result: Callable[[Any], None], set_error: Callable[[str], None]) -> None: raise NotImplementedError
    def unload(self) -> None: pass

try:
    from .qtdHelpers import (  # type: ignore
        list_safetensors,
        resolve_lora_list,
        seed_from_inputs,
        decode_and_resize_image,
        decode_and_resize_mask,
        pil_to_data_url,
        compute_sdxl_size,
        make_diffusers_progress_callback,  # kept (not used now but retained for compatibility)
        sanitize_adapter_name,
    )
except ImportError:
    from qtdHelpers import (
        list_safetensors,
        resolve_lora_list,
        seed_from_inputs,
        decode_and_resize_image,
        decode_and_resize_mask,
        pil_to_data_url,
        compute_sdxl_size,
        make_diffusers_progress_callback,
        sanitize_adapter_name,
    )


class SDXLBackend(Backend):
    id = "sdxl"
    _torch = None
    _diffusers = None
    _scheduler_cls = None
    _device = "cpu"
    _init_lock = threading.Lock()

    _pipe_text = None
    _pipe_i2i = None
    _current_model_path_text: Optional[str] = None
    _current_model_path_i2i: Optional[str] = None
    _current_adapter_names_text: List[str] = []
    _current_adapter_names_i2i: List[str] = []

    _MODEL_PATH = os.environ.get("NT6_SDXL_MODEL", r"C:\_MODELS-SD\StableDiffusion\juggernautXL_ragnarokBy.safetensors")
    _SD_MODELS_DIR = os.environ.get("NT6_SDXL_DIR", r"C:\_MODELS-SD\StableDiffusion")
    _LORAS_DIR = os.environ.get("NT6_LORAS_DIR", r"C:\_MODELS-SD\Lora")
    _NEG_PROMPT = os.environ.get("NT6_NEG_PROMPT", "blurry, lowres, deformed, extra limbs, bad anatomy, watermark, text")
    _DEFAULT_STEPS = int(os.environ.get("NT6_STEPS", "30"))
    _DEFAULT_CFG = float(os.environ.get("NT6_CFG", "7.5"))

    # ----------------- Public -----------------
    def describe_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        for name in list_safetensors(self._SD_MODELS_DIR):
            models.append(self._model_descriptor(name))
        if not models and isinstance(self._MODEL_PATH, str) and self._MODEL_PATH.strip():
            bn = os.path.basename(self._MODEL_PATH)
            models.append(self._model_descriptor(bn))
        return models

    def _model_descriptor(self, name: str) -> Dict[str, Any]:
        disp = os.path.splitext(name)[0]
        return {
            "id": f"sdxl:{name}",
            "backend": "sdxl",
            "displayName": f"SDXL: {disp}",
            "tags": ["sdxl", "t2i", "i2i"],
            "supportsMultiLoRA": True,
            "operations": [
                {
                    "id": "t2i",
                    "displayName": "Text to Image",
                    "inputs": [
                        {"name": "prompt", "type": "string", "required": True},
                        {"name": "width", "type": "int", "min": 64, "max": 2048, "step": 8, "default": 1024},
                        {"name": "height", "type": "int", "min": 64, "max": 2048, "step": 8, "default": 1024},
                        {"name": "num_inference_steps", "type": "int", "min": 1, "max": 200, "default": 40},
                        {"name": "guidance_scale", "type": "float", "min": 0, "max": 30, "step": 0.1, "default": 10.5},
                        {"name": "generator", "type": "int", "required": False},
                    ],
                },
                {
                    "id": "i2i",
                    "displayName": "Image to Image",
                    "inputs": [
                        {"name": "prompt", "type": "string", "required": True},
                        {"name": "width", "type": "int", "min": 64, "max": 2048, "step": 8, "default": 1024},
                        {"name": "height", "type": "int", "min": 64, "max": 2048, "step": 8, "default": 1024},
                        {"name": "strength", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.8},
                        {"name": "num_inference_steps", "type": "int", "min": 1, "max": 200, "default": 40},
                        {"name": "guidance_scale", "type": "float", "min": 0, "max": 30, "step": 0.1, "default": 10.5},
                        {"name": "generator", "type": "int", "required": False},
                        {"name": "init_image", "type": "image", "required": True},
                        {"name": "mask_image", "type": "image", "required": False},
                    ],
                },
            ],
        }

    def list_loras(self, model_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            for f in list_safetensors(self._LORAS_DIR):
                full = os.path.join(self._LORAS_DIR, f)
                out.append({"name": f, "path": full, "tags": ["sdxl"]})
        except Exception:
            pass
        return out

    # ----------------- Submit with staged progress -----------------
    def submit(
        self,
        *,
        model_id: str,
        operation: str,
        inputs: Dict[str, Any],
        loras: Optional[List[str]],
        set_progress: ProgressFn,
        set_result: Callable[[Any], None],
        set_error: Callable[[str], None],
        set_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        try:
            op = (operation or "").lower().strip()
            if op not in ("t2i", "i2i"):
                set_error(f"Unsupported operation for SDXL: {operation}")
                return

            model_file = self._model_file_from_id(model_id)
            model_path = self._resolve_model_path(model_file)
            if not (isinstance(model_path, str) and os.path.exists(model_path)):
                set_error(f"model not found: {model_path}")
                return

            prompt = (inputs or {}).get("prompt") or ""
            req_w = int((inputs or {}).get("width") or 1024)
            req_h = int((inputs or {}).get("height") or 1024)
            steps = int((inputs or {}).get("num_inference_steps") or self._DEFAULT_STEPS)
            guidance = float((inputs or {}).get("guidance_scale") or self._DEFAULT_CFG)
            seed = seed_from_inputs(inputs or {})
            # Use the requested dimensions directly, bypassing compute_sdxl_size.
            width, height = req_w, req_h
            lora_paths = resolve_lora_list(loras or [], self._LORAS_DIR)

            # Stage 0..10: model / pipeline preparation
            base_loaded = (self._pipe_text if op == "t2i" else self._pipe_i2i) is not None
            if not base_loaded:
                set_progress(1)          # loading_model start
                if set_status: set_status("Loading model")
            self._init_pipeline(model_path, pipe_kind=("text" if op == "t2i" else "i2i"),
                                progress_cb=(set_progress if not base_loaded else None))
            if not base_loaded:
                set_progress(6)          # after model load
                if set_status: set_status("Model loaded")
            if op == "t2i":
                self._apply_sdxl_loras(self._pipe_text, lora_paths, which="text")
            else:
                self._apply_sdxl_loras(self._pipe_i2i, lora_paths, which="i2i")
            if not base_loaded:
                set_progress(9)          # after lora application
                if set_status: set_status("LoRAs applied")
            set_progress(10)             # ready to generate
            if set_status: set_status("Generating")

            # Build generation progress callback mapping steps -> 10..99
            def gen_progress(step: int, timestep: int, latents):
                if steps <= 0:
                    return
                frac = min(1.0, max(0.0, step / float(steps)))
                pct = 10 + int(frac * 89)
                if pct >= 100:
                    pct = 99
                set_progress(pct)
                if set_status and (step % max(1, steps // 8) == 0):
                    set_status(f"Denoising {step}/{steps}")

            gen = None
            if seed is not None and isinstance(seed, int):
                gen = self._torch.Generator(device=self._device).manual_seed(seed)

            # --------------------------
            # Internal unified call helper to avoid deprecated callback warning
            # --------------------------
            if op == "t2i":
                call_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=self._NEG_PROMPT,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                )
                def _on_step_end(pipeline, step, timestep, callback_kwargs):
                    gen_progress(step, timestep, callback_kwargs.get("latents"))
                    return callback_kwargs
                call_kwargs["callback_on_step_end"] = _on_step_end
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                out = self._pipe_text(**call_kwargs).images[0]
                data_url = pil_to_data_url(out)
                set_progress(100)
                if set_status: set_status("Finalizing")
                set_result(data_url)
                return

            # i2i path
            strength = float((inputs or {}).get("strength") or 0.7)
            init_image_url = (inputs or {}).get("init_image") or ""
            if not init_image_url:
                set_error("init_image is required for i2i")
                return
            try:
                init_img = decode_and_resize_image(init_image_url, width, height)
            except Exception as e:
                set_error(f"invalid init_image: {e}")
                return

            out_img = None
            mask_url = (inputs or {}).get("mask_image")
            mask_img = decode_and_resize_mask(mask_url, width, height) if mask_url else None
            if mask_img is not None:
                try:
                    from diffusers import AutoPipelineForInpainting
                    pin = AutoPipelineForInpainting.from_pipe(self._pipe_i2i).to(self._device)
                    pin.set_progress_bar_config(disable=True)
                    inpaint_kwargs = dict(
                        prompt=prompt,
                        negative_prompt=self._NEG_PROMPT,
                        image=init_img,
                        mask_image=mask_img,
                        strength=max(0.0, min(1.0, strength)),
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                    )
                    def _on_step_end_ip(pipeline, step, timestep, callback_kwargs):
                        gen_progress(step, timestep, callback_kwargs.get("latents"))
                        return callback_kwargs
                    inpaint_kwargs["callback_on_step_end"] = _on_step_end_ip
                    inpaint_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                    out_img = pin(**inpaint_kwargs).images[0]
                except Exception:
                    out_img = None
            if out_img is None:
                call_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=self._NEG_PROMPT,
                    image=init_img,
                    strength=max(0.0, min(1.0, strength)),
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                )
                def _on_step_end_i2i(pipeline, step, timestep, callback_kwargs):
                    gen_progress(step, timestep, callback_kwargs.get("latents"))
                    return callback_kwargs
                call_kwargs["callback_on_step_end"] = _on_step_end_i2i
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                out_img = self._pipe_i2i(**call_kwargs).images[0]
            data_url = pil_to_data_url(out_img)
            set_progress(100)
            if set_status: set_status("Finalizing")
            set_result(data_url)
        except Exception as e:
            set_error(f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Unified loader for server (mirrors Qwen backend interface)
    # ------------------------------------------------------------------
    def ensure_pipeline(self, *, model_id: str, operation: str,
                        progress_cb: Optional[ProgressFn] = None,
                        status_cb: Optional[Callable[[str], None]] = None) -> None:
         """
         Ensure appropriate SDXL pipeline (text or i2i) is loaded.
         operation: 't2i' -> text pipeline, 'i2i' -> img2img pipeline.
         """
         op = (operation or "").lower().strip()
         pipe_kind = "text" if op == "t2i" else "i2i"
         model_file = self._model_file_from_id(model_id)
         model_path = self._resolve_model_path(model_file)
         if status_cb: status_cb(f"Loading SDXL ({pipe_kind}) pipeline")
         self._init_pipeline(model_path, pipe_kind=pipe_kind, progress_cb=progress_cb)
         if status_cb: status_cb("Pipeline ready")

    # ------------------------------------------------------------------
    # (Restored) path / model resolution helpers required by submit()
    # These were present originally but appear to have been removed,
    # causing AttributeError: '_model_file_from_id' not found.
    # ------------------------------------------------------------------
    def _model_file_from_id(self, model_id: str) -> str:
        if isinstance(model_id, str) and model_id.startswith("sdxl:"):
            return model_id.split(":", 1)[1]
        return os.path.basename(self._MODEL_PATH or "")

    def _model_dir(self) -> str:
        return os.path.dirname(self._MODEL_PATH) if self._MODEL_PATH else (self._SD_MODELS_DIR or "")

    def _resolve_model_path(self, model_file: str) -> str:
        if not model_file:
            return self._MODEL_PATH
        if os.path.isabs(model_file) and os.path.exists(model_file):
            return model_file
        return os.path.join(self._model_dir(), model_file)

    # ------------------------------------------------------------------
    # (Restored) internal heavy-load + pipeline init helpers
    # These were missing; submit()/ensure_pipeline depend on them.
    # Mirrors original implementation style to keep interface parity
    # with QwenEdit1Backend (which has _init_pipeline + ensure_pipeline).
    # ------------------------------------------------------------------
    def _load_heavy(self):
        if self._torch is None:
            import torch as _torch
            self._torch = _torch
        if self._diffusers is None:
            import diffusers as _diff
            self._diffusers = _diff
        if self._scheduler_cls is None:
            from diffusers import DPMSolverMultistepScheduler as _DPM
            self._scheduler_cls = _DPM

    def _init_pipeline(self, model_path: str, *, pipe_kind: str, progress_cb: Optional[ProgressFn] = None) -> None:
        """
        Idempotently initialize (or reuse) the required SDXL pipeline variant.
        pipe_kind: 'text' or 'i2i'.
        """
        with self._init_lock:
            self._load_heavy()
            self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
            pipe_kind = pipe_kind.lower().strip()
            if pipe_kind == "text":
                # Already loaded with same model? Reuse
                if self._pipe_text is not None and self._current_model_path_text == model_path:
                    return
                if progress_cb: progress_cb(2)
                P = self._diffusers.StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=self._torch.float16 if self._device == "cuda" else self._torch.float32,
                )
                if progress_cb: progress_cb(3)
                P.scheduler = self._scheduler_cls.from_config(P.scheduler.config)
                if progress_cb: progress_cb(4)
                P = P.to(self._device)
                if progress_cb: progress_cb(5)
                try:
                    P.set_progress_bar_config(disable=True)
                except Exception:
                    pass
                self._pipe_text = P
                self._current_model_path_text = model_path
                self._current_adapter_names_text = []
            else:
                if self._pipe_i2i is not None and self._current_model_path_i2i == model_path:
                    return
                if progress_cb: progress_cb(2)
                P = getattr(self._diffusers, "StableDiffusionXLImg2ImgPipeline").from_single_file(
                    model_path,
                    torch_dtype=self._torch.float16 if self._device == "cuda" else self._torch.float32,
                )
                if progress_cb: progress_cb(3)
                P.scheduler = self._scheduler_cls.from_config(P.scheduler.config)
                if progress_cb: progress_cb(4)
                P = P.to(self._device)
                if progress_cb: progress_cb(5)
                try:
                    P.set_progress_bar_config(disable=True)
                except Exception:
                    pass
                self._pipe_i2i = P
                self._current_model_path_i2i = model_path
                self._current_adapter_names_i2i = []

    def _apply_sdxl_loras(self, pipeline, lora_paths: List[str], *, which: str) -> None:
        """
        Re-apply list of LoRA weights (clears previous first).
        which: 'text' or 'i2i' (tracks which adapter list to update).
        """
        if pipeline is None:
            return
        try:
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
        except Exception:
            pass
        if not lora_paths:
            if which == "text":
                self._current_adapter_names_text = []
            else:
                self._current_adapter_names_i2i = []
            return
        adapter_names: List[str] = []
        for idx, path in enumerate(lora_paths):
            if not (isinstance(path, str) and os.path.isfile(path)):
                continue
            adapter = sanitize_adapter_name(path, idx)
            try:
                pipeline.load_lora_weights(path, adapter_name=adapter)
                adapter_names.append(adapter)
            except Exception:
                pass
        if not adapter_names:
            if which == "text":
                self._current_adapter_names_text = []
            else:
                self._current_adapter_names_i2i = []
            return
        try:
            pipeline.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))
        except Exception:
            # Fallback single adapter if multi fails
            try:
                pipeline.set_adapters(adapter_names[0], adapter_weights=[1.0])
                adapter_names = [adapter_names[0]]
            except Exception:
                adapter_names = []
        if which == "text":
            self._current_adapter_names_text = adapter_names
        else:
            self._current_adapter_names_i2i = adapter_names

    def unload(self) -> None:
        """
        Release pipelines & GPU memory (parity with Qwen backend unload()).
        """
        with self._init_lock:
            self._pipe_text = None
            self._pipe_i2i = None
            self._current_model_path_text = None
            self._current_model_path_i2i = None
            self._current_adapter_names_text = []
            self._current_adapter_names_i2i = []
            try:
                if self._torch is not None and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:
                pass
