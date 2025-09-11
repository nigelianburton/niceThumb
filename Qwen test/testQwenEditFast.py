import os
import math
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# Reduce allocator fragmentation (optional but helps with large models)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# Uncomment if you want to ensure fully offline after download
# os.environ.setdefault("HF_HUB_OFFLINE", "1")

def bf16_supported() -> bool:
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def enable_sdpa():
    try:
        # Prefer the new SDPA kernel API when available
        from torch.nn.attention import sdpa_kernel
        sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        print("[test] SDPA kernel: flash/mem_efficient enabled")
    except Exception as e:
        print(f"[test] SDPA kernel setup skipped: {e}")

def build_flowmatch_scheduler(pipe):
    """Apply FlowMatch Euler scheduler settings (safe fallback)."""
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        base_cfg = dict(getattr(pipe.scheduler, "config", {}))
        fm_cfg = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        base_cfg.update(fm_cfg)
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_cfg)
        print("[test] FlowMatchEulerDiscreteScheduler applied")
    except Exception as e:
        print(f"[test] FlowMatch scheduler not applied (fallback to default): {e}")

def main():
    app_t0 = time.perf_counter()
    app_dir = Path(__file__).resolve().parent

    # Local model snapshot (folder with model_index.json and subfolders)
    model_dir = Path(os.environ.get("NT6_QWEN_MODEL", r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit"))
    local_only = model_dir.is_dir()

    # Lightning LoRA (relative path, as provided)
    lora_path = app_dir / "Qwen-Image-Lightning" / "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"

    # Input image (I2I)
    src_path = app_dir / "lara.jpg"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source image: {src_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = bf16_supported()
    torch_dtype = torch.bfloat16 if (device == "cuda" and bf16_ok) else (torch.float16 if device == "cuda" else torch.float32)

    enable_sdpa()

    print("=== Qwen Image Edit (Lightning, memory-optimized) ===")
    print(f"- model_dir: {model_dir} (exists: {model_dir.exists()}, local_only={local_only})")
    print(f"- lora_path: {lora_path} (exists: {lora_path.exists()})")
    print(f"- src_path : {src_path} (exists: {src_path.exists()})")
    print(f"- device   : {device}")
    print(f"- CUDA bf16 supported: {bf16_ok}")
    print(f"- torch_dtype at load: {torch_dtype}")

    # Load pipeline (critical: use torch_dtype at load time)
    load_t0 = time.perf_counter()
    pipe = QwenImageEditPipeline.from_pretrained(
        str(model_dir if local_only else "Qwen/Qwen-Image-Edit"),
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )
    try:
        pipe = pipe.to(device)
    except Exception as e:
        print(f"[test] .to({device}) failed: {e}")

    # Memory optimizations
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            try: pipe.vae.enable_tiling()
            except Exception: pass
            try: pipe.vae.enable_slicing()
            except Exception: pass
            print("[test] VAE tiling/slicing enabled")
    except Exception:
        pass
    try:
        from diffusers.utils import is_xformers_available
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
            print("[test] xFormers attention enabled")
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
        print("[test] enable_model_cpu_offload() enabled")
    except Exception:
        try:
            pipe.enable_sequential_cpu_offload()
            print("[test] enable_sequential_cpu_offload() enabled")
        except Exception:
            pass
    try:
        pipe.unet.to(memory_format=torch.channels_last)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # Apply FlowMatch Euler scheduler (if available)
    build_flowmatch_scheduler(pipe)

    # Log: time to model loaded (ready to generate)
    load_t1 = time.perf_counter()
    print(f"[timing] model_loaded_sec={load_t1 - app_t0:.3f} (setup={load_t1 - load_t0:.3f} after from_pretrained)")

    # Attach Lightning LoRA (optional)
    if lora_path.exists():
        print("[test] Attaching Lightning LoRA...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe.load_lora_weights(str(lora_path))
            try:
                pipe.set_adapters("default", 1.0)
            except Exception:
                pass
        except Exception as e:
            print(f"[test] LoRA load failed: {e} (hint: pip install peft; if OOM, rely on CPU offload + VAE tiling)")
    else:
        print("[test] LoRA file not found; continuing without LoRA")

    # Prepare input; downscale if needed to keep memory reasonable
    img = Image.open(src_path).convert("RGB")
    max_side = 768
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)

    # Three edits with required prompts and distinct outputs
    base_prompt_suffix = "high quality portrait, cinematic lighting"
    hair_variants = [
        ("a schoolgirl uniform", app_dir / "lara_qwen_edit_blonde.jpg"),
        ("a cute little princess dress",    app_dir / "lara_qwen_edit_red.jpg"),
        ("a french maid dress",   app_dir / "lara_qwen_edit_pink.jpg"),
    ]

    steps = 8
    true_cfg = 1.0
    neg = " "
    gen = torch.Generator(device=device) if device == "cuda" else None
    if gen is not None:
        gen.manual_seed(0)

    for hair_text, out_path in hair_variants:
        full_prompt = f"Replace the woman's costume to {hair_text}, {base_prompt_suffix}"
        print(f"[test] Generating: '{full_prompt}' -> {out_path.name}")
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                image=img,
                prompt=full_prompt,
                negative_prompt=neg,
                true_cfg_scale=true_cfg,
                num_inference_steps=steps,
                generator=gen,
            )
            out_img = result.images[0]
        t1 = time.perf_counter()
        out_img.save(out_path, "JPEG", quality=92)
        print(f"[timing] gen_sec={t1 - t0:.3f} wrote={out_path}")

    print("[test] Done.")

if __name__ == "__main__":
    main()