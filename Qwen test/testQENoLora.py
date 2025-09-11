import os
import time
import math
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# Optional: install psutil to log CPU RAM (pip install psutil)
try:
    import psutil
except Exception:
    psutil = None

# Use the new allocator hint (helps fragmentation). Avoid deprecated CUDA var.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
# os.environ.setdefault("HF_HUB_OFFLINE", "1")  # uncomment to force offline

def bf16_supported() -> bool:
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def enable_sdpa():
    # Prefer only memory-efficient SDPA; avoid FLASH to prevent long first-call stalls on some setups
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        print("[test] SDPA: EFFICIENT attention enabled (torch.nn.attention.sdpa_kernel)")
        return
    except Exception as e:
        print(f"[test] SDPA new API skipped: {e}")
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
        print("[test] SDPA: mem_efficient enabled (torch.backends.cuda.sdp_kernel)")
    except Exception as e:
        print(f"[test] SDPA setup skipped: {e}")

def pretty_bytes(n: int) -> str:
    try:
        for unit in ["B","KB","MB","GB","TB"]:
            if n < 1024:
                return f"{n:.0f}{unit}"
            n /= 1024.0
        return f"{n:.1f}PB"
    except Exception:
        return str(n)

def gpu_mem_info(device_idx: int | None = None):
    if not torch.cuda.is_available():
        return None
    try:
        d = torch.cuda.current_device() if device_idx is None else device_idx
        props = torch.cuda.get_device_properties(d)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(d)
        allocated = torch.cuda.memory_allocated(d)
        free_cache = total - reserved
        free_global = total_global = None
        try:
            free_global, total_global = torch.cuda.mem_get_info(d)
        except Exception:
            pass
        return {
            "device": d,
            "total": total,
            "reserved": reserved,
            "allocated": allocated,
            "free_cache": free_cache,
            "free_global": free_global,
            "total_global": total_global,
        }
    except Exception:
        return None

def cpu_mem_info():
    try:
        out = {}
        if psutil is not None:
            vm = psutil.virtual_memory()
            out["sys_total"] = vm.total
            out["sys_used"] = vm.total - vm.available
            proc = psutil.Process(os.getpid())
            out["proc_rss"] = proc.memory_info().rss
        return out
    except Exception:
        return {}

def log_mem(label: str):
    g = gpu_mem_info()
    c = cpu_mem_info()
    parts = [f"[mem] {label}:"]
    if g:
        parts.append(
            f"GPU[{g['device']}] alloc={pretty_bytes(g['allocated'])} "
            f"reserved={pretty_bytes(g['reserved'])} "
            f"free_cache={pretty_bytes(max(0, g['free_cache']))} "
            f"total={pretty_bytes(g['total'])}"
        )
        if g and g.get("free_global") is not None:
            parts.append(f"cuda_free={pretty_bytes(g['free_global'])}/{pretty_bytes(g['total_global'])}")
    if c:
        if "proc_rss" in c:
            parts.append(f"CPU(proc_rss)={pretty_bytes(c['proc_rss'])}")
        if "sys_used" in c and "sys_total" in c:
            parts.append(f"CPU(sys)={pretty_bytes(c['sys_used'])}/{pretty_bytes(c['sys_total'])}")
    print("  ".join(parts))

def main():
    app_t0 = time.perf_counter()
    app_dir = Path(__file__).resolve().parent

    # Local model snapshot (folder with model_index.json, processor/, tokenizer/, transformer/, vae/, etc.)
    model_dir = Path(os.environ.get("NT6_QWEN_MODEL", r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit"))
    local_only = model_dir.is_dir()

    # Input image
    src_path = app_dir / "lara.jpg"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source image: {src_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = bf16_supported()

    # QUICK A/B: set use_fp16=True to test fp16 vs bf16 if stalls persist on FLASH/EFFICIENT backends
    use_fp16 = False
    desired_dtype = (
        torch.float16 if (device == "cuda" and use_fp16)
        else (torch.bfloat16 if (device == "cuda" and bf16_ok) else (torch.float16 if device == "cuda" else torch.float32))
    )

    enable_sdpa()

    print("=== Qwen Image Edit (no LoRA, CPU offload enabled) ===")
    print(f"- model_dir: {model_dir} (exists: {model_dir.exists()}, local_only={local_only})")
    print(f"- src_path : {src_path} (exists: {src_path.exists()})")
    print(f"- device   : {device}")
    print(f"- CUDA bf16 supported: {bf16_ok}")
    print(f"- torch_dtype at load: {desired_dtype}")

    # Load in reduced precision. QwenImageEditPipeline ignores dtype=; must use torch_dtype=.
    load_t0 = time.perf_counter()
    pipe = QwenImageEditPipeline.from_pretrained(
        str(model_dir if local_only else "Qwen/Qwen-Image-Edit"),
        torch_dtype=desired_dtype,   # keep torch_dtype despite deprecation warning
        use_safetensors=True,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    ).to(device)

    # Tweaks
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    try:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            try: pipe.vae.enable_tiling()
            except Exception: pass
            try: pipe.vae.enable_slicing()
            except Exception: pass
    except Exception:
        pass

    # Allow CPU offload for stability on large footprints
    enable_offload = True
    if enable_offload:
        try:
            pipe.enable_model_cpu_offload()
            print("[test] enable_model_cpu_offload() enabled")
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
                print("[test] enable_sequential_cpu_offload() enabled")
            except Exception:
                pass

    load_t1 = time.perf_counter()
    print(f"[timing] model_loaded_sec={load_t1 - app_t0:.3f} (setup={load_t1 - load_t0:.3f} after from_pretrained)")
    log_mem("after model load")

    # Prepare input; downscale if needed
    img = Image.open(src_path).convert("RGB")
    max_side = 768
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)

    steps = 8
    true_cfg = 1.0
    neg = " "
    base_prompt_suffix = "high quality portrait, cinematic lighting"

    variants = [
        ("blonde haired", app_dir / "lara_qwen_edit_blonde.jpg"),
        ("red haired",    app_dir / "lara_qwen_edit_red.jpg"),
        ("pink haired",   app_dir / "lara_qwen_edit_pink.jpg"),
    ]

    gen = torch.Generator(device=device) if device == "cuda" else None
    if gen is not None:
        gen.manual_seed(0)

    for idx, (hair_text, out_path) in enumerate(variants, start=1):
        full_prompt = f"Replace the woman's hair to {hair_text}, {base_prompt_suffix}"
        print(f"[test] Generating {idx}/3: '{full_prompt}' -> {out_path.name}")
        if torch.cuda.is_available():
            try: torch.cuda.reset_peak_memory_stats()
            except Exception: pass

        log_mem(f"before gen {idx}")

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = QwenImageEditPipeline.__call__(
                pipe,
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

        if torch.cuda.is_available():
            try:
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_res = torch.cuda.max_memory_reserved()
                print(f"[mem] peak_gpu_alloc={pretty_bytes(peak_alloc)} peak_gpu_reserved={pretty_bytes(peak_res)}")
            except Exception:
                pass
        log_mem(f"after gen {idx}")

    print("[test] Done.")

if __name__ == "__main__":
    main()