import os
import math
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# ========================
# Config flags (defaults)
# ========================
# Prevent GPU->CPU offload for speed (keep weights resident on GPU)
PREVENT_OFFLOAD: bool = False
# Use the pipeline's default scheduler (recommended for Qwen Image Edit)
USE_DEFAULT_SCHEDULER: bool = True
# When offload is prevented, force FP16 to avoid bf16 first-call stalls on some setups
FORCE_FP16_WHEN_NO_OFFLOAD: bool = True
# When offload is prevented, set allocator hints to reduce fragmentation
APPLY_ALLOC_CONF_WHEN_NO_OFFLOAD: bool = True
# New: preload all LoRAs once and switch adapters instead of unload/load
PRELOAD_LORAS: bool = True
# New: warm-up only once (global), not per adapter
WARMUP_ONCE: bool = True
# New: optionally fuse LoRA during inference for a small speed boost (if supported)
FUSE_AT_INFERENCE: bool = False
# ========================

# Optional: install psutil to log CPU RAM (pip install psutil)
try:
    import psutil  # noqa: F401
except Exception:
    psutil = None

# Apply allocator hints only if we prevent offload (to improve large contiguous allocations)
if PREVENT_OFFLOAD and APPLY_ALLOC_CONF_WHEN_NO_OFFLOAD:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

def bf16_supported() -> bool:
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def enable_sdpa():
    # Prefer only memory-efficient attention to avoid potential FLASH stalls
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        print("[test] SDPA: EFFICIENT attention enabled (torch.nn.attention.sdpa_kernel)")
        return "efficient"
    except Exception as e:
        print(f"[test] SDPA new API skipped: {e}")
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
        print("[test] SDPA: mem_efficient enabled (torch.backends.cuda.sdp_kernel)")
        return "efficient"
    except Exception as e:
        print(f"[test] SDPA setup skipped: {e}")
        return "none"

def build_flowmatch_scheduler(pipe):
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

# ---- Memory logging helpers ----
def pretty_bytes(n: int) -> str:
    try:
        for unit in ["B","KB","MB","GB","TB"]:
            if n < 1024:
                return f"{n:.0f}{unit}"
            n /= 1024.0
        return f"{n:.1f}PB"
    except Exception:
        return str(n)

def gpu_mem_info(device_idx=None):
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
        if g and g["free_global"] is not None:
            parts.append(f"cuda_free={pretty_bytes(g['free_global'])}/{pretty_bytes(g['total_global'])}")
    if c:
        if "proc_rss" in c:
            parts.append(f"CPU(proc_rss)={pretty_bytes(c['proc_rss'])}")
        if "sys_used" in c and "sys_total" in c:
            parts.append(f"CPU(sys)={pretty_bytes(c['sys_used'])}/{pretty_bytes(c['sys_total'])}")
    print("  ".join(parts))
# --------------------------------

def main():
    app_t0 = time.perf_counter()
    app_dir = Path(__file__).resolve().parent

    # Echo config
    print(f"[cfg] PREVENT_OFFLOAD={PREVENT_OFFLOAD}  USE_DEFAULT_SCHEDULER={USE_DEFAULT_SCHEDULER}  "
          f"FORCE_FP16_WHEN_NO_OFFLOAD={FORCE_FP16_WHEN_NO_OFFLOAD}  "
          f"APPLY_ALLOC_CONF_WHEN_NO_OFFLOAD={APPLY_ALLOC_CONF_WHEN_NO_OFFLOAD}  "
          f"PRELOAD_LORAS={PRELOAD_LORAS}  WARMUP_ONCE={WARMUP_ONCE}  FUSE_AT_INFERENCE={FUSE_AT_INFERENCE}")
    if PREVENT_OFFLOAD and APPLY_ALLOC_CONF_WHEN_NO_OFFLOAD:
        print(f"[cfg] PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF')}")

    model_dir = Path(os.environ.get("NT6_QWEN_MODEL", r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit"))
    local_only = model_dir.is_dir()

    lora_dir = app_dir / "Qwen-Image-Lightning"
    lora_files = [
        lora_dir / "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
        lora_dir / "Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
        lora_dir / "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        lora_dir / "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors",
    ]

    # Build named adapters
    adapters = {
        "l4_bf16": lora_files[0],
        "l4_full": lora_files[1],
        "l8_bf16": lora_files[2],
        "l8_full": lora_files[3],
    }
    adapter_steps = { "l4_bf16": 4, "l4_full": 4, "l8_bf16": 8, "l8_full": 8 }

    src_path = app_dir / "lara.jpg"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source image: {src_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = bf16_supported()

    # DType policy
    if device == "cuda":
        if PREVENT_OFFLOAD and FORCE_FP16_WHEN_NO_OFFLOAD:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        torch_dtype = torch.float32

    sdpa_mode = enable_sdpa()

    print("=== Qwen Image Edit (Lightning, memory-optimized, 4 LoRAs) ===")
    print(f"- model_dir: {model_dir} (exists: {model_dir.exists()}, local_only={local_only})")
    print(f"- lora_dir : {lora_dir} (exists: {lora_dir.exists()})")
    print(f"- src_path : {src_path} (exists: {src_path.exists()})")
    print(f"- device   : {device}")
    print(f"- CUDA bf16 supported: {bf16_ok}")
    print(f"- torch_dtype at load: {torch_dtype}, sdpa_mode={sdpa_mode}")

    load_t0 = time.perf_counter()
    pipe = QwenImageEditPipeline.from_pretrained(
        str(model_dir if local_only else "Qwen/Qwen-Image-Edit"),
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    ).to(device)

    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    try:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            try: pipe.vae.enable_tiling()
            except Exception: pass
            try: pipe.vae.enable_slicing()
            except Exception: pass
            print("[test] VAE tiling/slicing enabled")
    except Exception:
        pass

    # Offload policy
    enable_offload = not PREVENT_OFFLOAD
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

    try:
        pipe.unet.to(memory_format=torch.channels_last)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # Scheduler policy
    if USE_DEFAULT_SCHEDULER:
        print("[test] Using pipeline default scheduler")
    else:
        build_flowmatch_scheduler(pipe)

    load_t1 = time.perf_counter()
    print(f"[timing] model_loaded_sec={load_t1 - app_t0:.3f} (setup={load_t1 - load_t0:.3f} after from_pretrained)")
    log_mem("after model load")

    img = Image.open(src_path).convert("RGB")
    max_side = 768
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)

    true_cfg = 1.0
    neg = " "
    instruction = "Change the woman's hair to Blonde"

    # Optionally preload all adapters once
    loaded_adapters = []
    if PRELOAD_LORAS:
        print("[test] Preloading LoRAs...")
        for name, path in adapters.items():
            if not path.exists():
                print(f"[test] Skipping missing LoRA: {path}")
                continue
            try:
                pipe.load_lora_weights(str(path), adapter_name=name)
                loaded_adapters.append(name)
            except Exception as e:
                print(f"[test] preload failed for {name}: {e}")
        print(f"[test] Preloaded adapters: {loaded_adapters}")

    metrics = []

    # Global warm-up (once) to amortize JIT/init costs
    did_global_warmup = False
    if WARMUP_ONCE:
        warm_adapter = loaded_adapters[0] if loaded_adapters else None
        if warm_adapter:
            try:
                pipe.set_adapters(warm_adapter, 1.0)
            except Exception:
                pass
        print("[test] Global warm-up (1 step)...")
        gen_warm = torch.Generator(device=device) if device == "cuda" else None
        if gen_warm is not None:
            gen_warm.manual_seed(12345)
        warm0 = time.perf_counter()
        with torch.inference_mode():
            _ = QwenImageEditPipeline.__call__(
                pipe,
                image=img,
                prompt=instruction,
                negative_prompt=neg,
                true_cfg_scale=true_cfg,
                num_inference_steps=1,
                generator=gen_warm,
            )
        warm1 = time.perf_counter()
        print(f"[timing] global_warmup_sec={warm1 - warm0:.3f}")
        did_global_warmup = True

    # Iterate each adapter
    for name, path in adapters.items():
        steps = adapter_steps[name]
        tag = name
        out_path = (app_dir / f"lara_blonde_{steps}steps_{name}.jpg")
        print(f"\n[test] Adapter: {name} (steps={steps})")

        # Ensure adapter is ready
        if PRELOAD_LORAS and name in loaded_adapters:
            try:
                pipe.set_adapters(name, 1.0)
                print(f"[test] Switched to adapter: {name}")
            except Exception:
                pass
        else:
            # Fallback: load on demand
            try:
                pipe.load_lora_weights(str(path), adapter_name=name)
                pipe.set_adapters(name, 1.0)
                print(f"[test] Loaded adapter on demand: {name}")
            except Exception as e:
                print(f"[test] LoRA load failed ({name}): {e}; skipping")
                continue

        # Optional fuse/unfuse for small speed gain (if supported)
        fused = False
        if FUSE_AT_INFERENCE and hasattr(pipe, "fuse_lora"):
            try:
                pipe.fuse_lora()
                fused = True
                print("[test] fuse_lora() applied")
            except Exception:
                fused = False

        # Reset GPU peak stats
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # Timed run (synchronized)
        print(f"[test] Generating -> {out_path.name}")
        log_mem(f"before gen ({tag})")
        cpu_rss_before = None
        if psutil is not None:
            try:
                cpu_rss_before = psutil.Process(os.getpid()).memory_info().rss
            except Exception:
                cpu_rss_before = None

        gen_main = torch.Generator(device=device) if device == "cuda" else None
        if gen_main is not None:
            gen_main.manual_seed(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = QwenImageEditPipeline.__call__(
                pipe,
                image=img,
                prompt=instruction,
                negative_prompt=neg,
                true_cfg_scale=true_cfg,
                num_inference_steps=steps,
                generator=gen_main,
            )
            out_img = result.images[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gen_sec = t1 - t0

        out_img.save(out_path, "JPEG", quality=92)

        # Unfuse back if needed
        if fused and hasattr(pipe, "unfuse_lora"):
            try:
                pipe.unfuse_lora()
                print("[test] unfuse_lora() applied")
            except Exception:
                pass

        # Memory peaks
        peak_alloc = peak_res = None
        if torch.cuda.is_available():
            try:
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_res = torch.cuda.max_memory_reserved()
            except Exception:
                pass

        cpu_rss_after = None
        if psutil is not None:
            try:
                cpu_rss_after = psutil.Process(os.getpid()).memory_info().rss
            except Exception:
                cpu_rss_after = None
        cpu_rss_delta = (cpu_rss_after - cpu_rss_before) if (cpu_rss_before is not None and cpu_rss_after is not None) else None

        print(f"[timing] gen_sec={gen_sec:.3f} wrote={out_path}")
        if peak_alloc is not None:
            print(f"[mem] peak_gpu_alloc={pretty_bytes(peak_alloc)} peak_gpu_reserved={pretty_bytes(peak_res) if peak_res is not None else 'n/a'}")
        log_mem(f"after gen ({tag})")

        s_per_step = gen_sec / max(1, steps)
        it_per_s = steps / gen_sec if gen_sec > 0 else float("inf")
        metrics.append({
            "name": name,
            "steps": steps,
            "gen_sec": gen_sec,
            "s_per_step": s_per_step,
            "it_per_s": it_per_s,
            "peak_alloc": peak_alloc,
            "peak_reserved": peak_res,
            "cpu_rss_delta": cpu_rss_delta,
        })

    # Summary
    if metrics:
        print("\n=== Summary (fastest first) ===")
        metrics_sorted = sorted(metrics, key=lambda m: m["gen_sec"])
        for rank, m in enumerate(metrics_sorted, start=1):
            pa = pretty_bytes(m["peak_alloc"]) if m["peak_alloc"] is not None else "n/a"
            pr = pretty_bytes(m["peak_reserved"]) if m["peak_reserved"] is not None else "n/a"
            cr = pretty_bytes(m["cpu_rss_delta"]) if m["cpu_rss_delta"] is not None else "n/a"
            print(
                f"{rank:>2}. {m['name']} | steps={m['steps']} | gen={m['gen_sec']:.3f}s "
                f"(s/step={m['s_per_step']:.3f}, it/s={m['it_per_s']:.2f}) | "
                f"peak_alloc={pa} peak_res={pr} | cpu_rss_delta={cr}"
            )

    print("\n[test] Done.")

if __name__ == "__main__":
    main()