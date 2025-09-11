import os
import sys
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from diffusers import QwenImageEditPipeline

# ---------------- Environment & Configuration ----------------
DEFAULT_MODEL_DIR = os.environ.get("QWEN_DEFAULT_DIR", r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit")
FP8_FILE = os.environ.get("QWEN_FP8_FILE", r"C:\_MODELS-SD\Qwen\Qwen-Image_Edit-FP8\qwen_image_edit_fp8_e4m3fn.safetensors")
INPUT_IMAGE_PATH = os.environ.get("QWEN_TEST_IMAGE", r"T:\1.jpg")
PROMPT = os.environ.get("QWEN_TEST_PROMPT", "Replace the model's dress with a pink dress")

RUN_DEFAULT = os.environ.get("RUN_DEFAULT", "1").lower() in ("1", "true", "yes")
RUN_FP8 = os.environ.get("RUN_FP8", "1").lower() in ("1", "true", "yes")
RUN_QUANT = os.environ.get("RUN_QUANT", "1").lower() in ("1", "true", "yes")

DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "8"))
LIGHTNING_STEPS = int(os.environ.get("LIGHTNING_STEPS", "4"))
SEED = int(os.environ.get("SEED", "42"))

FORCE_BF16 = os.environ.get("FORCE_BF16", "0").lower() in ("1", "true", "yes")  # legacy
FORCE_DTYPE = os.environ.get("FORCE_DTYPE", "").strip().lower()  # fp16 | bf16 | fp32
SKIP_STATS = os.environ.get("SKIP_STATS", "0").lower() in ("1", "true", "yes")
CAPTURE_ERRORS = os.environ.get("CAPTURE_ERRORS", "1").lower() not in ("0", "false", "no")
ENABLE_EARLY_OFFLOAD = os.environ.get("ENABLE_EARLY_OFFLOAD", "0").lower() in ("1", "true", "yes")
ALLOW_FALLBACK = os.environ.get("ALLOW_FALLBACK", "1").lower() in ("1", "true", "yes")

LIGHTNING_LORA_REPO = os.environ.get("LIGHTNING_LORA_REPO", "lightx2v/Qwen-Image-Lightning")
LIGHTNING_LORA_WEIGHT = os.environ.get("LIGHTNING_LORA_WEIGHT", "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors")
LOCAL_LORA_DIR = os.environ.get("LOCAL_LORA_DIR", r"C:\_CONDA\niceThumb\Qwen-Image-Lightning")

# ---------------- Data Class ----------------
@dataclass
class RunResult:
    label: str
    load_time_s: float
    gen_time_s: float
    peak_mem_gb: Optional[float]
    image_path: Optional[str]
    extra_info: str = ""
    ok: bool = True
    error: Optional[str] = None
    crash: bool = False
    log: str = ""

# ---------------- Utilities ----------------
def _select_dtype(device: torch.device) -> torch.dtype:
    if FORCE_DTYPE in ("fp32", "float32"):
        return torch.float32
    if FORCE_DTYPE in ("bf16", "bfloat16"):
        return torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    if FORCE_DTYPE in ("fp16", "float16", "half"):
        return torch.float16
    if device.type == "cuda":
        if FORCE_BF16 and hasattr(torch, "bfloat16"):
            return torch.bfloat16
        # prefer bfloat16 first for stability on Ada
        if hasattr(torch, "bfloat16"):
            return torch.bfloat16
        return torch.float16
    return torch.float32

def _memory_reset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _mem_snapshot(tag: str):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated()/1024**3
    resv  = torch.cuda.memory_reserved()/1024**3
    peak  = torch.cuda.max_memory_allocated()/1024**3
    print(f"[mem][{tag}] alloc={alloc:.2f}GB reserved={resv:.2f}GB peak={peak:.2f}GB")

def _load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def _save_temp_image(img: Image.Image, prefix: str) -> str:
    out_dir = os.path.join(os.getcwd(), "test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{prefix}_{int(time.time()*1000)}.png"
    p = os.path.join(out_dir, name)
    img.save(p)
    return p

def _param_stats(module: torch.nn.Module, label: str):
    if SKIP_STATS:
        print(f"[stats][{label}] skipped")
        return
    tot = 0
    by: Dict[str, int] = {}
    for p in module.parameters():
        tot += p.numel()
        by[str(p.dtype)] = by.get(str(p.dtype), 0) + p.numel()
    print(f"[stats][{label}] total={tot:,}")
    for dt, elems in sorted(by.items()):
        size = 2 if ("16" in dt or "bfloat16" in dt) else (1 if "int8" in dt else 4)
        mb = elems * size / 1024**2
        print(f"  {dt:<14} elems={elems:>12,} approx={mb:>10.2f}MB")

def _image_black_or_nan(img: Image.Image) -> Tuple[bool, Dict[str, float]]:
    arr = np.array(img)
    if arr.ndim == 3:
        arrf = arr.astype(np.float32)
    else:
        arrf = arr[..., None].astype(np.float32)
    mean = float(arrf.mean())
    std = float(arrf.std())
    isnan = np.isnan(arrf).any()
    # define "black" as extremely low variance + low mean or nan presence
    is_black = (mean < 2.0 and std < 5.0) or isnan
    return is_black, {"mean": mean, "std": std, "nan": float(isnan)}

def _generate(pipe: QwenImageEditPipeline, steps: int, seed: int, label: str, image: Image.Image):
    print(f"[gen][{label}] steps={steps} size={image.size}")
    _mem_snapshot(f"{label}-pre-gen")
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(seed)
    exec_dtype = getattr(pipe, "unet", None)
    exec_dtype = exec_dtype.dtype if exec_dtype and hasattr(exec_dtype, "dtype") else torch.float16
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=exec_dtype):
                out = pipe(image, PROMPT, num_inference_steps=steps, generator=gen)
        else:
            out = pipe(image, PROMPT, num_inference_steps=steps, generator=gen)
    _mem_snapshot(f"{label}-post-gen")
    return out.images[0]

def _maybe_offload(pipe: QwenImageEditPipeline, label: str):
    if ENABLE_EARLY_OFFLOAD:
        try:
            pipe.enable_model_cpu_offload()
            print(f"[offload]{label} early offload enabled")
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
                print(f"[offload]{label} sequential offload fallback")
            except Exception as e:
                print(f"[offload]{label}[warn] offload not applied: {e}")

# ---------------- Base loaders ----------------
def _load_base_pipeline(dtype: torch.dtype) -> QwenImageEditPipeline:
    return QwenImageEditPipeline.from_pretrained(
        DEFAULT_MODEL_DIR,
        torch_dtype=dtype,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

def _inject_fp8(pipe: QwenImageEditPipeline, dtype: torch.dtype) -> int:
    from safetensors.torch import load_file
    state = load_file(FP8_FILE)
    casted = 0
    for k, v in list(state.items()):
        if "float8" in str(v.dtype).lower():
            state[k] = v.to(dtype)
            casted += 1
    tgt = getattr(pipe, "transformer", None)
    if tgt is None:
        raise RuntimeError("Transformer missing for FP8 injection")
    try:
        tgt.load_state_dict(state, strict=False, assign=True)
    except TypeError:
        tgt.load_state_dict(state, strict=False)
    print(f"[fp8] injected casted={casted}")
    return casted

# ---------------- Run patterns ----------------
def _run_default_internal(dtype: torch.dtype) -> Tuple[RunResult, Optional[QwenImageEditPipeline]]:
    label = f"Default({dtype})"
    rr = RunResult(label, 0, 0, None, None)
    start = time.perf_counter()
    pipe = _load_base_pipeline(dtype)
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    if ENABLE_EARLY_OFFLOAD:
        _maybe_offload(pipe, label)
    rr.load_time_s = time.perf_counter() - start
    _param_stats(pipe.transformer, "default.transformer")
    img = _load_image(INPUT_IMAGE_PATH)
    t1 = time.perf_counter()
    out = _generate(pipe, DEFAULT_STEPS, SEED, label, img)
    rr.gen_time_s = time.perf_counter() - t1
    black, stats = _image_black_or_nan(out)
    print(f"[check]{label} black={black} stats={stats}")
    if torch.cuda.is_available():
        rr.peak_mem_gb = torch.cuda.max_memory_allocated()/1024**3
    rr.image_path = _save_temp_image(out, "default")
    rr.extra_info = f"steps={DEFAULT_STEPS} dtype={dtype} black={black}"
    rr.ok = True
    return rr, (pipe if black else None)

def run_default() -> RunResult:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_dtype = _select_dtype(dev)
    print(f"[run]Default initial dtype={initial_dtype} early_offload={ENABLE_EARLY_OFFLOAD}")
    _memory_reset()
    rr, pipe_for_fallback = _run_default_internal(initial_dtype)
    if pipe_for_fallback and ALLOW_FALLBACK:
        print("[fallback] default run produced black/NaN image; retrying with safer dtype fp32...")
        try:
            # Rebuild pipeline in higher precision for reliability
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            rr_fp32, _ = _run_default_internal(torch.float32)
            if not rr_fp32.ok or rr_fp32.image_path is None:
                print("[fallback] fp32 retry failed; keeping original result")
            else:
                rr_fp32.label = "Default(fp32-fallback)"
                return rr_fp32
        except Exception as e:
            print(f"[fallback][error] {type(e).__name__}: {e}")
    return rr

def run_fp8() -> RunResult:
    if not os.path.isfile(FP8_FILE):
        return RunResult("FP8", 0, 0, None, None, ok=False, error=f"FP8 file missing: {FP8_FILE}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_dtype = _select_dtype(dev)
    label = f"FP8({initial_dtype})"
    print(f"[run]{label} early_offload={ENABLE_EARLY_OFFLOAD}")
    _memory_reset()
    rr = RunResult(label, 0, 0, None, None)
    try:
        start = time.perf_counter()
        pipe = _load_base_pipeline(initial_dtype)
        try: pipe.set_progress_bar_config(disable=True)
        except Exception: pass
        casted = _inject_fp8(pipe, initial_dtype)
        if ENABLE_EARLY_OFFLOAD:
            _maybe_offload(pipe, label)
        rr.load_time_s = time.perf_counter() - start
        _param_stats(pipe.transformer, "fp8.transformer")
        img = _load_image(INPUT_IMAGE_PATH)
        t1 = time.perf_counter()
        out = _generate(pipe, DEFAULT_STEPS, SEED, label, img)
        rr.gen_time_s = time.perf_counter() - t1
        black, stats = _image_black_or_nan(out)
        print(f"[check]{label} black={black} stats={stats}")
        if black and ALLOW_FALLBACK:
            print("[fallback] FP8 path produced black/NaN; retrying with fp32 rebuild + reinjection")
            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                start2 = time.perf_counter()
                pipe2 = _load_base_pipeline(torch.float32)
                _inject_fp8(pipe2, torch.float32)
                if ENABLE_EARLY_OFFLOAD:
                    _maybe_offload(pipe2, label+"-fallback")
                _param_stats(pipe2.transformer, "fp8.fallback.transformer")
                img2 = _load_image(INPUT_IMAGE_PATH)
                t2 = time.perf_counter()
                out2 = _generate(pipe2, DEFAULT_STEPS, SEED, label+"-fallback", img2)
                rr.gen_time_s = time.perf_counter() - t2
                black2, stats2 = _image_black_or_nan(out2)
                print(f"[check]{label}-fallback black={black2} stats={stats2}")
                if not black2:
                    rr.load_time_s += (time.perf_counter() - start2)
                    out = out2
                    initial_dtype = torch.float32
                    black = False
                    rr.label = "FP8(fp32-fallback)"
            except Exception as e:
                print(f"[fallback][fp8][error] {type(e).__name__}: {e}")
        if torch.cuda.is_available():
            rr.peak_mem_gb = torch.cuda.max_memory_allocated()/1024**3
        rr.image_path = _save_temp_image(out, "fp8")
        rr.extra_info = f"steps={DEFAULT_STEPS} dtype={initial_dtype} casted={casted}"
        rr.ok = True
    except Exception as e:
        rr.ok = False
        rr.error = f"{type(e).__name__}: {e}"
        print(f"[run]{label}[error] {rr.error}")
        if not CAPTURE_ERRORS:
            raise
    return rr

def run_quant_lora() -> RunResult:
    label = "Quant+Lightning"
    rr = RunResult(label, 0, 0, None, None)
    if not RUN_QUANT:
        rr.ok = False
        rr.error = "Quant run disabled"
        return rr
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _select_dtype(dev)
    print(f"[run]{label} dtype={dtype}")
    _memory_reset()
    try:
        try:
            from diffusers import (
                QwenImageEditPipeline,
                QwenImageTransformer2DModel,
                BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            )
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                BitsAndBytesConfig as HFBitsAndBytesConfig
            )
        except ImportError as ie:
            rr.ok = False
            rr.error = f"bitsandbytes / quant deps missing: {ie}"
            return rr

        diff_q = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
        hf_q = HFBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

        start = time.perf_counter()
        transformer = QwenImageTransformer2DModel.from_pretrained(
            DEFAULT_MODEL_DIR,
            subfolder="transformer",
            quantization_config=diff_q,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
        ).to("cpu")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            DEFAULT_MODEL_DIR,
            subfolder="text_encoder",
            quantization_config=hf_q,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
        ).to("cpu")
        pipe = QwenImageEditPipeline.from_pretrained(
            DEFAULT_MODEL_DIR,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        try: pipe.set_progress_bar_config(disable=True)
        except Exception: pass
        try: pipe.enable_model_cpu_offload()
        except Exception:
            try: pipe.enable_sequential_cpu_offload()
            except Exception: pass
        rr.load_time_s = time.perf_counter() - start

        # LoRA
        lora_loaded = False
        local_candidate = os.path.join(LOCAL_LORA_DIR, LIGHTNING_LORA_WEIGHT)
        try:
            if os.path.isfile(local_candidate):
                pipe.load_lora_weights(LOCAL_LORA_DIR, weight_name=LIGHTNING_LORA_WEIGHT)
                lora_loaded = True
                print(f"[lora]{label} local {LIGHTNING_LORA_WEIGHT}")
            else:
                pipe.load_lora_weights(LIGHTNING_LORA_REPO, weight_name=LIGHTNING_LORA_WEIGHT)
                lora_loaded = True
                print(f"[lora]{label} repo {LIGHTNING_LORA_REPO}/{LIGHTNING_LORA_WEIGHT}")
        except Exception as le:
            print(f"[lora]{label}[warn] load failed: {le}")

        img = _load_image(INPUT_IMAGE_PATH)
        t1 = time.perf_counter()
        out = _generate(pipe, LIGHTNING_STEPS, SEED, label, img)
        rr.gen_time_s = time.perf_counter() - t1
        black, stats = _image_black_or_nan(out)
        print(f"[check]{label} black={black} stats={stats}")
        if torch.cuda.is_available():
            rr.peak_mem_gb = torch.cuda.max_memory_allocated()/1024**3
        rr.image_path = _save_temp_image(out, "quant_lora")
        rr.extra_info = f"4bit nf4 steps={LIGHTNING_STEPS} LoRA={'yes' if lora_loaded else 'no'} black={black}"
    except Exception as e:
        rr.ok = False
        rr.error = f"{type(e).__name__}: {e}"
        print(f"[run]{label}[error] {rr.error}")
        if not CAPTURE_ERRORS: raise
    return rr

# ---------------- Subprocess Isolation ----------------
def _worker(mode: str, queue: mp.Queue):
    try:
        if mode == "default":
            res = run_default()
        elif mode == "fp8":
            res = run_fp8()
        elif mode == "quant":
            res = run_quant_lora()
        else:
            raise ValueError(f"Unknown mode {mode}")
        queue.put(res)
    except Exception as e:
        tb = traceback.format_exc()
        queue.put(RunResult(mode, 0, 0, None, None, ok=False, crash=True,
                            error=f"{type(e).__name__}: {e}", log=tb))

def run_in_subprocess(mode: str, label: str) -> RunResult:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(mode, q))
    p.start()
    p.join()
    if not q.empty():
        r = q.get()
        r.label = label
        return r
    return RunResult(label, 0, 0, None, None, ok=False, crash=True, error="Process crashed (no report)")

# ---------------- Summary & UI ----------------
def summarize(results: List[RunResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"[{r.label}] ok={r.ok} crash={r.crash}")
        if r.ok:
            lines.append(f"  load={r.load_time_s:.2f}s gen={r.gen_time_s:.2f}s peak={r.peak_mem_gb:.2f}GB")
            lines.append(f"  extra: {r.extra_info}")
        else:
            lines.append(f"  error: {r.error}")
            if r.log:
                lines.extend(["    " + l for l in r.log.splitlines()[:6]])
    oks = [x for x in results if x.ok]
    if len(oks) > 1:
        base = oks[0]
        lines.append("--- Pairwise vs first ok ---")
        for o in oks[1:]:
            lines.append(f"{o.label} loadΔ={o.load_time_s-base.load_time_s:+.2f}s genΔ={o.gen_time_s-base.gen_time_s:+.2f}s peakΔ={(o.peak_mem_gb or 0)-(base.peak_mem_gb or 0):+.2f}GB")
    return "\n".join(lines)

def show_images(results: List[RunResult]):
    root = tk.Tk()
    root.title("Qwen Image Edit Comparison")
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    refs = []
    for idx, r in enumerate(results):
        col = tk.Frame(frame, bd=3, relief=tk.GROOVE)
        col.grid(row=0, column=idx, sticky="nsew")
        tk.Label(col, text=r.label, font=("Segoe UI", 12, "bold")).pack(pady=4)
        if r.ok and r.image_path and os.path.isfile(r.image_path):
            im = Image.open(r.image_path)
            max_dim = 900
            if im.width > max_dim or im.height > max_dim:
                s = min(max_dim/im.width, max_dim/im.height)
                im = im.resize((int(im.width*s), int(im.height*s)), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(im)
            refs.append(tk_img)
            tk.Label(col, image=tk_img).pack()
            tk.Label(col, text=f"load {r.load_time_s:.2f}s gen {r.gen_time_s:.2f}s\npeak {r.peak_mem_gb:.2f}GB").pack(pady=4)
            tk.Label(col, text=r.extra_info, wraplength=360, justify="left").pack(pady=4)
        else:
            tk.Label(col, text="FAILED" if not r.ok else "NO IMAGE", fg="red").pack(pady=6)
            tk.Label(col, text=r.error or "", wraplength=360, justify="left").pack()
    box = tk.Text(root, height=18, width=170)
    box.pack(fill=tk.BOTH, expand=False)
    box.insert("1.0", summarize(results))
    box.configure(state="disabled")
    root.mainloop()

# ---------------- Main ----------------
def main():
    print("=== System Info ===")
    print("Python:", sys.version)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda,
              "Compute:", torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM Free {free/1024**3:.2f}GB / Total {total/1024**3:.2f}GB")
    else:
        print("CUDA not available")
    print(f"RUN_DEFAULT={RUN_DEFAULT} RUN_FP8={RUN_FP8} RUN_QUANT={RUN_QUANT} FORCE_DTYPE={FORCE_DTYPE or 'auto'} "
          f"EARLY_OFFLOAD={ENABLE_EARLY_OFFLOAD} ALLOW_FALLBACK={ALLOW_FALLBACK}")

    results: List[RunResult] = []
    if RUN_DEFAULT:
        print("\n=== RUN: Default ===")
        results.append(run_in_subprocess("default", "Default"))
    if RUN_QUANT:
        print("\n=== RUN: Quantized + Lightning ===")
        results.append(run_in_subprocess("quant", "Quant+Lightning"))
    if RUN_FP8:
        print("\n=== RUN: FP8 Injection ===")
        results.append(run_in_subprocess("fp8", "FP8"))

    print("\n=== Summary ===")
    print(summarize(results))
    show_images(results)

if __name__ == "__main__":
    mp.freeze_support()
    main()