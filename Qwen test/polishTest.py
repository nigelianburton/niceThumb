import os
import time
import torch
import datetime
import numpy as np
import gradio as gr
from PIL import Image
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
import math

# ========================================================
# CONFIG FLAGS (Default: existing behavior)
# ========================================================
# If True, use a locally preloaded Qwen Image Edit model directory instead of pulling from HuggingFace.
# Directory must contain the expected subfolders: transformer/, text_encoder/, etc.
USE_PRELOADED_QWEN_EDIT = True
# Allow environment override for model directory (highest priority)
PRELOADED_QWEN_EDIT_PATH = (
    os.environ.get("QWEN_IMAGE_EDIT_DIR")
    or os.environ.get("QWEN_MODEL_LOCAL_DIR")
    or r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit"
)

# Precision flags (default False = original 4-bit quantized behavior)
USE_HIGH_PRECISION_VISION = False    # If True: vision transformer loads full precision (no 4-bit)
USE_HIGH_PRECISION_TEXT   = True     # If True: text encoder loads full precision (no 4-bit)

# Default starting image path (if present)
DEFAULT_START_IMAGE = r"C:\_CONDA\niceThumb\lara.jpg"

# ========================================================
# LoRA / LIGHTNING CONFIG (enhanced local overrides)
# ========================================================
# 1) Lightning LoRA directory (env override)
LIGHTNING_LORA_DIR = os.environ.get("QWEN_LIGHTNING_DIR", r"C:\_CONDA\niceThumb\Qwen-Image-Lightning")
# 2) Lightning LoRA weight filename (must exist in directory if local). If not found, fallback to HF repo call.
LIGHTNING_LORA_FILENAME = os.environ.get("QWEN_LIGHTNING_LORA", "Qwen-Image-Lightning-8steps-V1.1.safetensors")
# 3) Additional LoRA base directory (env override)
ADDITIONAL_LORA_DIR = os.environ.get("QWEN_EXTRA_LORA_DIR", r"C:\_MODELS-SD\Qwen\Qwen-Lora")
# 4) Hard-coded list of additional LoRA filenames (empty by default). Relative to ADDITIONAL_LORA_DIR.
ADDITIONAL_LORAS = []  # e.g. ["style1.safetensors", "lighting_boost.safetensors"]
# Optional env to append comma-separated additional loras (filenames)
_env_extra = os.environ.get("QWEN_EXTRA_LORAS", "")
if _env_extra.strip():
    for name in [x.strip() for x in _env_extra.split(",") if x.strip()]:
        if name not in ADDITIONAL_LORAS:
            ADDITIONAL_LORAS.append(name)

# ========================================================
# GPU / TIMING UTILITIES
# ========================================================

def _gpu_mem():
    if not torch.cuda.is_available():
        return "CUDA not available"
    device = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(device)
    used_b = total_b - free_b
    def fmt(x):  # show GB with 2 decimals
        return f"{x/1024**3:.2f}GB"
    return f"used {fmt(used_b)} / total {fmt(total_b)} (free {fmt(free_b)})"

def _log_step(label, start_time):
    elapsed = time.time() - start_time
    print(f"⏱ {label} | elapsed {elapsed:.2f}s | GPU: {_gpu_mem()}")

# ========================================================
# MODEL LOADING FUNCTIONS
# ========================================================

def load_model():
    start_time = time.time()
    print("🔄 Loading model... (approx 2m30s first time)")
    print(f"🧠 GPU VRAM at start: {_gpu_mem()}")

    # Choose model source (local preloaded vs HuggingFace hub)
    if USE_PRELOADED_QWEN_EDIT:
        if os.path.isdir(PRELOADED_QWEN_EDIT_PATH):
            model_id = PRELOADED_QWEN_EDIT_PATH
            print(f"📂 Using preloaded model directory: {model_id}")
        else:
            print(f"⚠️ Preloaded model path not found: {PRELOADED_QWEN_EDIT_PATH}. Falling back to HuggingFace hub.")
            model_id = "Qwen/Qwen-Image-Edit"
    else:
        model_id = "Qwen/Qwen-Image-Edit"

    torch_dtype = torch.bfloat16

    # 1. Visual Transformer
    if USE_HIGH_PRECISION_VISION:
        print("1/6 - Loading visual transformer (HIGH precision, bf16)...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        ).to("cpu")
    else:
        print("1/6 - Loading visual transformer (4-bit NF4)...")
        quantization_config_diffusers = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config_diffusers,
            torch_dtype=torch_dtype,
        ).to("cpu")
    _log_step("Loaded visual transformer", start_time)

    # 2. Text Encoder
    if USE_HIGH_PRECISION_TEXT:
        print("2/6 - Loading text encoder (HIGH precision, bf16)...")
        quantization_config_transformers = None
    else:
        print("2/6 - Loading text encoder (4-bit NF4)...")
        quantization_config_transformers = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    text_encoder_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    )
    if quantization_config_transformers is not None:
        text_encoder_kwargs["quantization_config"] = quantization_config_transformers
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(**text_encoder_kwargs).to("cpu")
    _log_step("Loaded text encoder", start_time)

    # 3. Pipeline
    print("3/6 - Building pipeline...")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )
    _log_step("Pipeline constructed", start_time)

    # 4. LoRA (Lightning + additional)
    print("4/6 - Loading LoRA weights...")
    try:
        lightning_loaded = False
        local_lightning_path = os.path.join(LIGHTNING_LORA_DIR, LIGHTNING_LORA_FILENAME)
        if os.path.isfile(local_lightning_path):
            print(f"🔌 Loading local Lightning LoRA: {local_lightning_path}")
            pipe.load_lora_weights(local_lightning_path)
            lightning_loaded = True
        else:
            print(f"ℹ️ Local Lightning LoRA not found ({local_lightning_path}), trying HF repo.")
            try:
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Lightning",
                    weight_name=LIGHTNING_LORA_FILENAME
                )
                lightning_loaded = True
            except Exception as e2:
                print(f"⚠️ HF Lightning LoRA load failed: {e2}")

        if lightning_loaded:
            print("✅ Lightning LoRA loaded.")

        # Additional LoRAs
        if ADDITIONAL_LORAS:
            print("➕ Loading additional LoRAs...")
            loaded_names = []
            for l_name in ADDITIONAL_LORAS:
                full_path = os.path.join(ADDITIONAL_LORA_DIR, l_name)
                if os.path.isfile(full_path):
                    adapter_name = os.path.splitext(os.path.basename(l_name))[0]
                    try:
                        pipe.load_lora_weights(full_path, adapter_name=adapter_name)
                        loaded_names.append(adapter_name)
                        print(f"  • Loaded additional LoRA: {full_path} (adapter: {adapter_name})")
                    except Exception as ex_l:
                        print(f"  ⚠️ Failed additional LoRA {full_path}: {ex_l}")
                else:
                    print(f"  ⚠️ Additional LoRA file missing: {full_path}")
            # If adapter management is supported, we could activate them (equal weights)
            if loaded_names:
                try:
                    if hasattr(pipe, "set_adapters"):
                        adapters_weights = [1.0] * len(loaded_names)
                        pipe.set_adapters(loaded_names, adapters_weights)
                        print(f"✅ Activated adapters: {loaded_names}")
                except Exception as ex_set:
                    print(f"⚠️ Failed to activate adapters collectively: {ex_set}")

        _log_step("LoRA loading stage complete", start_time)
    except Exception as e:
        print(f"⚠️ LoRA load phase error: {e}")
        _log_step("LoRA load failed (continuing)", start_time)

    # 5. CPU offload
    print("5/6 - Enabling CPU offload...")
    pipe.enable_model_cpu_offload()
    _log_step("CPU offload enabled", start_time)

    # 6. Finalization
    print("6/6 - Finalizing inference setup...")
    _log_step("Inference setup finalized", start_time)
    print("=" * 60)

    elapsed = time.time() - start_time
    print(f"✅ Model ready in {elapsed:.2f}s | Final GPU: {_gpu_mem()}")
    return pipe

# ========================================================
# GLOBAL PIPELINE
# ========================================================

try:
    pipe = load_model()
except Exception as e:
    print(f"❌ Model failed to load: {e}")
    raise

# ========================================================
# IMAGE SIZE CHECK AND RESIZING FUNCTION
# ========================================================

TARGET_AREA = 1024 * 1024  # 1,048,576

def resize_image_if_needed(image):
    width, height = image.size
    orig_area = width * height
    if width % 4 == 0 and height % 4 == 0 and orig_area == TARGET_AREA:
        print("✅ Image already matches target area and 4x alignment.")
        return image

    aspect = width / height
    scale = math.sqrt(TARGET_AREA / orig_area)
    ideal_w = width * scale
    def mult4(x): return max(4, int(round(x / 4.0)) * 4)

    w_floor = max(4, (int(ideal_w) // 4) * 4)
    w_ceil = w_floor + 4
    w_candidates = {w_floor, w_ceil, mult4(ideal_w)}

    candidates = []
    for w_c in w_candidates:
        if w_c <= 0: continue
        ideal_h = w_c / aspect
        h_floor = max(4, (int(ideal_h) // 4) * 4)
        h_ceil = h_floor + 4
        for h_c in {h_floor, h_ceil, mult4(ideal_h)}:
            if h_c <= 0: continue
            area = w_c * h_c
            candidates.append((abs(area - TARGET_AREA), abs((w_c / h_c) - aspect), area, w_c, h_c))

    candidates.sort()
    _, _, final_area, new_w, new_h = candidates[0]
    print(f"📏 Resizing: {width}x{height} ({orig_area}) -> {new_w}x{new_h} ({final_area}) target {TARGET_AREA}")
    if new_w == width and new_h == height:
        print("✅ Computed size equals original; reuse image.")
        return image
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

# ========================================================
# INTERNAL: STEP LOGGING PATCH
# ========================================================

def _patch_scheduler_for_logging(local_pipe, steps, gen_start_time):
    if not hasattr(local_pipe, "scheduler") or not hasattr(local_pipe.scheduler, "step"):
        return lambda: None
    sched = local_pipe.scheduler
    original_step = sched.step
    counter = {"i": 0}

    def logged_step(model_output, timestep, *args, **kwargs):
        counter["i"] += 1
        elapsed = time.time() - gen_start_time
        print(f"🧩 Step {counter['i']}/{steps} | timestep {timestep} | {elapsed:.2f}s | GPU: {_gpu_mem()}")
        return original_step(model_output, timestep, *args, **kwargs)

    sched.step = logged_step
    def restore(): sched.step = original_step
    return restore

# ========================================================
# GRADIO IMAGE EDIT FUNCTION
# ========================================================

def edit_image(input_image, prompt):
    steps = 8
    if input_image is None:
        return None, "Please upload an image."
    if not prompt.strip():
        return None, "Please enter a prompt."
    try:
        processed_image = resize_image_if_needed(input_image)
        gen_start = time.time()
        print(f"🚀 Generation start | GPU: {_gpu_mem()}")
        restore = _patch_scheduler_for_logging(pipe, steps, gen_start)
        try:
            result = pipe(
                image=processed_image,
                prompt=prompt,
                num_inference_steps=steps
            ).images[0]
        finally:
            restore()
        total_elapsed = time.time() - gen_start
        print(f"✅ Image successfully edited in {total_elapsed:.2f}s | Final GPU: {_gpu_mem()}")
        save_status = save_image_locally(result)
        print(save_status)
        return result, f"✅ Successfully edited! ({total_elapsed:.2f} s)\n{save_status}"
    except Exception as e:
        return None, f"❌ An error occurred: {str(e)}"

def save_image_locally(image_input):
    if image_input is None:
        return "❌ No image to save!"
    try:
        if isinstance(image_input, np.ndarray):
            image_pil = Image.fromarray(np.clip(image_input, 0, 255).astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            image_pil = image_input
        else:
            return "❌ Unsupported image format."
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"qwie_{timestamp}.png")
        image_pil.save(output_path)
        return f"✅ Saved: {os.path.abspath(output_path)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ========================================================
# PRE-LOAD DEFAULT IMAGE
# ========================================================

def _load_default_image():
    if os.path.isfile(DEFAULT_START_IMAGE):
        try:
            img = Image.open(DEFAULT_START_IMAGE).convert("RGB")
            print(f"🖼 Loaded default start image: {DEFAULT_START_IMAGE}")
            return img
        except Exception as e:
            print(f"⚠️ Failed to load default image '{DEFAULT_START_IMAGE}': {e}")
    else:
        print(f"ℹ️ Default start image not found: {DEFAULT_START_IMAGE}")
    return None

_default_img = _load_default_image()

# ========================================================
# GRADIO UI
# ========================================================

with gr.Blocks(title="🎨 Qwen-Image Edit - Local App") as demo:
    gr.Markdown("""
    # 🎨 Image Editing with Qwen-Image Edit
    Edit your images with text! (Example: 'Make the sweater have red stripes')
    """)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                elem_id="input_img",
                value=_default_img
            )
            prompt = gr.Textbox(
                label="Prompt (Description)",
                placeholder="Make the sweater have red stripes",
                value="change the color of clothes to pink"
            )
            btn = gr.Button("🎨 Apply Edit", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Output Image", elem_id="output_img")
            status = gr.Textbox(label="Status", value="Ready")
    btn.click(
        fn=edit_image,
        inputs=[input_image, prompt],
        outputs=[output_image, status]
    )
    gr.Markdown("""
    <br>
    <small>
    Model: <a href="https://huggingface.co/Qwen/Qwen-Image-Edit" target="_blank">Qwen/Qwen-Image-Edit</a> |
    LoRA: <a href="https://huggingface.co/lightx2v/Qwen-Image-Lightning" target="_blank">lightx2v/Qwen-Image-Lightning</a><br>
    Note: First run takes longer (model loading). Subsequent runs are faster. <br>
    On average an edit takes about 1 minute 30 seconds.
    </small>
    """)

# ========================================================
# ENVIRONMENT SANITIZATION
# ========================================================

def _sanitize_loopback_env():
    loopback_hosts = ["localhost", "127.0.0.1", "::1"]
    keys = ["NO_PROXY", "no_proxy"]
    existing = []
    for k in keys:
        val = os.environ.get(k)
        if val:
            existing.extend([p.strip() for p in val.split(",") if p.strip()])
    merged = set(existing) | set(loopback_hosts)
    for k in keys:
        os.environ[k] = ",".join(sorted(merged))
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
        os.environ.pop(k, None)

# ========================================================
# START APPLICATION
# ========================================================

if __name__ == "__main__":
    _sanitize_loopback_env()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False
    )