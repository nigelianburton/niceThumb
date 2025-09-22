# -*- coding: utf-8 -*-
"""
Centralized constants for file locations and configuration in the qtd diffusion system.

This module contains all file path constants and environment variable configurations
used across the SDXL and Qwen backends, as well as server configuration.
"""

import os
import sys

# ==========================================================
# Model Directory Discovery
# ==========================================================

def _find_models_sd_base():
    """
    Find the _MODELS-SD base directory by checking C: first, then D:.
    Returns the base path if found, otherwise exits with error.
    """
    # Check C: drive first
    c_path = r"C:\_MODELS-SD"
    if os.path.isdir(c_path):
        return c_path
    
    # Check D: drive as fallback
    d_path = r"D:\_MODELS-SD"
    if os.path.isdir(d_path):
        return d_path
    
    # Neither found - exit with error
    print(f"ERROR: _MODELS-SD directory not found on either C: ({c_path}) or D: ({d_path})")
    print("Please ensure the models directory exists in one of these locations:")
    print(f"  - {c_path}")
    print(f"  - {d_path}")
    sys.exit(1)

# Discover the models base directory
_MODELS_SD_BASE = _find_models_sd_base()

# ==========================================================
# SDXL Backend Constants
# ==========================================================

# SDXL model file location
SDXL_MODEL_PATH = os.environ.get("NT6_SDXL_MODEL", os.path.join(_MODELS_SD_BASE, "StableDiffusion", "juggernautXL_ragnarokBy.safetensors"))

# SDXL models directory
SDXL_MODELS_DIR = os.environ.get("NT6_SDXL_DIR", os.path.join(_MODELS_SD_BASE, "StableDiffusion"))

# SDXL LoRA directory
SDXL_LORAS_DIR = os.environ.get("NT6_LORAS_DIR", os.path.join(_MODELS_SD_BASE, "Lora"))

# SDXL default negative prompt
SDXL_NEG_PROMPT = os.environ.get("NT6_NEG_PROMPT", "blurry, lowres, deformed, extra limbs, bad anatomy, watermark, text")

# SDXL default inference steps
SDXL_DEFAULT_STEPS = int(os.environ.get("NT6_STEPS", "30"))

# SDXL default CFG scale
SDXL_DEFAULT_CFG = float(os.environ.get("NT6_CFG", "7.5"))

# ==========================================================
# Qwen Backend Constants
# ==========================================================

# Use local preloaded model directory (else fallback to HF hub)
QWEN_USE_PRELOADED = True

# Qwen model directory with fallback chain
QWEN_PRELOADED_PATH = (
    os.environ.get("QWEN_IMAGE_EDIT_DIR")
    or os.environ.get("QWEN_MODEL_LOCAL_DIR")
    or os.path.join(_MODELS_SD_BASE, "Qwen", "Qwen-Image-Edit")
)

# Precision flags (match polishTest defaults)
# False => 4-bit quantization for that component; True => full bf16 precision
QWEN_USE_HIGH_PRECISION_VISION = False
QWEN_USE_HIGH_PRECISION_TEXT = True

# Lightning LoRA configuration
QWEN_LIGHTNING_LORA_DIR = os.environ.get("QWEN_LIGHTNING_DIR", r"C:\_CONDA\niceThumb\Qwen-Image-Lightning")
QWEN_LIGHTNING_LORA_FILENAME = os.environ.get("QWEN_LIGHTNING_LORA", "Qwen-Image-Lightning-8steps-V1.1.safetensors")

# Additional LoRA directory & list (multi-adapter)
QWEN_ADDITIONAL_LORA_DIR = os.environ.get("QWEN_EXTRA_LORA_DIR", os.path.join(_MODELS_SD_BASE, "Qwen", "Qwen-Lora"))

# Parse additional LoRAs from environment
QWEN_ADDITIONAL_LORAS = []
_env_extra = os.environ.get("QWEN_EXTRA_LORAS", "")
if _env_extra.strip():
    for _n in [x.strip() for x in _env_extra.split(",") if x.strip()]:
        if _n not in QWEN_ADDITIONAL_LORAS:
            QWEN_ADDITIONAL_LORAS.append(_n)

# Verbose logging (GPU/time/step) toggle
QWEN_VERBOSE = True

# Target normalization area (matches polishTest)
QWEN_TARGET_AREA = 1024 * 1024  # 1,048,576

# Default inference steps (if client omits)
QWEN_DEFAULT_STEPS = 8

# HF fallback model id
QWEN_HF_MODEL_ID = "Qwen/Qwen-Image-Edit"

# ==========================================================
# Server Constants
# ==========================================================

# Server host
QTD_SERVER_HOST = os.environ.get("QTD_HOST", "127.0.0.1")

# Server port
QTD_SERVER_PORT = int(os.environ.get("QTD_PORT", "5015"))

# HTTP logging controls
QTD_QUIET_HTTP_LOG = False
QTD_QUIET_PROGRESS_ONLY = True