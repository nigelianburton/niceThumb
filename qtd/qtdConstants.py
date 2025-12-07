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
    Find the _MODELS_SD base directory by checking C:, D:, then T:.
    Returns the base path if found, otherwise exits with error.
    """
    # Check C: drive first
    c_path = r"C:\_MODELS_SD"
    if os.path.isdir(c_path):
        return c_path
    
    # Check D: drive as fallback
    d_path = r"D:\_MODELS_SD"
    if os.path.isdir(d_path):
        return d_path
    
    # Check T: drive as final fallback
    t_path = r"T:\_MODELS_SD"
    if os.path.isdir(t_path):
        return t_path
    
    # None found - exit with red error
    print(f"\033[91mERROR: _MODELS_SD directory not found on any drive (C:, D:, T:)\033[0m")
    print("Please ensure the models directory exists in one of these locations:")
    print(f"  - {c_path}")
    print(f"  - {d_path}")
    print(f"  - {t_path}")
    sys.exit(1)

# Discover the models base directory
_MODELS_SD = _find_models_sd_base()

def _check_and_warn_folder(folder_path, folder_name):
    """
    Check if folder exists and is not empty. Show red error if missing or empty.
    Returns True if folder exists and has content, False otherwise.
    """
    if not os.path.isdir(folder_path):
        print(f"\033[91mERROR: {folder_name} folder not found: {folder_path}\033[0m")
        return False
    
    try:
        contents = os.listdir(folder_path)
        if not contents:
            print(f"\033[91mERROR: {folder_name} folder is empty: {folder_path}\033[0m")
            return False
    except Exception:
        print(f"\033[91mERROR: Cannot read {folder_name} folder: {folder_path}\033[0m")
        return False
    
    return True

# ==========================================================
# SDXL Backend Constants
# ==========================================================

# SDXL base directory
SDXL_DIR = _MODELS_SD

# SDXL models directory
SDXL_MODELS = os.path.join(_MODELS_SD, "StableDiffusion")
_check_and_warn_folder(SDXL_MODELS, "SDXL StableDiffusion")

# SDXL LoRA directory  
SDXL_LORA = os.path.join(_MODELS_SD, "Lora")
_check_and_warn_folder(SDXL_LORA, "SDXL Lora")

# SDXL model file location (fallback for legacy compatibility)
SDXL_MODEL_PATH = os.environ.get("NT6_SDXL_MODEL", os.path.join(SDXL_MODELS, "juggernautXL_ragnarokBy.safetensors"))

# Legacy constants for backward compatibility
SDXL_MODELS_DIR = SDXL_MODELS  # replacing NT6_SDXL_DIR
SDXL_LORAS_DIR = SDXL_LORA     # replacing NT6_LORAS_DIR

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

# Qwen models directory
QWEN_MODELS = os.path.join(_MODELS_SD, "QwenModels")
_check_and_warn_folder(QWEN_MODELS, "Qwen Models")

# Qwen model directory with fallback chain
QWEN_PRELOADED_PATH = (
    os.environ.get("QWEN_IMAGE_EDIT_DIR")
    or os.environ.get("QWEN_MODEL_LOCAL_DIR")
    or os.path.join(QWEN_MODELS, "Qwen-Image-Edit")
)

# Precision flags (match polishTest defaults)
# False => 4-bit quantization for that component; True => full bf16 precision
QWEN_USE_HIGH_PRECISION_VISION = False
QWEN_USE_HIGH_PRECISION_TEXT = True

# Qwen Speedup LoRA directory
QWEN_SPEEDUP_LORAS = os.path.join(_MODELS_SD, "QwenLightning")
_check_and_warn_folder(QWEN_SPEEDUP_LORAS, "Qwen Speedup LoRAs")

# Lightning LoRA configuration (legacy compatibility)
QWEN_LIGHTNING_LORA_DIR = os.environ.get("QWEN_LIGHTNING_DIR", QWEN_SPEEDUP_LORAS)
QWEN_LIGHTNING_LORA_FILENAME = os.environ.get("QWEN_LIGHTNING_LORA", "Qwen-Image-Lightning-8steps-V1.1.safetensors")

# Qwen additional LoRA directory
QWEN_LORA = os.path.join(_MODELS_SD, "QwenLora")
_check_and_warn_folder(QWEN_LORA, "Qwen Lora")

# Additional LoRA directory & list (multi-adapter)
QWEN_ADDITIONAL_LORA_DIR = os.environ.get("QWEN_EXTRA_LORA_DIR", QWEN_LORA)

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