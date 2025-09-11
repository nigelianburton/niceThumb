import argparse
import torch
import os
from diffusers.utils import load_image
from diffusers import QwenImageTransformer2DModel, QwenImageEditPipeline
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model
from pathlib import Path

app_dir = Path(__file__).resolve().parent
src_path = "lara.jpg"
def_prompt = "Replace the girl's outfit with a green dress, high quality portrait, cinematic lighting"

def parse_args():
    parser = argparse.ArgumentParser(description='Edit images using Qwen-Image-Edit model')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--cpu_offload_blocks', type=int, default=30, help='Number of transformer blocks to offload to CPU')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable memory pinning')
    parser.add_argument('--image', type=str, default=src_path,
                        help='Path to input image or URL')
    parser.add_argument('--prompt', type=str, default=def_prompt,
                        help='Text prompt for image editing')
    parser.add_argument('--negative_prompt', type=str, default=' ',
                        help='Negative prompt for image editing')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--true_cfg_scale', type=float, default=4.0,
                        help='Classifier free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for generation')
    parser.add_argument('--output', type=str, default='qwen_image_edit.png',
                        help='Output image path')
    return parser.parse_args()

args = parse_args()
model_id = "Qwen/Qwen-Image-Edit"

with no_init_weights():
    transformer = QwenImageTransformer2DModel.from_config(
        QwenImageTransformer2DModel.load_config(
            model_id, subfolder="transformer",
        ),
    ).to(torch.bfloat16)

DFloat11Model.from_pretrained(
    "DFloat11/Qwen-Image-Edit-DF11",
    device="cpu",
    cpu_offload=args.cpu_offload,
    cpu_offload_blocks=args.cpu_offload_blocks,
    pin_memory=not args.no_pin_memory,
    bfloat16_model=transformer,
)

pipeline = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
pipeline.set_progress_bar_config(disable=None)

image = load_image(args.image)
inputs = {
    "image": image,
    "prompt": args.prompt,
    "generator": torch.manual_seed(args.seed),
    "true_cfg_scale": args.true_cfg_scale,
    "negative_prompt": args.negative_prompt,
    "num_inference_steps": args.num_inference_steps,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(args.output)

max_gpu_memory = torch.cuda.max_memory_allocated()
print(f"Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
