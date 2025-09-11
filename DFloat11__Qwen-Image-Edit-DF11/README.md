---
base_model:
- Qwen/Qwen-Image-Edit
base_model_relation: quantized
tags:
- dfloat11
- df11
- lossless compression
- 70% size, 100% accuracy
pipeline_tag: image-to-image
---

# DFloat11 Compressed Model: `Qwen/Qwen-Image-Edit`

This is a **DFloat11 losslessly compressed** version of the original `Qwen/Qwen-Image-Edit` model. It reduces model size by **32%** compared to the original BFloat16 model, while maintaining **bit-identical outputs** and supporting **efficient GPU inference**.

🔥🔥🔥 Thanks to DFloat11 compression, Qwen-Image-Edit can now run on **a single 32GB GPU**, or on **a single 24GB GPU with CPU offloading**, while maintaining full model quality. 🔥🔥🔥

### 📊 Performance Comparison

| Model                                          | Model Size | Peak GPU Memory                              | Generation Time (A100 GPU) |
|------------------------------------------------|------------|----------------------------------------------|----------------------------|
| Qwen-Image-Edit (BFloat16)                     | ~41 GB     | OOM                                          | -                          |
| Qwen-Image-Edit (DFloat11)                     | 28.43 GB   | 30.11 GB                                     | 280 seconds                |
| Qwen-Image-Edit (DFloat11 + CPU Offloading)    | 28.43 GB   | 22.71 GB                                     | 570 seconds                |

### 🔧 How to Use

1. Install or upgrade the DFloat11 pip package *(installs the CUDA kernel automatically; requires a CUDA-compatible GPU and PyTorch installed)*:

    ```bash
    pip install -U dfloat11[cuda12]
    ```

2. Install or upgrade diffusers:

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

3. Save the following code to a Python file `qwen_image_edit.py`:

    ```python
    import argparse
    import torch
    from diffusers.utils import load_image
    from diffusers import QwenImageTransformer2DModel, QwenImageEditPipeline
    from transformers.modeling_utils import no_init_weights
    from dfloat11 import DFloat11Model

    def parse_args():
        parser = argparse.ArgumentParser(description='Edit images using Qwen-Image-Edit model')
        parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading')
        parser.add_argument('--cpu_offload_blocks', type=int, default=30, help='Number of transformer blocks to offload to CPU')
        parser.add_argument('--no_pin_memory', action='store_true', help='Disable memory pinning')
        parser.add_argument('--image', type=str, default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
                            help='Path to input image or URL')
        parser.add_argument('--prompt', type=str, default='Add a hat to the cat.',
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
    ```

4. To run without CPU offloading (32GB VRAM required):
    ```bash
    python qwen_image_edit.py
    ```

    To run with CPU offloading (24GB VRAM required, 50GB CPU RAM required):
    ```bash
    python qwen_image_edit.py --cpu_offload
    ```

    If you are getting out of (CPU or GPU) memory errors, try limiting the number of offloaded blocks or disabling memory-pinning:
    ```bash
    # Offload only 12 blocks (offloading more blocks uses less GPU memory and more CPU memory; offloading less blocks is faster):
    python qwen_image_edit.py --cpu_offload --cpu_offload_blocks 12

    # Disable memory-pinning (the most memory efficient way, but could be slower):
    python qwen_image_edit.py --cpu_offload --cpu_offload_blocks 60 --no_pin_memory
    ```

### 🔍 How It Works

We apply **Huffman coding** to losslessly compress the exponent bits of BFloat16 model weights, which are highly compressible (their 8 bits carry only ~2.6 bits of actual information). To enable fast inference, we implement a highly efficient CUDA kernel that performs on-the-fly weight decompression directly on the GPU.

The result is a model that is **~32% smaller**, delivers **bit-identical outputs**, and achieves performance **comparable to the original** BFloat16 model.

Learn more in our [research paper](https://arxiv.org/abs/2504.11651).

### 📄 Learn More

* **Paper**: [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float](https://arxiv.org/abs/2504.11651)
* **GitHub**: [https://github.com/LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)
* **HuggingFace**: [https://huggingface.co/DFloat11](https://huggingface.co/DFloat11)
