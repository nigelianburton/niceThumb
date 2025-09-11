# Project Architecture & Guidance for AI Assistance

## High-Level Overview
NiceThumb is a PyQt6 desktop application for:
- Browsing file system assets (images, videos)
- Previewing images & videos
- Editing images with modular paint/diffusion tools
- Invoking Stable Diffusion (text→image, image→image/edit) via a local HTTP backend
- Optional MCP (Model Context Protocol) tool integration for scripted diffusion workflows

A sibling MCP server (qtMcp.py) exposes higher-level batch/story operations that forward to the diffusion HTTP API.

## Primary Entry Points
| Concern | File | Role |
|---------|------|------|
| Application bootstrap | qtMain.py | Launches UI, ensures diffusion service running, mounts browser + editors |
| Browser/grid | qtBrowse.py | Thumbnail grid, selection logic, async thumbnail generation |
| Image editing & diffusion tools | qtPaintMain.py | PaintView: modular tool orchestration (paint/mask/blur/clone/select/diffuse) |
| Diffusion HTTP server | qtd/qtdServer.py | Flask API: job submission, progress, events, (extendable endpoints) |
| Diffusion backends | qtd/qtdSDXL.py / qtd/qtdQEdit1.py | Separate pipeline loaders (SDXL vs Qwen Image Edit) |
| Shared diffusion helpers | qtd/qtdHelpers.py | Image decoding, resizing, memory utilities, LoRA resolution, progress patching |
| Paint tool plugins | qt_paint_tools/* | Individual tool implementations (blur, selection, mask, diffuse, etc.) |
| MCP integration | qtMcp.py | FastMCP tools: generate_storybook, switch_character |
| LLM request preview (UI) | qtPreviewLLM.py | Polls /llm_requests (endpoint must be added to server) |

## UI Layer (PyQt6)
- Main window (qtMain.py):
  - Left: BrowserView (from qtBrowse.py)
  - Right stack: PreviewView (image/video) and PaintView (editing/diffusion)
  - Dynamically adds PreviewLLMView when LLM mode activated
- Browsing:
  - Thumbnails generated lazily (async QRunnable) for videos; direct load for images
  - Selection supports Ctrl (toggle), Shift (range), multi-file state
- Editing (PaintView):
  - Tool activation pattern: on_selected(**params), optional on_deselected()
  - Cursor + brush preview updated per tool
  - Diffusion tool relies on composition provider + mask provider callbacks

## Diffusion Subsystem
### Server (qtd/qtdServer.py)
- Endpoints:
  - /ping, /models, /loras
  - /jobs/submit (POST job), /progress/<id>, /events/<id> (SSE)
  - (Extensible) Suggested: /llm_requests for qtPreviewLLM.py
- Job lifecycle:
  - Creates JOBS[job_id] dict: {status, percent, status_text, result?, error?}
  - Backends receive: set_progress, set_result, set_error, set_status
  - Percent mapping: 0..100 (backend maps internal phases → 1..10 → step loop 10..99 → 100)
- Status text:
  - Now enriched via set_status for both SDXL and Qwen backends
  - Do NOT overload with per-step spam (throttle ~ every N steps)

### Backends
#### SDXLBackend (qtdSDXL.py)
- Two pipelines: text (t2i) and img2img (i2i) variant
- Uses diffusers StableDiffusionXLPipeline / Img2Img pipeline
- Progress:
  - Uses new diffusers callback_on_step_end API exclusively (legacy callback removed)
- LoRA:
  - Applied via load_lora_weights() + set_adapters()
  - Re-applied per job based on requested list

#### QwenEdit1Backend (qtdQEdit1.py)
- Single QwenImageEditPipeline repurposed for:
  - Edit (i2i) with optional init image
  - t2i simulated by generating synthetic blurred base image (ensures pipeline still uses edit path)
- Mixed precision / quantization toggles:
  - Vision: optional 4-bit NF4
  - Text encoder: optional 4-bit NF4 or bf16
- Progress:
  - Scheduler.step monkey-patch (not using callback_on_step_end)
  - patch_scheduler_progress kept shared in qtdHelpers but Qwen currently uses local wrapper to inject status text
- LoRA:
  - Lightning LoRA (one) + multiple additional adapters (multi-activation)
- Deterministic resizing to ~1M pixels (area 1024x1024 target) with aspect preserved (dims %4)

### Shared Helpers (qtdHelpers.py)
Include:
- decode_data_url_to_pil / pil_to_data_url
- seed_from_inputs
- resize_to_area_preserve_aspect
- ensure_rgb
- patch_scheduler_progress (generic scheduler .step wrapper)
- LoRA path resolution + safe adapter naming
- empty_cuda_cache, bf16_supported

When modifying helpers:
- Maintain backward compatibility (return shape, side effects)
- Avoid calling GPU APIs unless torch.cuda.is_available()

## MCP (qtMcp.py)
- FastMCP server defining two tools:
  - generate_storybook(pages[1..5])
  - switch_character(tokens, reference_image_path?, weight, seed?)
- Forwards JSON to diffusion HTTP endpoints (distinct from standard /jobs flow)
- Does NOT interact with qtPreviewLLM.py directly

## LLM Preview (qtPreviewLLM.py)
- Polls /llm_requests every 2s (endpoint currently absent — must be added if used)
- Standalone runnable (--api-base, --interval)
- Safe to import dynamically in qtMain

## Coding & Contribution Guidelines
1. Avoid breaking sibling backend:
   - When adding progress/status features, apply minimal, parallel changes to both SDXL and Qwen.
   - Keep signatures backward compatible; add optional params at end with defaults.
2. Never remove internal helper methods (_load_libs, _load_transformer, etc.) without ensuring callers updated.
3. Progress mapping:
   - Reserve 0–10 for initialization, 10–99 for iterative steps, 100 for terminal state.
4. Status text:
   - Use concise verbs: "Initializing", "Loading vision transformer", "Denoising 4/8", "Finalizing"
5. Thread safety:
   - All pipeline inits are guarded by _init_lock.
   - Do not perform UI operations from diffusion threads.
6. Memory:
   - Prefer enabling model CPU offload if available (try/except).
   - After unload(), call empty_cuda_cache().
7. Synthetic t2i (Qwen):
   - Keep base image deterministic unless explicitly extended (reproducibility).
8. LoRA:
   - When applying multiple adapters: set_adapters(list, weights) with uniform weights unless user-specified.
   - Always unload/reset previous adapters before applying new list for a job.
9. Endpoint extension pattern:
   - Add global container + max size (e.g., LLM_REQUESTS) in qtdServer.py
   - Provide GET (list) + POST (append) for simple feeds.
10. Reliability:
   - Wrap all external (disk, network, torch) calls in try/except; emit clear error via set_error().
   - Avoid raising uncaught exceptions inside backend.submit or ensure_pipeline.

## Typical Flow (Text to Image – SDXL)
1. Client POST /jobs/submit {modelId, operation:"t2i", inputs}
2. Server optionally ensure_pipeline()
3. Backend:
   - Progress: 1 (starting) → staged (2..10) → steps (10..99) → 100
   - Status: "Loading model" → "LoRAs applied" → "Generating" → step updates → "Finalizing"
4. Result: base64 data URL returned in /progress poll or SSE /events

## Typical Flow (Edit – Qwen)
1. Client POST /jobs/submit {modelId:"qwen1:image-edit", operation:"edit", init_image}
2. ensure_pipeline (lazy) if needed
3. Image decoded + resized + RGB-normalized
4. Scheduler patched for progress mapping
5. Call pipeline(image=..., prompt=...)
6. Report final image

## Adding a New Diffusion Backend
- Create new file qtd/qtd<New>.py subclassing Backend
- Implement describe_models(), submit(), ensure_pipeline()
- Follow progress conventions
- Register in BACKENDS dict + rebuild_model_index()
- Reuse qtdHelpers functions for shared concerns (seed, resize, progress if possible)

## Style / Conventions
- Prefer explicit imports over wildcard
- Type annotate public methods
- Keep print diagnostics prefixed: [backend][qwen1] / [backend][sdxl] / [qtd][job]
- Avoid adding blocking sleeps inside Flask request context threads (keep diffusion in worker thread)

## What NOT To Do
- Do not refactor both backends simultaneously without incremental verification.
- Do not hardcode GPU device indices; rely on torch.cuda.current_device().
- Do not introduce global state coupled across backends (keep each isolated).
- Do not emit per-step status text more than ~8–10 times per job (throttle).

## Quick Reference
- Launch app: python qtMain.py
- Run MCP server: python qtMcp.py
- Standalone LLM panel: python qtPreviewLLM.py --api-base http://127.0.0.1:5015
- Add fake LLM request (after adding endpoint):
  curl -X POST -H "Content-Type: application/json" -d "{\"description\":\"Test\"}" http://127.0.0.1:5015/llm_requests

## Extension Checklist (Before Committing)
- [ ] Backends load after cold start (both SDXL + Qwen)
- [ ] /models lists expected models
- [ ] /jobs/submit returns progress and final image
- [ ] No AttributeError for helper methods (_load_libs etc.)
- [ ] Status text progresses through meaningful phases
- [ ] No deprecated diffusers warnings (SDXL uses callback_on_step_end)
- [ ] Qwen scheduler patch restored on exit (restore_fn executed)
- [ ] Memory not leaking across multiple job runs (GPU usage stabilizes)
- [ ] Optional: /llm_requests added if qtPreviewLLM is used

Use this document to preserve architecture integrity and avoid regressions when proposing automated changes.