# DaSiWa LLM / VLM Nodes

Run local Transformers, llama.cpp GGUF, or Ollama text models inside a ComfyUI workflow.

## Nodes

### DaSiWa LLM Model Selector

Creates a lightweight `DASIWA_LLM_CONFIG` bundle. It does not output a live model object, which helps the analyze node unload memory reliably after generation.

Place full Hugging Face-style model folders in:

```text
ComfyUI/models/llm/
```

For the `transformers` backend, the folder must include `config.json`, tokenizer files, processor files for vision models, and `.safetensors` weights. A single `.safetensors` file is not enough for LLM chat inference.

For the `llama_cpp` backend, select a local `.gguf` file. This requires a CUDA-enabled `llama-cpp-python` installation in the ComfyUI Python environment; it is intentionally optional so a generic dependency install does not replace a GPU build with a CPU-only wheel. Install it with `CMAKE_ARGS="-DGGML_CUDA=on" /path/to/ComfyUI/venv/bin/pip install --force-reinstall --no-cache-dir llama-cpp-python`. The `ollama` backend instead sends requests to the configured Ollama server and uses `ollama_model` as its model name. Loopback URLs work by default. Set `allow_remote_ollama` only when intentionally sending prompts to a trusted non-loopback server.

You can also enter a Hugging Face repo id in `hf_repo_id`, such as:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

When `download_if_missing` is enabled, the node downloads the repo snapshot into `ComfyUI/models/llm` and reuses that local folder on later runs. The local folder name uses `owner--repo`, plus the revision when it is not `main`.

For gated or private repos, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the ComfyUI environment. The node does not expose a token widget, so secrets are not stored in workflow JSON.

Important controls:

- `task`: use `auto` for most workflows. Connect images to use a vision-language model.
- `hf_repo_id`: optional Hugging Face repo id. Overrides the model dropdown when set.
- `hf_revision`: branch, tag, or commit, defaulting to `main`.
- `download_if_missing`: fetch the repo into `models/llm` when it is not already available locally.
- `device`: `auto`, `cuda`, or `cpu`.
- `dtype`: `auto`, `float16`, `bfloat16`, or `float32`.
- `quantization`: optional Transformers model-weight `8bit` or `4bit`, requiring `bitsandbytes`.
- `kv_cache_implementation`: Transformers generation cache strategy. `quantized` reduces long-generation VRAM use; use only with a compatible Transformers cache backend.
- `kv_cache_quant_backend`, `kv_cache_nbits`, and `kv_cache_residual_length`: controls used only for a quantized Transformers KV cache.
- `cache_mode`: `cached` keeps the DaSiWa backend loaded. `unload_after_run` unloads DaSiWa models, requests ComfyUI to unload its managed models, garbage-collects Python objects, and clears the device allocator after each response. For Ollama it additionally sends `keep_alive: 0` so the separate Ollama server releases its model.
- `trust_remote_code`: enable only for models that require trusted custom model code.
- `llama_n_ctx`, `llama_n_gpu_layers`, `llama_n_threads`, and `llama_chat_format`: llama.cpp GGUF controls. `-1` GPU layers requests full offload; `0` uses CPU only.
- `ollama_model`, `ollama_url`, and `ollama_timeout`: Ollama API controls.
- `allow_remote_ollama`: permits prompt requests to a non-loopback Ollama host. It is off by default; URLs containing credentials, query strings, or fragments are rejected.

### DaSiWa LLM Analyze

Runs the selected model and returns:

- `response`: generated `STRING`
- `info`: model path, cache mode, image count, and resize setting

Inputs:

- `llm_config`: from DaSiWa LLM Model Selector
- `system_prompt_preset`: preset instruction selector. `custom` uses the `system_prompt` widget.
- `system_prompt`: visible custom system instruction widget
- `prompt`: visible task prompt widget
- `images`: native ComfyUI `IMAGE` input, compatible with Load Image and VHS/image-sequence frame batches
- `text_input`: connected text to analyze

System prompt presets:

- `custom`: use the system prompt widget exactly as written.
- `enhance_video_ltx23`: turn input text plus optional image into one flowing LTX-2.3 video prompt with shot, scene, action, character cues, camera movement, atmosphere, and audio.
- `enhance_video_wan22`: turn input text plus optional image into a detailed Wan2.2 video prompt, preserving image identity for I2V/TI2V and enriching motion, setting, lighting, and camera language.
- `caption_image_*`: caption a single image.
- `caption_video_*`: caption sampled video frames as one coherent clip.

Caption preset suffixes:

- Detail: `simple`, `detailed`, `very_detailed`.
- Style: `mixed`, `tag`, `natural`.
- `mixed`: booru-style tags followed by one natural-language sentence.
- `tag`: comma-separated WD14/Pony/Illustrious-style tags only.
- `natural`: descriptive natural language for FLUX, Wan, LTX, SD3, and similar prompt-following models.

Video/image-sequence handling:

- ComfyUI and VHS expose videos as an `IMAGE` batch.
- `max_frames` limits how many frames are sent to the VLM.
- `frame_strategy` chooses first, middle, last, every nth, or evenly spaced frames.
- `resize_max_px` downscales frames before inference to save VRAM.
- `resize_algorithm` selects the downscale filter: `lanczos`, `bicubic`, `bilinear`, `hamming`, `box`, or `nearest`.
- `max_input_tokens` optionally truncates long text/context input before generation. This can reduce attention memory for long prompts.
- `use_kv_cache` is a per-generation toggle for Transformers. Turning it off may reduce peak memory for some models, but generation is slower. The implementation and quantization strategy belong to Model Selector because they are backend configuration.
- `memory_cleanup` uses the same full cleanup path before and/or after the node: DaSiWa model cache, ComfyUI managed models, Python garbage, and the device allocator are cleared. This intentionally makes later image/video models reload rather than retain VRAM/RAM.

## Notes

Text-only LLMs can analyze text and prompts. Image or video-frame analysis currently requires a vision-language model with a compatible `AutoProcessor`, such as Qwen-VL/LLaVA-style Transformers model folders. The initial llama.cpp and Ollama backends are text-only.

Image compression is intentionally not exposed as a memory option. Lossless compression can preserve file quality, but after the VLM processor decodes the image it does not reduce vision token count or runtime VRAM. Use `max_frames` and `resize_max_px` for image/video memory control.

GGUF loading is provided through llama.cpp, not through ComfyUI diffusion checkpoint loaders. ComfyUI's native `MODEL` / `CLIP` / `VAE` objects represent diffusion models and cannot be used as LLM/VLM Transformers objects.
