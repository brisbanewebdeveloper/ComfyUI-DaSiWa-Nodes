# News & Changelog

Collection-wide news and change history for the DaSiWa Custom Nodes — one place to see what changed across every node. Per-node deep dives (UI guides, wiring, options) stay in their own docs, linked from the README.

This changelog covers the last two months: **2026-06-29 → 2026-08-29**. The first commit in this window is from 2026-07-05; older history lives in the git log. All entries are listed **newest first**.

## News

- **Advanced LoRA Loader trash button (08-29):** every slot row gets a trash button (plain canvas paths, ASCII — a drawn trash-can, no emoji) directly right of the info button — it resets that slot's LoRA back to "None" (strengths and multipliers kept, exactly like selecting None in the picker), so you can unstack a LoRA without reopening the picker. The info button shifted slightly left to make room. No backend change; version bump to 0.4.29.
- **Seed Control queue roll (08-29):** Random mode rolls a fresh seed on every prompt build under the new Vue frontend — the legacy extension-level queue hook stopped dispatching there, and the Run button's internal queue path doesn't call the global queue entry point either, so the roll now runs by wrapping the app's prompt build (`app.graphToPrompt`), the choke point every Run routes through (Pixaroma-seed-node style; Fixed mode and external-seed links unchanged).
- **LoRA info button (08-29):** the Advanced LoRA Loader rows now have an ⓘ glyph at the right edge. It opens a panel with the LoRA's Civitai link (looked up by the file's SHA-256, cached in `lorainfo/`), trigger/trained words from the safetensors header and Civitai (click-select, copy), and preview images (Civitai + a local sidecar `*.png` next to the LoRA if present).
- **Advanced LoRA Loader: universal rename + PDD/ACC support + opt-in cache (08-29):** the LTX-2-only loader is renamed to a universal **Advanced LoRA Loader** (serialized node ID unchanged), forwards PDD/ACC LoRA metadata to Core so PDD/ACC head banks activate, and gains an **opt-in** `use_cache` button (default off) that caches each unique LoRA file across slots.
- **H3 Cache compatibility & quality parity (08-29):** PDD LoRA head bank support (ComfyUI 0.34+) and per-token denoise-mask parity with Core.
- **Image Inpaint mode for the Director (08-28):** a 5-frame image-to-video pass through the native `MiniMaxH3ImageToVideo` node; the `inpaint_requested` output lets downstream sampling branch on mode.
- **Director 2.0 frozen (08-28):** the experimental v2 fork is removed from the nodepack and preserved under `frozen/`; v1 plus Image Inpaint is the supported path.
- **Seed Control node (08-26):** the Director's seed panel extracted into a standalone node with full 64-bit seeds, a Random|Fixed switch, and a NOISE-compatible output.
- **Issue forms (08-17):** structured bug-report and feature-request forms; the DaSiWa node list syncs automatically from the node registry.
- **MiniMax H3 family complete (08-16):** the collection now ships the full H3 stack — **MiniMax H3 Director** (timeline authoring, integrated 08-03), **MiniMax H3 Cache** (approximate block-stack residual cache), and **Patch Comfy Kitchen Attention** (INT8 attention model patch).
- **License: Apache 2.0 → GPL v3 (08-04).**
- **New authoring nodes (July):** Wildcard & Preset Prompt Builder, LLM / VLM Analyze (GGUF and Ollama backends), DaSiWa System Monitor, DaSiWa Torch Resize, and Enhanced Video Combine.

## Releases

Quick reference for the version bumps inside this window, newest first:

| Version | Date | Headline |
|---|---|---|
| 0.4.33 | 09-04 | System Monitor chips: stable fixed-width formatting; fast disks switch to GB/s (#36) |
| 0.4.30 | 08-29 | Seed Control: Random-mode roll via the graphToPrompt choke point; panel DOM syncs after a run |
| 0.4.29 | 08-29 | Advanced LoRA Loader trash button (per-row, resets the slot to None) |
| 0.4.28 | 08-29 | LoRA info button in the Advanced LoRA Loader (Civitai link, trigger words, images) |
| 0.4.27 | 08-29 | Advanced LoRA Loader universal rename; PDD/ACC metadata passthrough; opt-in cache button |
| 0.4.26 | 08-29 | H3 Cache PDD head-bank + per-token mask support |
| 0.4.25 | 08-28 | Director v1 Image Inpaint mode; `inpaint_requested` output switch |
| 0.4.24 | 08-28 | Director 2.0 freeze (v2 archived to `frozen/`) |
| 0.4.23 | 08-26 | Seed Control node + Director seed panel revamp |
| 0.4.22 | 08-25 | LTX-2.3 loader inline value editor |
| 0.4.21 | 08-23 | RTX lazy output allocation; Director v2 preview combo widget (pre-freeze) |
| 0.4.19 | 08-17 | RTX Upscaler `use_mmap` disk fallback + auto model unload; wildcard library additions |
| 0.4.17 | 08-16 | H3 Director prompt serialization + legacy preservation; H3 patch-grid alignment; script import fix (#28) |
| 0.4.14 | 08-16 | H3 Director resolution panel, grouped dropdowns, crop preview; H3 Cache + Kitchen Attention nodes added |
| 0.4.13 | 08-15 | System Monitor chip sizing + container-safe probes; Video Combine audio counter fix; LoRA loader nullish fallback |
| 0.4.10 | 08-10 | REF2VA `imd` KeyError and packed stereo audio duration fixes (PR #15, #17); external prompt input (PR #21, 08-12) |
| 0.4.6 | 08-05 | H3 Director prompt-builder suite (builders, thumbnails, helper buttons) |
| 0.4.5 | 08-05 | MythicAlchemy v12 H3 workflow |
| 0.4.1 | 08-04 | H3 Director docs + changelog; license switch to GPL v3 |
| 0.3.8 | 08-03 | MiniMax H3 Director integrated; hardened video preview fallback |
| 0.2.17 | 07-21 | Animated image outputs (AVIF / WebP) |
| 0.2.12 | 07-17 | DaSiWa Torch Resize node |
| 0.2.9 | 07-16 | DaSiWa System Monitor node |

## Changelog

### MiniMax H3 Director (v1)

- **08-28:** **Director 2.0 frozen:** the v2 fork is removed from the nodepack and preserved under `frozen/`; Image Inpaint is documented as a v1 feature.
- **08-28:** **Image Inpaint mode** in v1: normalizer, 5-frame single-image guide, Guide-node conditioning + latent emission; `ref2va_requested` output renamed to `inpaint_requested`; request outputs now follow the mode widget; auto aspect resolves by timeline slot instead of insertion order.
- **08-26:** seed panel revamp: spinner column slimmed and aligned with the input.
- **08-19 / 08-21:** Director forked into independent v1 + v2 on 08-19; v2 gained a seed control panel, pill modals, external sampling/shift override sockets, and a socketless preview tiny-VAE combo widget (08-23); resized prompt-field heights now persist across re-renders and workflow reloads in both versions (08-21); verified dead code removed from both.
- **08-16:** Resolution/Aspect/Input-Scaling panel with external overwrite inputs; `frame_rate` FLOAT input + output (legacy `external_prompt` input dropped); paste-replace onto a selected tile; WAV `.wave` alias + RIFF duration fallback; ▶ Play crop button with a draggable preview range; Ctrl+Enter run shortcut preserved and timeline wheel forwarded to the canvas; resolution dropdowns grouped by orientation / ###p / MP with **Native (ShortEdge 768px / 2048px)** labels; `non_diegetic_music` starts empty (N/A only at assembly time); nodes re-registered under the **DaSiWa/MiniMax H3** category; canvases aligned to the H3 32-px patch grid; legacy pre-builder prompts migrated into a Simple-prompt builder state instead of being dropped; `emit()` now serializes the resolved prompt back to the `prompt` widget; fixed the double `../` in scripts/app.js + api.js imports (#28). Simple/Structured prompt-mode toggle (persisted, honored by Preview Prompt).
- **08-10:** external prompt input. (PR #21)
- **08-08:** packed stereo audio de-interleaved so stereo WAV references keep their real duration; integer PCM scaled by magnitude (a full-scale sample no longer trips the ~90 dB attenuation guard). (PR #17)
- **08-06:** fixed `KeyError 'imd'` on every REF2VA run; ref schema normalized for mixed v1/v2 workflows.
- **08-05:** prompt-builder suite: unified builder UI with mode-aware forms, REF2VA simplified to six free-text fields with v1→v2 backward-compat merge, **Insert [Shot N]** / **Prefill Labels & Summary** / **Preview Prompt** helper buttons, video thumbnail previews, single-line toolbar with Clear/Remove/?, dark-blue audio lane, and prompt-builder fields persisted across workflow reloads. Fixes: trim sliders only drag when grabbed, images drag out of locked L2VA slots, slot-capacity checks corrected, KeyError guards on `p2_shot`/`last_shot`, non-string builder values, textarea onChange receives the value instead of the Event.
- **08-04:** full refresh; helper/JS/test alignment; native generation-length alignment; VAE/CLIP naming corrected to the Comfy-Org repo; wiring diagram added and "Guide replaces native nodes" clarified in docs.
- **08-03:** initial integration of the timeline-based H3 Director and Guide nodes.
- **Earlier additions (pre-window):** embedded video audio extraction (a video can supply its own audio reference; V / A / V+A stream switch per video clip), standalone-audio trim support with waveform preview + draggable crop markers, attached soundtracks that share a video's trim window, FL2VA ↔ REF2VA mode-switch safety (incompatible references are preserved, not deleted), and hardened video duration detection with a container-level fallback.

### MiniMax H3 Cache & Patch Comfy Kitchen Attention

- **08-29:** **PDD compatibility:** the node detects the live `FinalLayer.forward` signature at patch time and passes the ComfyUI 0.34+ PDD sigma-schedule arguments, so the PDD LoRA head bank works with cache enabled. **Per-token masks:** honors `denoise_mask` / `audio_denoise_mask` exactly like Core — mixed masks run masked rows at their own strength via per-row `rows_to_mod_index` modulation; absent or uniform masks collapse to the scalar path, byte-identical to the previous behaviour. **Spectrum patch artifact:** `patches/comfyui-spectrum-minimax-h3-pdd.patch` for xmarre's ComfyUI-Spectrum-MiniMax-H3 v0.2.20, which silently degrades on the PDD signature.
- **08-16:** both nodes added: an approximate, model-scoped whole-block-stack residual cache (relative-L1 threshold sampling, 15–90% sampling window, bounded cache hits, auto/CUDA/CPU storage) and a one-input INT8-attention model patch; both are model-clone patches and chain in either order.

### Seed Control

- **08-26:** new standalone Seed Control node extracted from the Director seed panel: full 64-bit unsigned seeds, Random|Fixed segmented switch, spinner with hold-to-repeat, Last 10 seeds history, external override socket, and INT + NOISE outputs; lossless 64-bit display via decimal-string mirroring; mode, last seed, and history persist with the workflow.

### Advanced LoRA Loader (formerly LTX-2.3)

- **08-29:** **Trash button (0.4.29):** each slot row gained a trash button (plain canvas paths, ASCII-drawn trash-can, no emoji) placed directly right of the info button. Clicking it resets that slot's LoRA back to "None" (the STR / VIS / A multipliers are kept — exactly like picking "None" in the slot picker), so a LoRA can be unstacked without reopening the picker. The info button shifted slightly left (x 976 → 962) to make room; the trash button occupies the previous info x. UI-only, no backend change.
- **08-29:** **LoRA info button (0.4.28):** each slot row gained an ⓘ glyph at its right edge. It opens an info panel with the LoRA's Civitai link (SHA-256 looked up on Civitai's `model-versions/by-hash` API, results cached in `lorainfo/`), trigger/trained words from the safetensors header and Civitai (click-select, copy), and preview images (Civitai plus a local sidecar image next to the LoRA if present). Two new GET-only routes (`/dasiwa/ltx2/lorainfo`, `/dasiwa/ltx2/loraimg`) in `nodes/lora_info.py`; no new dependencies.
- **08-29:** **PDD/Acc LoRA head-bank guard (warn-only):** when a PDD LoRA (`pdd_num_steps`) is applied to a single-head H3 model, the loader prints a console warning — the incompatible `final_layer` head-bank `set`/`bias` keys are **kept**, so the protective core shape crash at `comfy/lora.py` still aborts the run (deliberate: no circumvention until an upstream patch lands). The guard warns only on positive evidence (bank width and model width both readable and different); a genuine PDD model whose width matches is untouched. **Recommended fix:** pair PDD LoRAs with a PDD model (`final_layer.video_out [3072,5376]`). Bugfix — no version bump.
- **08-29:** **universal rename + PDD/ACC support + opt-in cache:** the LTX-2-only `DaSiWa LTX-2 Master Loader` is renamed to the **Advanced LoRA Loader** — module/class/JS/docs/display name changed internally, but the *serialized* node ID `DaSiWa_LTX2LoraLoader` and display name stay stable for saved workflows (display now drops the "DaSiWa" prefix). The file is read with `return_metadata=True` and the PDD/ACC metadata is forwarded to Core's `load_lora_for_models`, so PDD / ACC LoRA head banks activate (older builds fall back via `TypeError`). A new **⚡ CACHE** button in the control strip enables an opt-in per-path LRU cache (max 4 entries, **off by default**), so a LoRA reused across slots is read once.
- **08-25:** **inline value editor:** the STR, VIS, and AUDIO pills open an in-canvas editor instead of a `prompt()` dialog; the editor is pinned to its pill and tracks pan and zoom; documented.
- **08-15:** STR / VIS / A value editors no longer bounce off 0 (nullish fallback).
- **07-11:** LoRA list refreshes dynamically.

### RTX Upscaler & Refiner

- **08-23:** output batch now allocated lazily (kernel-decided, no up-front memory pressure); `use_mmap` **off by default**.
- **08-17:** **`use_mmap` disk-backed fallback** as a permission switch (VRAM → RAM → disk chain) plus default-on `auto_unload_models`.
- **08-14:** `empty_cache` made optional to restore API workflow compatibility (fixes #24, PR #26).
- **08-06:** VRAM-aware output allocation; optional `empty_cache` switch; fixed `use_mmap` swap bloat.
- **08-01:** adaptive CPU RAM reserve shared with the Watermark compositor.

### Enhanced Video Combine

- **08-22 / 08-25:** preview checkboxes (Autoplay, Mute) persist across reloads (PR #30); permanent Mute checkbox persisted with node properties.
- **08-21:** drifted combo/boolean widget values self-heal on load; audio_codec positional drift repair; MythicAlchemy v16 workflow with clean video-combine widgets.
- **08-15:** fixed audio outputs overwriting each other (counter always reset to 1).
- **08-03:** browser-compatible AV1 auto-encoding; Auto codec precedence documented (AV1 → VP9 → H.264; H.265 excluded from Auto).
- **07-22:** video frames streamed to FFmpeg instead of temp files.
- **07-21:** animated image outputs: Animated AVIF (GPU AV1 or software) and Animated WebP, with accelerated AVIF encoding.
- **07-19:** node added (IMAGE batch → video with optional AUDIO muxing and in-node preview); frame exports published to ComfyUI Assets; re-encode on queued runs; audio settings preserved in saved workflows.

### Wildcard & Preset Prompt Builder

- **08-17:** library additions: Background, Intimate Backgrounds, Weapons & Items; duplicate entries merged.
- **07-31:** segment picker interface; picker layout restored after workflow reload.
- **07-30:** node added: dual Booru / Natural-Language wildcard library, weighted bounded prompts, reproducible rerolls.

### LLM / VLM Analyze

- **07-30:** LLM cache and GGUF backends added (local GGUF via llama.cpp alongside Ollama and Hugging Face download).

### DaSiWa System Monitor

- **09-04:** stable chip width — % and °C values pad to 3 digits and disk throughput to one decimal, so the toolbar stops reflowing as values tick; values ≥ 1000 MB/s switch to `x.x GB/s` so fast (M2-class) disks don't overflow the chip (issue #36).
- **08-28:** Windows CIM probe now cached — no more per-second PowerShell spawns.
- **08-15:** content-sized chips (no more LITE text clipping at 4K); real disable flag + container-safe probes (`DASWA_SYSTEM_MONITOR=0`).
- **08-01:** disk telemetry + docking.
- **07-16:** node added; configurable Lite/Full display; dockable placement.

### Torch Resize

- **07-17:** node added: Lanczos batch-aware resizing on the PyTorch build ComfyUI already uses; README section rewritten as user-facing benefits.

### Resolution Scale Calculator

- **07-16:** matches the ComfyUI-native MP convention (1024²) (#4); new 1 MP resolution preset.

### Node Status Switch

- **07-31:** promoted boolean inputs now sync live.

### Metadata Image Saver

- **07-31:** skips cleanly when no image is received.

### Repository & tooling

- **08-23 / 08-26:** DaSiWa node list in the bug-report issue form synced.
- **08-19:** `.projectatlas/` build artifacts gitignored.
- **08-17:** node dropdown auto-synced from `NODE_DISPLAY_NAME_MAPPINGS`.
- **08-16:** bug-report + feature-request issue forms added; unregistered System Monitor entry dropped from the node dropdown.
- **08-04:** **license switched from Apache 2.0 to GPL v3.**
- **08-01:** node output logging centralized; modules renamed; tests folder moved behind dot-prefix.
- **07-19:** local Hermes state ignored; watermark assets + implementation plans documented.
- **07-05:** supports the new ComfyUI `validate_inputs` signature.
