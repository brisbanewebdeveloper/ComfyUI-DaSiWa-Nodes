# MiniMax H3 Director

A timeline-based authoring node for ComfyUI's native MiniMax H3 models. It centralizes media management, ordering, trimming, per-reference prompts, and the global prompt into one workflow node, validates H3 constraints before execution, and routes everything to the installed native MiniMax H3 implementation — no duplicate backend logic.

## Changelog (latest additions)

Since the last GitHub release (August 2026):

### New features

- **Optional integrated executor:** **DaSiWa MiniMax H3 Director Executor** consumes the existing guide and performs native conditioning, SigmaShift, sampling, video decode, and optional audio decode. The Guide remains available for external sampler graphs.
- **Simple / Structured prompt-mode toggle:** a mode-bar switch changes how builder fields assemble into the final prompt — **Structured** keeps the labelled sections (headers added upstream), **Simple** renders one flat, header-less block. Persisted in `builder_state` (`prompt_mode`), restored on load, honored by **Preview Prompt**; defaults to Structured for backward compatibility.
- **Grouped column dropdowns:** Aspect / Resolution / Input scaling now render as grouped, ascending columns (aspect by orientation, resolution by ###p / MP tier) with the auto options relabelled **Native (ShortEdge 768px)** (Resolution) and **Native (ShortEdge 2048px)** (Input scaling); menus clamp to the node viewport and scroll natively with the mouse wheel.
- **Frame rate:** new `frame_rate` FLOAT input (0.1–240, default 24) sets the output FPS and is emitted as a `frame_rate` output for downstream nodes to read; the legacy `external_prompt` input was dropped in favour of `external_prompt_overwrite`.
- **Crop preview:** a ▶ Play crop button plays only the current crop range, and the preview range itself is draggable for quick scrubbing.
- **Paste-replace:** pasting (Ctrl+V) onto a selected media tile replaces that tile in place, preserving its slot position instead of appending to the end.
- **Category rename:** the Director and Guide nodes now register under the **DaSiWa/MiniMax H3** ComfyUI category.
- **Companion nodes:** the MiniMax H3 family gains **MiniMax H3 Cache** (approximate block-stack residual cache) and **Patch Comfy Kitchen Attention** (INT8 attention model patch) — see the README and [minimax_h3_cache.md](minimax_h3_cache.md).
- **non_diegetic_music default:** the field starts empty in all modes; `N/A` is applied only at prompt-assembly time when the field is left blank.
- **Resolution panel:** new Aspect / Resolution / Input scaling selectors under the mode controls (all default to **Auto**). Auto aspect follows the first active visual reference (4:3 fallback with no media); Auto resolution sets a 768 px short side. Fixed aspect ratios, DaSiWa MP and fixed-resolution presets, and CUSTOM values all snap to H3's 32-px transformer-patch grid. H3's VAE has 16-px latent cells, but its transformer groups them in 2×2 patches; a 16-px-only edge can therefore fail at sampling. Input scaling (Off / Auto / Target / Fit / Fill and crop / Fit and pad / Long side with divisible crop) preprocesses visual references through the included DaSiWa Torch Resize before they reach H3 — Auto never upscales sources at or below a 2048 px short edge.
- **External overwrite inputs:** optional `external_prompt_overwrite` (STRING) replaces the assembled builder output when connected and non-empty. Connect both `external_width_overwrite` and `external_height_overwrite` (INT) to replace the Director canvas; they accept arbitrary positive values and their alignment is the user's responsibility. An active complete dimension pair disables the Director Aspect/Resolution/Input Scaling controls and passes visual media through unchanged.
- **REF2VA prompt builder redesigned:** simplified to six free-text fields (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music). Headers are added automatically when the prompt is sent upstream. Legacy v1 structured-builder data is merged backward-compatibly into these fields.
- **Insert [Shot N] button:** opens a small dialog asking for a shot number, then inserts `[Shot N] ` at the current cursor position in the appropriate text area — no more manual typing.
- **Prefill Labels & Summary button (REF2VA):** scans your inserted media and generates `<Picture N>`, `<Video N>`, and `<Audio N>` label lines plus a task-prefixed summary line so you can focus on editing instead of boilerplate.
- **Preview Prompt button:** opens a popup showing the exact prompt that will be sent to MiniMax H3 (including all section headers, alignment lines, and assembled fields). Includes a copy-to-clipboard button so you can verify or reuse the result offline.
- **Video thumbnail previews:** every uploaded video now extracts its first frame and displays it as a background behind the clip tile, replacing the generic icon. Makes it easy to tell references apart without opening each one.
- **Cleaner toolbar:** consolidated into a single horizontal row with mode buttons on the left and Clear/Remove/? controls on the right; removed redundant bubble elements and pulsing glows for a quieter interface.
- **Dark-blue audio lane accent:** distinguishes the audio lane visually from the Image/Video lane and the green "+ empty-slot" indicators.
- **Sampling controls + external override sockets:** a new **Sampling** row exposes `sampler`, `scheduler`, `steps`, `shift_video` and `shift_audio` (persisted into the hidden `internal_execution` block, round-tripped through `timeline_data`). Five optional inputs (`external_sampler`, `external_scheduler`, `external_steps`, `external_shift_video`, `external_shift_audio`) let another node override them; precedence is external > internal > built-in default, and a connected socket disables the local fields with an "external sampling connected" note.
- **Built-in live step preview:** the Director now decodes per-step frames itself (no KJ preview node needed) and streams them into its own **Preview & Output** panel. Toggle and limits live in the ☰ *Preview & Output options* menu: **live step preview** (default on, true bypass when off), **max resolution** (default 1024 px), **frames** (1 = still JPEG, >1 = animated WebP or NVENC H.264 MP4) and **fps**. Decoding precedence: `preview_tiny_vae` widget (a `models/vae_approx` combo rendered like a plain model selector — the Director's JS strips the optional socket's ring so no input dot shows, e.g. `taeh3.safetensors`, the H3 tiny decoder core's VAELoader cannot build) > `preview_vae` socket (full-quality `vae.decode()`) > core previewer > latent-to-RGB fallback. The media preview popup is a third narrower than before (400 px instead of 600 px).

### Earlier additions

- **Embedded video audio extraction:** videos can now supply their own audio reference. Select A or V+A on a video clip to decode its embedded audio track with the same trim range used for frames.
- **Per-video stream switch (V / A / V+A):** compact controls on each video tile let you treat it as Video only, Audio only, or Video+embedded-audio without adding separate audio slots.
- **Trim support for standalone audio:** audio clips now show a waveform preview and draggable left/right crop markers, behaving identically to video trimming.
- **Attached soundtrack trim alignment:** external soundtracks linked to a video share its trim window automatically.
- **Mode-switch safety:** toggling FL2VA ↔ REF2VA preserves incompatible references instead of deleting them; they reappear when you switch back.
- **Hardened video duration detection:** fallback to container-level duration when stream metadata is incomplete.

### Bug fixes

- **Director node-ID collision:** the DaSiWa Director now registers as `DaSiWaMiniMaxH3Director` instead of the shared `MiniMaxH3Director` ID, so it can coexist with other Director packages. DaSiWa workflow graphs saved under the old ID are identified by their socket/widget schema and migrated before graph configuration.
- **Legacy Director prompt preservation:** pre-builder Director workflows stored their editable prompt in the old `prompt` widget or the timeline JSON, before `builder_state` existed. On load, such a prompt is now detected and migrated into a lossless Simple-prompt builder state instead of being silently dropped, so old videos drop into ComfyUI with their prompt intact. Existing builder content always wins, so current workflows load idempotently.
- **Serialized standard prompt:** `emit()` now writes the resolved prompt back to the `prompt` widget (and fires its callback) on every state change, so the standard ComfyUI prompt widget stays in sync with the assembled builder output and round-trips through `widgets_values` on save/load.
- **Legacy widget order preserved (`frame_rate` moved to the end):** the `frame_rate` input inserted at widget position 5 in a recent change shifted every pre-`frame_rate` save one slot — old videos loaded with an empty prompt and crashed the queue (`float('match')`). `frame_rate` is now appended **last** in the required input list so the eight legacy widgets keep their original positional order; `build_guide` additionally coerces a non-numeric `frame_rate` (e.g. the stale 9th value from an old save) to the 24.0 default instead of raising.
- **WAV `.wave` extension + RIFF duration fallback:** `.wave` is now accepted as an audio extension alongside `.wav`, and when the container/stream carries no usable duration the loader falls back to parsing the RIFF data-chunk size so a clip still reports a real length instead of failing or mis-sizing.
- **Ctrl+Enter run shortcut preserved:** typing in builder text areas no longer swallows the ComfyUI Ctrl+Enter "queue prompt" shortcut, and timeline wheel events are forwarded to the canvas so the graph still scrolls while hovering the node.
- **Crop-playback end guard:** removed the pre-seek pause that was clearing the crop-range end guard, so ▶ Play crop stays clamped to the selected range.
- **Packed stereo audio duration (WAV/PCM):** PyAV returns planar formats (MP3/AAC/OGG) as `(channels, samples)` but packed formats (s16 PCM WAV, flt, s32) as one interleaved row. `load_audio()` read the shape as always `(channels, samples)`, so a stereo WAV measured twice its real length — a 10 s reference was rejected as "20 s", and under 7.5 s it reached the model at double speed with every crop offset on the wrong sample. New `decode_audio_frame()` derives channel count from the frame, de-interleaves packed frames, and normalizes integer PCM to float; both the standalone-audio and embedded-video-audio loaders share it.
- **Integer PCM scaling by magnitude:** signed PCM was divided by `np.iinfo(dtype).max`, leaving a full-scale `-32768` at `-1.00003` (just outside the unit range). A single full-scale sample then tripped a legacy `abs().max() > 1` guard in `load_embedded_video_audio()`, which divided the whole soundtrack by 32768 — a ~90 dB attenuation. Signed formats are now scaled by `-iinfo(dtype).min` and unsigned 8-bit (silence at 128) is centered on its midpoint; the legacy guard is dropped because `decode_audio_frame()` now guarantees the unit range.
- **KeyError `'imd'` on every REF2VA run (PR #15):** REF2VA prompts no longer hit a missing `imd` key during prompt assembly.
- Non-string values in prompt-builder fields no longer crash with `.strip()` TypeError.
- Textarea `onChange` callbacks now receive the string value, not the raw Event object.
- Missing `p2_shot`/`last_shot` builder keys guarded against KeyError outside FL2VA/L2VA modes.
- Trim sliders only respond when clicking directly on crop markers, preventing accidental drags on the track background.
- Images can now be dragged out of locked L2VA slot positions instead of being stuck forever.
- Slot-capacity checks corrected so images are accepted in all valid endpoint modes.
- Center drag handles restored on prompt-field resize bars.

## Quick overview

- One node holds all your references, trims, ordering, endpoint frames, and prompts.
- Two modes: FL2VA (text/image endpoints) and REF2VA (multi-image/video/audio references).
- Two timeline lanes: Image/Video + Audio. Click a lane to select it; paste / drop media there.
- Per-video stream switch: choose Video only, Audio only, or Video+embedded-audio with identical trim ranges.
- Standalone audio clips can be trimmed with left/right handles just like video.
- Video thumbnails: each uploaded video shows its first frame as a background preview behind the clip tile.
- Simple / Structured prompt mode: toggle how builder fields assemble into the final prompt (persisted per workflow).
- Frame rate: `frame_rate` input (0.1–240, default 24) sets the output FPS and is re-emitted as an output.
- Crop preview: ▶ Play crop plays only the current crop range; the preview range is draggable.
- Paste-replace: pasting over a selected tile replaces it in place, keeping its slot.
- Mode-specific prompt builders:
  - FL2VA/I2VA/L2VA/T2VA: guided fields for description and audio sections with automatic alignment headers.
  - REF2VA: six free-text sections (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music) with helper buttons — Insert Shot, Prefill Labels & Summary, and Preview Prompt.
- Only the selected FL2VA or REF2VA model is loaded; the Guide node calls ComfyUI's built-in H3 nodes.

## Installation and graph setup

Install dependencies and restart ComfyUI:

```bash
pip install -r requirements.txt
```

Ensure your ComfyUI version includes native MiniMax H3 support. Add the Director and choose one execution route from `DaSiWa/MiniMax H3`:

1. **DaSiWa MiniMax H3 Director** — your timeline, references, and prompt editor.
2. **MiniMax H3 Director Guide** — validation and routing to native H3 nodes.
3. **DaSiWa MiniMax H3 Director Executor** — optional integrated sampling and decode route.

Wire them like this:

```text
┌──────────────────────────────────────┐
│ UNET Loader                          │     diffusion_models/*.safetensors
│ CLIP Loader                          │     text_encoders/qwen3vl_32b_minimax_h3_*.safetensors
│ VAE Loader (visual)                  │     vae/minimax_h3_video_vae_fp16.safetensors
│ VAE Loader (audio)                   │     vae/minimax_h3_audio_vae_fp32.safetensors
│                                      │     (only for REF2VA mode)
└───┬────────────┬──────────┬──────────┘
    │            │          │
    ▼            ▼          ▼
┌──────────────────────────────────────────────┐
│ DaSiWa MiniMax H3 Director                   │
│  - add/edit references                       │
│  - set trims, ordering, prompts              │
│  - emits structured "guide" dict             │
└────┬─────────────────────────────────────────┘
     │ guide
     ▼
┌──────────────────────────────────────────────┐
│ DaSiwa MiniMax H3 Director Guide             │
│  - validates director output                 │
│  - assembles final prompt                    │
│  - CALLS the native ComfyUI H3 nodes:        │
│      • MiniMaxH3ImageToVideo   (FL2VA)       │
│      • MiniMaxH3ReferenceToVideo (REF2VA)    │
│  - you NEVER wire those native nodes yourself│
└────┬─────────────────────────────────────────┘
     │ positive, latent
     ▼
┌──────────────────────────────────────────────┐
│ Standard ComfyUI sampling/decoding chain     │
│  - KSampler                                  │
│  - VAE Decode                                │
│  - Enhanced Video Combine / Image Save etc.  │
└──────────────────────────────────────────────┘
```

Connections detail:

- Director `guide` → Guide `guide` for a custom sampling graph, or Executor `guide` for the integrated route
- `CLIP` → Guide `clip`
- Visual `VAE` → Guide `vae`
- In REF2VA: audio VAE → Guide `audio_vae`
- Guide outputs `positive` and `latent` → standard MiniMax H3 sampler/decoder chain

For the integrated route:

- Selected MiniMax H3 model → Executor `model`
- `CLIP` → Executor `clip`
- Visual `VAE` → Executor `video_vae`
- Director `guide` → Executor `guide`
- In REF2VA, or to decode generated FL2VA audio: audio VAE → Executor `audio_vae`
- Executor outputs decoded images/audio plus FPS, frame count, sampled latent, and a run report

### Multi-segment sequence route

The sequence route leaves the existing Director, Guide, and single-shot Executor unchanged:

1. Create one Director node per shot and connect each `guide` to **DaSiWa MiniMax H3 Sequence Segment**.
2. Set `run` per segment. When `run` is off, provide `source_images` (and optionally `source_audio`) or a matching disk cache.
3. Give cached segments a stable `cache_key`. Change that key whenever their reference or source media changes.
4. Add segment outputs to the autogrowing **DaSiWa MiniMax H3 Sequence Plan** in playback order.
5. Connect the plan, CLIP, visual VAE, audio VAE, and the FL2VA/REF2VA models to **DaSiWa MiniMax H3 Sequence Executor**.

The Executor provides:

- lazy model routing for mixed FL2VA and REF2VA plans;
- deterministic per-segment seeds (`seed + segment index + seed_offset`);
- optional 5/22/39/56-frame synchronized AV latent continuity;
- CPU-only retained handoff latents and model cleanup between segments;
- local disk cache/resume under `output/dasiwa_h3_sequence_cache`;
- cached or source passthrough for unselected segments;
- concatenated `images` and `audio`, plus FPS, total frame count, and a JSON run report.
- native 24 FPS output, matching MiniMax H3's fixed synchronized AV grid.

All guides in one plan must use the same canvas. `cache_key` is deliberately explicit because loaded reference tensors do not retain a reliable content identity; reusing a key after media changes can load stale output. Cache files are local and are read with PyTorch's restricted weights-only loader.

Important: the Guide node replaces and wraps ComfyUI's native `MiniMaxH3ImageToVideo` and `MiniMaxH3ReferenceToVideo` nodes. You do not add or wire those native nodes yourself — the Guide calls them internally based on the chosen mode.

The Director has optional model sockets (`fl2va_model`, `ref2va_model`) for lazy loading: connect whichever model matches your active mode. The Guide refuses REF2VA without an audio VAE connected.

The registered Director node ID is `DaSiWaMiniMaxH3Director`. Workflows opened in the ComfyUI frontend are automatically migrated from the former DaSiWa `MiniMaxH3Director` ID; API prompt JSON should use the new ID directly.

## Modes at a glance

### FL2VA (First/Last Frame to Video)

Use for text-only generation, single-image starting frames, or first+last frame interpolation.

- Zero images → pure T2VA
- One image → first-frame conditioning (I2VA-style)
- Two images → first and last frame interpolation (true FL2VA)
- Up to 2 image slots; video/audio files are blocked in this mode.
- Endpoint images require an alignment instruction line as the very first line of the global prompt.

Alignment instruction patterns:

For one endpoint (first frame only):
```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: ...
```

For two endpoints (first and last frame):
```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {duration}.00-second mark of the target video.

integrated_multimodal_description: ...
```

Replace `{duration}` with your Director duration setting (e.g. `8.00`). Images map left-to-right by slot order: first image = Picture 1, second = Picture 2. FL2VA always uses `integrated_multimodal_description`, not `detailed_description`.

### REF2VA (Reference to Video)

Use when you want the generated video to borrow identity, appearance, motion, composition, or sound from existing media.

- Up to 9 images, 3 videos, 3 audio clips, 12 files total.
- Each video clip offers three modes via small buttons on its clip tile:
  - **V** — Video only (frames as visual reference)
  - **A** — Audio only (extract embedded audio as reference; no video frames)
  - **V+A** — Video + embedded audio (both streams decoded with the same trim range)
- Standalone audio clips show a waveform preview and support left/right crop markers.
- Attach an external soundtrack file to any video; it shares the same trim window.
- Keep at least one image or video when using audio.

When a video provides audio (via A/V+A or attached soundtrack), that audio becomes `<Audio N>` in the reference numbering. Embedded audio and separate soundtracks share the same time crop as the host video frames.

## UI walkthrough: controls and buttons

Open the node and read top-to-bottom.

### Toolbar row

- **Title label:** "MiniMax H3 Director"
- **Model Mode buttons (left):** T2VA, I2VA, FL2VA, L2VA, REF2VA shown as small pills. Active mode has purple highlight; click to switch modes. When switching modes:
  - Going to FL2VA hides non-image references but keeps them in memory so they reappear when you switch back.
  - Going to REF2VA restores all previously added media.
- **Prompt Mode toggle:** a **Simple** / **Structured** pair next to the mode buttons switches how builder fields assemble into the final prompt (Structured keeps the labelled sections, Simple renders one flat block). The selection is persisted and restored on load.
- **Clear button:** always visible; removes all media and prompts from the timeline. With no content it is dimmed and reports "Nothing to clear." instead of clearing.
- **Remove button:** appears when a clip is selected; deletes that item.
- **? button:** opens the online documentation on GitHub.

### Timeline area

The main workspace has two horizontal lanes stacked vertically.

#### Lane selection

- Click anywhere on a lane to select it. The selected lane gets a highlight and displays "· selected".
- Selection determines where pasted media goes:
  - Select **Image/Video** lane, then Ctrl+V or drop files → images/videos land here.
  - Select **Audio** lane, then Ctrl+V or drop files → audio lands here.
- FL2VA disables the Audio lane entirely.

#### Adding media

Three ways:

1. **+ buttons:** each empty slot shows a "+"; click to open a file picker filtered for that lane type.
2. **Drag-and-drop:** drag files from your OS directly onto the desired lane.
3. **Paste:** select a lane, focus the node, press Ctrl+V with images/videos/audio on your clipboard.

Each uploaded file is stored in ComfyUI's `input/` directory and linked by relative path.

#### Clip tiles

Each media item renders as a labeled tile inside its lane.

Common elements:

- **Background preview:** images show themselves; videos display their first-frame thumbnail extracted at upload time. Makes it easy to identify references visually without opening previews.
- **Identity badge (top-left):** shows the MiniMax reference label used in prompts: `Picture 1`, `Video 2`, `Audio 1`, etc., assigned by type in timeline order.
- **Label text:** filename, duration, current crop range (for video/audio), and a truncated preview of the media prompt if present.
- **Selection outline:** click a tile to select it; enables the Remove button and populates the Media Prompt editor below.
- **Drag-to-reorder:** grab a tile and move it horizontally; dropping near another slot swaps positions.

Video-specific elements:

- **Stream selector (top-right):** three tiny buttons:
  - `V` — treat as video-only reference
  - `A` — extract and use only the embedded audio track
  - `V+A` — use both video frames and embedded audio
- **Crop markers:** vertical lines overlaid on the clip showing the selected start/end within the source. Drag these markers left/right to adjust which portion of the source is sent to MiniMax.
- **Crop readout:** displays current crop range vs source duration.

Audio-specific elements:

- **Waveform preview:** rendered canvas showing amplitude peaks across the entire source.
- **Crop markers:** same concept as video; drag to select the 2–15 second segment to use.

Images:

- No trimming; treated as single-frame anchors.

#### Removing/disabling items

- Click a tile then hit the toolbar **Remove** button, or press Delete/Backspace while a tile is selected.
- Items can also be disabled via the legacy list view (hidden by default).

### Prompt editors

Below the timeline is a unified prompt-builder panel whose layout depends on the active mode. Both editors have resizable text areas with drag-handle bars at the bottom; heights persist in the workflow JSON.

#### FL2VA / I2VA / L2VA / T2VA builder

Three labeled text areas:

- **integrated_multimodal_description** — main scene/action/camera/environment description with optional `[Shot N]` markers. An **Insert [Shot N]** button pops up a dialog and places the marker at your cursor.
- **overall_soundscape** — ambient sounds, dialogue, effects.
- **non_diegetic_music** — background score or `N/A`.

Alignment instruction lines (for I2VA/FL2VA/L2VA) are generated automatically based on mode and duration; you do not type them manually.

#### REF2VA builder

Six labeled text areas matching the official full-reference format. Section headers (`subject_definitions:` etc.) are appended automatically by the backend; you write only the content:

- **subject_definitions** — define `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>` entries and what each contributes.
- **summary** — task-type prefix (`[reference generation + audio reference]`) plus one-line intent statement.
- **retention_analysis** — per-label retention markers (`fully_preserved`, `attribute_transfer`, etc.) with brief rationale.
- **detailed_description** — shot-by-shot narrative using `[Shot N]` and timestamps.
- **overall_soundscape** — audio environment.
- **non_diegetic_music** — score or `N/A`.

Helper buttons above the fields:

- **Insert [Shot N]** — asks for a shot number, inserts `[Shot N] ` at the cursor in the `detailed_description` area.
- **Prefill Labels & Summary** — scans your timeline items and writes initial `<Picture N>`, `<Video N>`, `<Audio N>` label lines plus a summary template referencing them. Edit freely afterward.
- **Preview Prompt** — opens a popup showing exactly how the final prompt will look once section headers and any alignment lines are applied. Includes a copy-to-clipboard button.

## Limits and validation

MiniMax H3 enforces hard caps; the Director checks these before sending data downstream:

- Reference clip length: minimum 2 seconds, maximum 15 seconds each.
- Combined visual total: ≤ 15 seconds.
- Combined audio total: ≤ 15 seconds.
- Slot counts (REF2VA): max 9 images, 3 videos, 3 audio clips, 12 total files.
- FL2VA: max 2 images; no video/audio allowed.
- Path safety: all input paths resolve strictly under ComfyUI's input directory.

Violations appear as red status messages inside the node. Fix them before queuing.

## How processing flows: upstream and downstream

Understanding the data path makes wiring and debugging easier.

### Upstream inputs (what feeds into the Director)

- **Resolution panel / duration / frame rate:** the Resolution panel under the mode controls drives the hidden width/height widgets. **Aspect: Auto** reads the first image or video reference and preserves its aspect; **Resolution: Auto** sets the resulting short side to 768 px. Common horizontal/vertical aspect choices plus DaSiWa MP and fixed-resolution presets are rounded to MiniMax's 16-pixel grid. Both selectors offer **CUSTOM** values for manual aspect, MP, or exact pixels. All three selectors default to **Auto**. The third **Input scaling** selector preprocesses visual references through the included DaSiWa Torch Resize implementation before they reach H3: **Off** preserves the original tensor, **Auto** preserves its aspect with a 2048-px short edge only when that would downscale the source (smaller inputs pass through unchanged), **Target - Selected Aspect & Resolution** stretches it to the selected Director canvas, and **Fit**, **Fill and crop**, **Fit and pad**, and **Long side with divisible crop** use the corresponding Torch Resize aspect modes against that canvas. Audio is never resized. When both external dimension overwrite inputs are connected, these Director calculations and preprocessing are disabled. The `frame_rate` FLOAT input (0.1–240, default 24) sets the output frame rate and is re-emitted as a `frame_rate` output for downstream nodes.
- **Optional model sockets** (`fl2va_model`, `ref2va_model`): connect only the model matching your current mode; the Guide uses them lazily.
- All media is managed inside the node UI (upload/paste/drop), but paths ultimately live in ComfyUI's `input/` folder.

### Inside the Director

On queue, the Director executes this sequence:

1. Reads current mode (FL2VA or REF2VA).
2. Iterates over all enabled timeline items in slot order.
   - For FL2VA: keeps only image items (max 2); discards others temporarily.
   - For REF2VA: processes images, videos, and audio respecting slot limits.
3. Loads each asset:
   - Images → resized tensors.
   - Videos → decoded to `frame_rate` fps frame batches (default 24), cropped according to trim_start/trim_end.
   - Audio → decoded waveforms, cropped identically.
   - For videos in A or V+A mode → embedded audio is extracted using the same crop window.
   - Attached soundtracks → loaded and cropped using the host video's trim range.
4. Builds a structured `guide` dictionary containing:
   - Mode flag, dimensions, duration, and frame rate.
   - Ordered lists of images, videos, audios with metadata.
   - Endpoint frames (FL2VA).
   - Reference maps keyed as `ref_image_N`, `ref_video_N`, `ref_audio_N`, `ref_video_audio_N`.
5. Reads the mode-specific prompt-builder state:
   - FL2VA/I2VA/L2VA/T2VA: uses integrated_multimodal_description, overall_soundscape, non_diegetic_music fields.
   - REF2VA: uses the six free-text sections (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music).
   - Alignment lines (I2VA/FL2VA/L2VA) are injected automatically based on mode and duration.

This `guide` object and `builder_state` are passed out to the Guide node.

### The Guide node

The Guide is a thin adapter between your authored timeline and ComfyUI's native H3 nodes:

1. Validates the incoming `guide`:
   - Confirms mode consistency.
   - Checks that required models/CLIP/VAEs are connected.
   - For REF2VA, ensures an audio VAE exists.
2. Assembles the final prompt via the prompt-builder helper:
   - Reads `builder_state` from the Director.
   - For FL2VA/I2VA/L2VA/T2VA: injects alignment lines (when applicable), combines integrated_multimodal_description + overall_soundscape + non_diegetic_music into the canonical format.
   - For REF2VA: wraps the six user-written sections with their standard headers (`subject_definitions:`, `summary:`, etc.). Legacy v1 structured-builder data is merged automatically if present.
   - Writes the result as `resolved_prompt`.
3. Routes to the appropriate native node:
   - FL2VA → calls `MiniMaxH3ImageToVideo` with endpoint frames and prompt.
   - REF2VA → calls `MiniMaxH3ReferenceToVideo` with all reference maps and prompt.
4. Emits standard ComfyUI outputs:
   - `positive` (conditioning)
   - `latent` (image batch)
   - These feed downstream samplers and decoders exactly like any other H3 workflow.

You never call the native MiniMax H3 nodes directly when using Director+Guide; the Guide abstracts that away.

## Model chain: patching & preview

The Director's `fl2va_model` and `ref2va_model` inputs are plain `MODEL` sockets, so any node that outputs `MODEL` can sit upstream of the Director — a LoRA loader, the **MiniMax H3 Cache** patcher, **Patch Comfy Kitchen Attention**, or a KJ `ModelPreviewOverrideKJ`. This is the same forward chain you already use:

```
Checkpoint.MODEL → LoRALoader.MODEL → [KJ.model → KJ.MODEL] → Director.fl2va_model
Checkpoint.CLIP  → LoRALoader.clip  → Director.clip
```

Three rules keep the chain valid:

1. **Forward chain only — never a loop.** A patcher's output feeds *into* the Director's model input; it must never come back out of the Director. The Director is a terminal media node (it emits `frame_rate`, `duration`, `images`, never `MODEL`), and a wire from the Director back into its own model input would be a graph `dependency_cycle`, which ComfyUI's validation rejects.
2. **One loader per model, in mode order.** `select_execution_model` picks `ref2va_model` for REF2VA and `fl2va_model` otherwise; the active input must be connected (the unconnected twin may stay empty).
3. **Type-safe wires.** ComfyUI only lets you connect type-compatible sockets, so `LoRA.MODEL → Director.fl2va_model` is legal but `LoRA.MODEL → Director.clip` is not. No name or type resolution happens at runtime — the socket you plugged in arrives as the keyword-argument named for that socket.

### Sampling settings

The five sampling fields (`sampler`, `scheduler`, `steps`, `shift_video`, `shift_audio`) live in the **Sampling** row of the node and persist in `internal_execution` (round-tripped through `timeline_data`), surviving reloads. Backend precedence: **external socket > internal UI value > built-in default** (`res_multistep` / `simple` / 25 / 11 / 4). Connect an `external_*` sampling input to override from another node; an empty/zero external value falls back to the internal value, and a connected socket disables the local fields with a note.

### Built-in live step preview

The Director decodes per-step denoising frames itself and streams them into its own **Preview & Output** panel — no preview node is required, and the default ComfyUI previewer (`--preview-method`) needs no enabling: the wrapper is independent of it.

- **Toggle:** **live step preview** in the ☰ *Preview & Output options* menu (default on). When off the backend skips the whole decode path (no tiny-VAE decode, no `send_sync`) — a true bypass, not just a hidden pane.
- **Limits:** `preview_max_resolution` (default 1024 px, 0 = full), `preview_frames` (1 = still JPEG; >1 = animated WebP, or NVENC H.264 MP4 when PyAV probes NVENC), `preview_fps` (default 12).
- **Decoder precedence:** `preview_tiny_vae` combo widget (a `models/vae_approx` filename selector, e.g. `taeh3.safetensors` — the special H3 tiny decoder that core's `VAELoader` cannot build; rendered like a model selector, no input ring on the node edge) > `preview_vae` input socket (full-quality `vae.decode()`) > core latent previewer > latent-to-RGB fallback. The tiny-VAE path clamps output before uint8 because taeh3 output is not [0,1]-guaranteed. Implementation note: the widget must stay a **bare-list combo spec** — `(_vae_approx_options(), {"default": "none", ...})`, the same shape as core `VAELoader.vae_name`. The legacy `("STRING", {"combo": [...]})` tuple renders as a free-text field (no dropdown) in frontend v1.49.x. The hollow optional-input ring is removed **client-side**: the Director's JS (`install()` in `js/minimax_h3_director_v2.js`, run on node create + workflow load) sets the `preview_tiny_vae` socket's `shape = null` — a bare-list combo that is *optional* would otherwise draw a `HollowCircle` ring, while core's *required* `VAELoader.vae_name` draws none. A backend `socketless: True` option-dict key is ineffective on the legacy `INPUT_TYPES` path in v1.49.x (the frontend's `socketless` gate reads `widget.options`, which `addComboWidget` only ever populates with `{values, advanced, hidden}`), so the ring strip is the only reliable mechanism.
- **KJ alternative:** a KJ `ModelPreviewOverrideKJ` in the chain remains a valid external preview. Because KJ's preview is a native `OUTER_SAMPLE` model wrapper — not a DOM hook on the graph sampler — its own widget also animates during the Director's internal-execution runs. Canonical wiring is unchanged: `Settings.MODEL → KJ.model → KJ.MODEL → Director.ref2va_model`.

## Dense prompting guide

MiniMax H3 expects structured natural language rather than keyword piles. Use the official field names and shot/timestamp conventions.

### FL2VA prompt structure

Text-only (no endpoint images):
```text
integrated_multimodal_description: [Shot 1] ... [Shot 2] At 00:04.500, ...

overall_soundscape: ...

non_diegetic_music: ...
```

With endpoint images: prepend the alignment instruction line shown earlier, blank line, then the three fields above.

### REF2VA prompt structure

Use the six-section format. Define assets once and reuse labels consistently.

Label rules:
- `<Subject N>`: visible content abstracted from references (people, objects, scenes, clothing, actions). This is what actually appears.
- `<Picture N>`: ONLY for concrete frame anchors (opening/key/last frame, storyboard). If an image defines style/appearance only, cite it inside a Subject and don't create a Picture entry.
- `<Video N>`: structural roles only (editing source, continuation, pacing/cuts/rhythm). Specific subjects/actions/styles from a video belong under Subjects.
- `<Audio N>`: copied or referenced audio signals (dialogue, music, ambience, voice timbre).

Template:
```text
subject_definitions:
<Subject 1> is the woman in <Picture 1>, with short dark hair and a red coat.
<Picture 1> is the opening-frame anchor for [Shot 1].
<Subject 2> is the walking motion taken from <Video 1>.
<Video 1> provides the camera path and pacing structure.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] Use <Subject 1> from <Picture 1>, the motion and pacing of <Video 1>, and the voice character of <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity and clothing remain consistent.
<Picture 1> ([Shot 1] first frame): fully_preserved - opening composition anchor.
<Subject 2> (motion transferred to <Subject 1>): attribute_transfer - walk rhythm is applied to <Subject 1>.
<Video 1> (pacing structure): weak_reference - general timing and camera rhythm are retained.
<Audio 1>: reference - timbre and delivery are followed without copying the signal.

detailed_description: [Shot 1] ... [Shot 2] At 00:04.500, ...
overall_soundscape: ...
non_diegetic_music: ...
```

Summary task-type prefixes (combine with ` + `):
- `[keyframe completion]`, `[reference generation]`, `[video editing]`, `[video continuation]`, `[audio reuse]`, `[audio reference]`

Retention markers:
- Visual: `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`
- Audio: `fully_copy`, `partially_copy`, `reference`, `weak_reference`

Number labels by timeline order: images first (Pictures), then videos, then audio. Keep meanings stable everywhere.

### Shots and timestamps

- `[Shot 1]` starts without a timestamp. Later shots begin at cut times in `MM:SS.mmm`.
- Use timestamps for actual cuts or important transitions, not every sentence.
- Timestamps are relative to the generated output timeline; keep them within your duration and align with media timeline when relevant.
- Single continuous shot: one `[Shot 1]` block, describe temporal changes inline.
- Distinguish source vs target timing for references: `At 00:05.000 in the target video, reproduce the hand gesture seen near 00:02.400 in <Video 1>`.

### Camera motion vocabulary

Use as natural English inside shots:
- Types: Zoom/Push/Pan/Truck/Tilt/Pedestal/Arc/Tracking/Shake/POV/Roll
- Amplitude: `with small/large amplitude`
- Speed: `at slow/fast speed`

Example: `The camera pushes in with small amplitude at slow speed toward her hands.`

### Dialogue and special tokens

- Speaker IDs: `(S1)`, `(S2)` by vocal appearance order; reuse consistently.
- Exact dialogue: `<d>[Language] ...</d>` preserving original language/punctuation.
- Voiceover: say `in an off-screen voiceover` and specify lips remain closed after the tag.
- Cross-cut speech: `<scenetrans>` at both sides; `<cutoff>` when truncated.
- On-screen text: double-quote in English, preserve original characters exactly.

### Practical workflow

1. Choose FL2VA for endpoint/text work; REF2VA for multi-reference transfer.
2. Add only references that contribute specific identity, motion, layout, or sound.
3. Trim videos/audio to the strongest 2–15s segments; respect totals.
4. Use the mode-specific prompt builder:
   - FL2VA/I2VA/L2VA/T2VA: fill the three guided fields; alignment lines appear automatically.
   - REF2VA: click **Prefill Labels & Summary** to scaffold your labels, then edit subject definitions and detailed description. Use **Insert [Shot N]** for clean shot markers.
5. Click **Preview Prompt** to verify the exact output before queuing.
6. Verify duration, aspect ratio, and motion match your references; queue through the Guide.

Official MiniMax H3 guides (canonical conventions):
- [Video Prompt Writing Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [Full-Reference Rewrite Format Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
