import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEFAULT_BUILDER_STATE = mode => {
  if (mode === "REF2VA") {
    return { version: 2, mode: "REF2VA", duration: 5, ref: { subject_definitions: "", summary: "", retention_analysis: "", detailed_description: "", soundscape: "", music: "" } };
  }
  return { version: 1, mode: mode || "FL2VA", imd: "", soundscape: "", music: "", duration: 5, ref: { subject_defs: [], summary_types: ["reference generation"], summary_text: "", retention: [], style_line: "", detail: "", soundscape: "", music: "" } };
};
const DEFAULT_STATE = { version: 1, items: [], prompt_blocks: [], builder_state: null, resolution: null, seed_control: { mode: "random", last_seed: null, recent: [] } };
const MAX = { image: 9, video: 3, audio: 3, total: 12 };
// H3's VAE emits 16px latent cells and the diffusion transformer patchifies
// them in 2×2 groups, so both canvas edges must be divisible by 32.
const MINIMAX_MULTIPLE = 32;
const ASPECT_OPTIONS = [["auto", "Auto"], ["1:1", "1:1"], ["16:9", "16:9"], ["9:16", "9:16"], ["2:1", "2:1"], ["1:2", "1:2"], ["3:2", "3:2"], ["2:3", "2:3"], ["4:3", "4:3"], ["3:4", "3:4"], ["4:5", "4:5"], ["5:4", "5:4"], ["custom", "CUSTOM"]];
const RESOLUTION_PRESETS = {
  "144p": 0.0352, "240p": 0.0977, "360p": 0.22, "480p": 0.391, "540p": 0.494, "576p": 0.396,
  "720p": 0.879, "900p": 1.373, "1024p": 1.00, "1080p": 1.978, "1152p": 2.25, "1440p": 3.516,
  "2160p": 7.91, "2K": 3.906, "4K": 7.91,
  "0.26 MP - Preview": 0.26, "0.36 MP - Small": 0.36, "0.52 MP - SD": 0.52, "0.65 MP - Balanced": 0.65,
  "0.83 MP - HD": 0.83, "1.00 MP - 1024p": 1.00, "1.05 MP - HD+": 1.05, "1.20 MP - HD++": 1.20,
  "1.35 MP - 2K lite": 1.35, "1.55 MP - 2K": 1.55, "1.65 MP - 2K+": 1.65, "1.75 MP - QHD": 1.75,
  "2.10 MP - FHD": 2.10, "3.30 MP - QHD+": 3.30, "4.75 MP - 2K Pro": 4.75, "6.50 MP - Production": 6.50, "8.30 MP - UHD": 8.30,
};
const REPOSITORY_URL = "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/main/docs/minimax_h3_director.md";
const IMAGE_EXTENSIONS = new Set(["avif", "bmp", "gif", "heic", "heif", "jpeg", "jpg", "jxl", "png", "tif", "tiff", "webp"]);
const VIDEO_EXTENSIONS = new Set(["3gp", "avi", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts", "ts", "webm", "wmv"]);
const AUDIO_EXTENSIONS = new Set(["aac", "aif", "aiff", "alac", "amr", "ape", "caf", "flac", "m4a", "mka", "mp3", "oga", "ogg", "opus", "wav", "wave", "weba", "wma"]);
const AUTOMATIC_SAVE_METADATA_KEYS = new Set(["model_hash", "text_positive", "text_negative", "text_steps", "text_cfg", "text_sampler", "text_scheduler", "text_seed", "text_model"]);
const IMAGE_SAVE_OPTION_KEYS = new Set(["filename_prefix", "file_format", "compression"]);
const VIDEO_SAVE_OPTION_KEYS = new Set(["filename_prefix", "codec", "container", "bit_depth", "quality", "pingpong", "crop_to_audio", "audio_codec", "audio_bitrate", "save_first_frame", "save_last_frame"]);
// Pill / modal display names for each post-process stage (cleaner than the raw id).
const PP_STAGE_LABELS = {
  frame_interpolation: "Frame Interpolation",
  torch_resize: "Torch Resize",
  model_upscale: "Model Upscaler",
  rtx_refiner: "RTX Upscaler & Refiner",
  watermark: "Watermark",
};
const ppStageLabel = (id) => PP_STAGE_LABELS[id] || String(id || "").replaceAll("_", " ");
// Full option spec for each post-process stage, mirroring the real standalone nodes.
// The burger modal always renders every field below (filling missing keys with the
// default), so a freshly-saved stage still exposes the complete option set.
const PP_QUALITY_OPTIONS = ["Low", "Medium", "High", "Ultra"];
const PP_STAGE_FIELDS = {
  frame_interpolation: [
    { key: "factor", type: "number", min: 1, max: 4, step: 0.5, default: 2 },
    { key: "model", type: "modelFolder", folder: "frame_interpolation" },
  ],
  torch_resize: [
    { key: "size_mode", type: "combo", options: ["Multiplier", "Target resolution"], default: "Multiplier" },
    { key: "scale_multiplier", type: "number", min: 0.25, max: 8, step: 0.25, default: 2 },
    { key: "interpolation", type: "combo", options: ["Nearest", "Bilinear", "Bicubic", "Area", "Lanczos"], default: "Lanczos" },
  ],
  model_upscale: [
    { key: "model_name", type: "modelFolder", folder: "upscale_models" },
  ],
  rtx_refiner: [
    { key: "denoise", type: "boolean", default: false },
    { key: "denoise_quality", type: "combo", options: PP_QUALITY_OPTIONS, default: "Ultra" },
    { key: "deblur", type: "boolean", default: false },
    { key: "deblur_quality", type: "combo", options: PP_QUALITY_OPTIONS, default: "Ultra" },
    { key: "upscale", type: "combo", options: ["Off", "VSR", "High Bitrate"], default: "VSR" },
    { key: "upscale_quality", type: "combo", options: PP_QUALITY_OPTIONS, default: "Ultra" },
    { key: "resize_type", type: "combo", options: ["Keep Ratio", "Manual", "Preset Ratio", "Scale", "Same Size"], default: "Scale" },
    { key: "scale", type: "number", min: 1, max: 4, step: 0.05, default: 2.0 },
    { key: "megapixels", type: "number", min: 0.01, max: 64, step: 0.01, default: 2.0 },
    { key: "width", type: "number", min: 64, max: 8192, step: 8, default: 1920 },
    { key: "height", type: "number", min: 64, max: 8192, step: 8, default: 1080 },
    { key: "divisible_by", type: "combo", options: ["8", "16", "32", "64", "128"], default: "8" },
    { key: "ratio_preset", type: "combo", options: ["1:1", "4:3", "3:2", "16:9", "21:9"], default: "16:9" },
    { key: "resize_method", type: "combo", options: ["Center Crop (Fill)", "Letterbox (Fit)"], default: "Center Crop (Fill)" },
    { key: "device_id", type: "number", min: 0, max: 8, step: 1, default: 0 },
    { key: "empty_cache", type: "boolean", default: false },
    { key: "use_mmap", type: "boolean", default: false },
    { key: "auto_unload_models", type: "boolean", default: true },
  ],
  watermark: [
    { key: "watermark_path", type: "imagePicker" },
    { key: "position", type: "combo", options: ["bottom-right", "bottom-left", "top-right", "top-left", "center"], default: "bottom-right" },
  ],
};
// Saver option specs (image / video) for the save-node gear, same field-list format.
const PP_IMAGE_SAVE_FIELDS = [
  { key: "filename_prefix", type: "text", default: "Director" },
  { key: "file_format", type: "combo", options: ["webp", "png"], default: "webp" },
  { key: "compression", type: "number", min: 0, max: 100, step: 1, default: 15 },
];
const PP_VIDEO_SAVE_FIELDS = [
  { key: "filename_prefix", type: "text", default: "Director/video" },
  { key: "codec", type: "combo", options: ["Auto", "AV1", "VP9", "H.265 (HEVC)", "H.264"], default: "Auto" },
  { key: "container", type: "combo", options: ["Auto", "WebM", "MKV", "MP4", "Animated WebP", "Animated AVIF"], default: "Auto" },
  { key: "bit_depth", type: "combo", options: ["Auto", "8-bit", "10-bit"], default: "Auto" },
  { key: "quality", type: "number", min: 0, max: 51, step: 1, default: 20 },
  { key: "pingpong", type: "boolean", default: false },
  { key: "crop_to_audio", type: "boolean", default: false },
  { key: "audio_codec", type: "combo", options: ["Auto", "AAC", "Opus", "MP3"], default: "Auto" },
  { key: "audio_bitrate", type: "combo", options: ["64k", "96k", "128k", "160k", "192k", "256k", "320k"], default: "192k" },
  { key: "save_first_frame", type: "boolean", default: false },
  { key: "save_last_frame", type: "boolean", default: false },
];
// Tuning for the MiniMax H3 Cache optimization (opened from its pill), mirroring the
// standalone DaSiWa "MiniMax H3 Cache" node.
const PP_H3_CACHE_FIELDS = [
  { key: "reuse_threshold", type: "number", min: 0, max: 1, step: 0.01, default: 0.05 },
  { key: "start_percent", type: "number", min: 0, max: 1, step: 0.01, default: 0.15 },
  { key: "end_percent", type: "number", min: 0, max: 1, step: 0.01, default: 0.90 },
  { key: "max_steps", type: "number", min: 1, max: 10, step: 1, default: 2 },
  { key: "device", type: "combo", options: ["auto", "cuda", "cpu"], default: "auto" },
  { key: "verbose", type: "boolean", default: false },
];
// MiniMax H3 sampling contract (comfy.samplers KSamplerSelect/BasicScheduler names).
const SAMPLER_OPTIONS = ["euler", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2s_ancestral", "dpmpp_sde", "lms", "dpm_2", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "uni_pc"];
const SCHEDULER_OPTIONS = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"];
// Built-in live step-preview settings (rendered inside the ☰ "Preview & Output options" modal).
const PREVIEW_OPTION_FIELDS = [
  { key: "live_step_preview", type: "boolean", default: true },
  { key: "preview_max_resolution", type: "number", default: 1024, min: 0, max: 8192 },
  { key: "preview_frames", type: "number", default: 1, min: 1, max: 32 },
  { key: "preview_fps", type: "number", default: 12, min: 1, max: 60 },
];
// Fetch a ComfyUI model-folder listing (e.g. "upscale_models", "frame_interpolation").
async function ppLoadModelList(folder) {
  try {
    const res = await fetch(api.apiURL(`/models/${encodeURIComponent(folder)}`));
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data) ? data.map(String) : [];
  } catch { return []; }
}
// Fetch the ComfyUI input-folder image listing (the DaSiWa route registered by
// nodes/input_images.py). Used to feed the watermark picker with real images.
async function ppLoadInputImages() {
  try {
    const res = await fetch(api.apiURL("/dasiwa/input-images"));
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.images) ? data.images.map(String) : [];
  } catch { return []; }
}
// Build a "select from the ComfyUI input folder" control: a dropdown fed by the
// input-folder listing, a live thumbnail preview of the chosen image, and an
// upload button that drops a new image into the input folder and selects it.
// Sets target[key] as the user interacts; returns immediately (list loads async).
function ppImagePicker(field, key, target, inputStyle, currentValue) {
  const box = document.createElement("div"); box.style.cssText = "display:flex;flex-direction:column;gap:5px";
  const row = document.createElement("div"); row.style.cssText = "display:flex;gap:6px;align-items:center";
  const select = document.createElement("select"); select.style.cssText = inputStyle + ";flex:1;min-width:0"; select.disabled = true;
  const ph = document.createElement("option"); ph.value = "(loading…)"; ph.textContent = "(loading…)"; ph.selected = true; select.append(ph);
  const uploadBtn = document.createElement("button"); uploadBtn.textContent = "Upload"; uploadBtn.title = "Upload a watermark image into the ComfyUI input folder";
  uploadBtn.style.cssText = "padding:3px 9px;background:rgba(74,144,217,.15);border:1px solid rgba(74,144,217,.4);border-radius:4px;color:#a8d4ff;cursor:pointer;font-size:11px;white-space:nowrap";
  row.append(select, uploadBtn);
  const preview = document.createElement("div");
  const renderPreview = (path) => {
    preview.innerHTML = ""; if (!path) return;
    const img = document.createElement("img"); img.src = viewUrl(path); img.style.cssText = "max-width:140px;max-height:64px;object-fit:contain;border:1px solid #2f5478;border-radius:4px;background:#080d11;padding:2px";
    const cap = document.createElement("span"); cap.textContent = path; cap.style.cssText = "display:block;font-size:9px;color:#6fa8e0;margin-top:2px";
    preview.append(img, cap);
  };
  const populate = (images) => {
    select.innerHTML = "";
    const opts = images.length ? images : ["(no images in input folder)"];
    for (const opt of opts) { const o = document.createElement("option"); o.value = opt; o.textContent = opt; if (opt === currentValue) o.selected = true; select.append(o); }
    if (currentValue && !images.includes(currentValue)) { const o = document.createElement("option"); o.value = currentValue; o.textContent = `${currentValue} (current)`; o.selected = true; select.append(o); }
    select.disabled = false;
    select.onchange = () => { target[key] = select.value; renderPreview(select.value); };
    renderPreview(select.value);
  };
  const refresh = () => ppLoadInputImages().then(populate).catch(() => populate([]));
  const fileInput = document.createElement("input"); fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
  uploadBtn.onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    const file = fileInput.files[0]; fileInput.value = ""; if (!file) return;
    uploadBtn.disabled = true;
    try {
      const rel = await uploadFile(file, uploadBtn);
      await refresh();
      if (Array.from(select.options).some(o => o.value === rel)) select.value = rel;
      target[key] = rel; renderPreview(rel);
      uploadBtn.textContent = "Upload";
    } catch {
      uploadBtn.textContent = "Failed";
    } finally {
      uploadBtn.disabled = false;
    }
  };
  box.append(row, preview, fileInput);
  field.append(box);
  void refresh();
}
let cssInstalled = false;

function installStyles() {
  if (cssInstalled) return;
  cssInstalled = true;
  const style = document.createElement("style");
  style.textContent = `
    .ds-h3{box-sizing:border-box;width:100%;min-width:0;min-height:850px;align-self:stretch;background:transparent;border:0;border-radius:0;padding:0;font:12px system-ui,sans-serif;display:flex;flex-direction:column;gap:6px;overflow:visible}
    .ds-h3 button{background:#202b35;color:#dbe7f0;border:1px solid #40515e;border-radius:4px;padding:4px 7px;cursor:pointer}.ds-h3 button:hover{background:#2c3c49}.ds-h3-lane-add{position:absolute;right:6px;z-index:3;width:22px;height:22px;padding:0!important;border-radius:50%!important;font-size:17px;line-height:18px;background:rgba(70,150,105,.3)!important;border-color:rgba(126,210,157,.75)!important;color:#bff3d0!important}
    .ds-h3-actions{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.ds-h3-modebar{display:flex;gap:4px;padding:4px;background:#0d1217;border:1px solid #344452;border-radius:6px}.ds-h3-modebar button{padding:4px 9px!important;border-radius:999px!important;background:transparent!important;color:#9fb3c2!important}.ds-h3-modebar button:hover{background:rgba(126,235,167,.10)!important;box-shadow:0 0 10px rgba(126,235,167,.5)}.ds-h3-modebar button.active{color:#efe6ff!important;border-color:rgba(177,128,255,.8)!important;box-shadow:0 0 10px rgba(151,91,255,.6);font-weight:700}.ds-h3-clear-btn,.ds-h3-remove-btn{padding:3px 7px!important;font-size:11px;border-radius:999px!important}.ds-h3-modebar .ds-h3-clear-btn{background:rgba(255,100,100,.08)!important;color:#ffb0b0!important;border:1px solid rgba(255,100,100,.35)!important}.ds-h3-modebar .ds-h3-remove-btn{background:rgba(255,150,60,.08)!important;color:#ffcfab!important;border:1px solid rgba(255,150,60,.3)!important}.ds-h3-modebar .ds-h3-clear-btn:hover{background:rgba(255,100,100,.35)!important;color:#ffe2e2!important;border-color:rgba(255,100,100,.95)!important;box-shadow:0 0 14px rgba(255,100,100,.75)}.ds-h3-modebar .ds-h3-remove-btn:hover{background:rgba(255,150,60,.35)!important;color:#ffeadb!important;border-color:rgba(255,150,60,.95)!important;box-shadow:0 0 14px rgba(255,150,60,.75)}.ds-h3-modebar .ds-h3-clear-btn-empty{opacity:.45}.ds-h3-modebar .ds-h3-clear-btn-empty:hover{opacity:1}.ds-h3-prompt{width:100%;min-height:88px;box-sizing:border-box;background:#0d1217;color:#e5eef4;border:1px solid #40515e;border-radius:4px;padding:7px;resize:vertical}.ds-h3-prompt-panel{width:100%;box-sizing:border-box;border:0;border-radius:0;padding:0;display:flex;flex-direction:column;gap:6px;background:transparent}.ds-h3-status{min-height:16px;color:#f3c67a}.ds-h3-info-field{box-sizing:border-box;min-height:28px;border:1px solid #40515e;border-radius:4px;padding:6px 7px;background:#0d1217}.ds-h3-status.error{color:#ff6f6f;font-weight:700}.ds-h3-small{font-size:11px;color:#9fb3c2}.ds-h3-ruler{position:relative;height:19px;color:#8fa3b2;font-size:10px;white-space:nowrap;overflow:hidden}.ds-h3-ruler span{position:absolute;top:1px;border-left:1px solid #587084;padding-left:2px;height:16px}.ds-h3-track{position:relative;min-height:280px;max-width:100%;overflow-x:auto;overflow-y:hidden;background:#0b1015;border:1px solid #344452;border-radius:5px;padding:7px 6px 6px}.ds-h3-track::before{content:none}.ds-h3-track-inner{position:relative;min-width:100%;height:204px;overflow:visible;background:repeating-linear-gradient(90deg,#111a21 0,#111a21 49px,#1b2933 50px)}.ds-h3-track-inner::after{content:'';position:absolute;left:var(--insert-x,-8px);top:0;height:204px;border-left:2px solid #f3c67a;pointer-events:none}.ds-h3-track-inner.over{outline:2px solid #8dd7ff;outline-offset:-2px}.ds-h3-timeline-lane{position:absolute;left:0;right:0;height:120px;box-sizing:border-box;border-bottom:1px solid #344452;cursor:pointer}.ds-h3-empty-slot{position:absolute;top:21px;height:88px;box-sizing:border-box;border:1px dashed #587084;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#7890a0;font-size:10px;pointer-events:none}.ds-h3-timeline-lane.disabled .ds-h3-empty-slot{display:none}.ds-h3-timeline-lane.visual{top:0;background:rgba(17,30,39,.72)}.ds-h3-timeline-lane.audio{top:120px;background:rgba(22,49,36,.55)}.ds-h3-timeline-lane.selected{box-shadow:inset 0 0 0 2px #8dd7ff}.ds-h3-timeline-lane.disabled{background:rgba(51,55,60,.72);filter:grayscale(1);cursor:not-allowed}.ds-h3-timeline-lane.disabled::after{content:"Not supported by the selected mode";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#a1a8ad;font-size:11px;font-weight:600;background:rgba(0,0,0,.35);pointer-events:none}.ds-h3-lane-label{position:absolute;left:5px;top:2px;color:#8fa3b2;font-size:10px;text-transform:uppercase;pointer-events:none;z-index:1}.ds-h3-grip{position:absolute;top:0;width:11px;height:100%;cursor:ew-resize;background:rgba(255,255,255,.22);z-index:4}.ds-h3-grip.left{left:0;border-right:1px solid rgba(255,255,255,.65)}.ds-h3-grip.right{right:0;border-left:1px solid rgba(255,255,255,.65)}.ds-h3-clip{position:absolute;top:18px;height:48px;min-width:64px;box-sizing:border-box;border:1px solid #73c7ef;border-radius:4px;background:#1b4558;color:#e5eef4;padding:6px 14px 19px;cursor:grab;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ds-h3-clip.image,.ds-h3-clip.video{min-width:112px!important;width:112px!important;height:112px;top:7px}.ds-h3-clip.audio{border-color:#7ecf9d;background:#254b38;top:7px;height:112px}.ds-h3-waveform{position:absolute;inset:22px 12px 55px;width:calc(100% - 24px);height:calc(100% - 77px);pointer-events:none;opacity:.9}.ds-h3-crop-marker,.ds-h3-audio-crop-marker{position:absolute;top:20px;bottom:18px;width:4px;background:#fff;box-shadow:0 0 4px #000;cursor:ew-resize;z-index:6}.ds-h3-crop-marker.start,.ds-h3-audio-crop-marker.start{background:#f3c67a}.ds-h3-crop-marker.end,.ds-h3-audio-crop-marker.end{background:#8dd7ff;transform:translateX(-4px)}.ds-h3-crop-readout{position:absolute;left:14px;right:14px;bottom:3px;font-size:10px;line-height:12px;color:#d9f5e2;background:rgba(0,0,0,.36);pointer-events:none;text-align:center;overflow:hidden;white-space:nowrap}.ds-h3-clip-close{position:absolute!important;right:2px;top:2px;width:18px;height:18px;padding:0!important;line-height:15px!important;font-size:16px;color:#fff!important;background:rgba(105,28,28,.9)!important;border-color:#f08080!important;z-index:5}.ds-h3-clip.video{border-color:#b887d8;background:#432e52}.ds-h3-clip.text{border-color:#83c98a;background:#27442d}
  `;
  style.textContent += `.ds-h3-preview-overlay{position:fixed;inset:0;z-index:10001;display:flex;align-items:center;justify-content:center;background:rgba(8,10,14,.6)}.ds-h3-preview-panel{width:min(400px,90vw);max-height:85vh;display:flex;flex-direction:column;background:#111820;border:1px solid #40515e;border-radius:10px;overflow:hidden;box-shadow:0 8px 32px #000}.ds-h3-preview-header,.ds-h3-preview-meta{padding:8px 12px;background:#0d1217;color:#dbe7f0}.ds-h3-preview-header{display:flex;justify-content:space-between;border-bottom:1px solid #344452}.ds-h3-preview-body{padding:12px;background:#090d11;display:flex;justify-content:center}.ds-h3-preview-media{max-width:100%;max-height:40vh;object-fit:contain}.ds-h3-preview-controls{padding:8px 12px;background:#0d1217;display:flex;flex-direction:column;gap:6px}.ds-h3-preview-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:11px;color:#9fb3c2}.ds-h3-preview-row input[type="number"]{width:60px;padding:2px 4px;background:#111a21;color:#dbe7f0;border:1px solid #40515e;border-radius:3px}.ds-h3-preview-row input[type="text"],.ds-h3-preview-row textarea{flex:1;min-width:150px;padding:3px 5px;background:#111a21;color:#dbe7f0;border:1px solid #40515e;border-radius:3px;font-size:11px}.ds-h3-preview-row textarea{resize:vertical;min-height:30px}.ds-h3-preview-meta{font-size:11px;color:#9fb3c2;border-top:1px solid #344452}`;
  style.textContent += `
    .ds-h3-docs{font-weight:700;min-width:26px;padding:4px 8px!important}.ds-h3-ruler{display:none}.ds-h3-clip.video{width:var(--clip-width)!important;min-width:180px!important}.ds-h3-clip-identity{position:absolute;left:5px;top:4px;z-index:5;padding:1px 4px;border-radius:3px;background:rgba(0,0,0,.65);color:#fff;font-size:10px;font-weight:700;pointer-events:none}.ds-h3-video-scale{position:absolute;left:12px;right:12px;top:27px;height:44px;pointer-events:none;opacity:.8;background:repeating-linear-gradient(90deg,rgba(216,174,245,.82) 0,rgba(216,174,245,.82) 1px,transparent 1px,transparent 12px),linear-gradient(transparent 48%,rgba(216,174,245,.7) 49%,rgba(216,174,245,.7) 52%,transparent 53%)}.ds-h3-prompt-panel{min-height:120px;overflow:visible}.ds-h3-prompt-field{position:relative;flex:none;min-height:90px}.ds-h3-prompt-panel .ds-h3-prompt-field>.ds-h3-prompt{height:100%;min-height:0;padding-bottom:16px;resize:none}.ds-h3-prompt-field-resizer{position:absolute;bottom:0;left:0;width:100%;height:12px;cursor:ns-resize;display:flex;justify-content:center;align-items:flex-end;padding-bottom:4px;box-sizing:border-box;z-index:2;touch-action:none}.ds-h3-prompt-field-resizer::after{content:"";width:40px;height:4px;background:rgba(255,255,255,.16);border-radius:2px}.ds-h3-prompt-field-resizer:hover::after,.ds-h3-prompt-field-resizer.active::after{background:rgba(141,215,255,.8)}
    .ds-h3-video-stream-controls{position:absolute;right:5px;top:4px;z-index:7;display:flex;gap:3px}.ds-h3-clip.selected .ds-h3-video-stream-controls{right:24px}.ds-h3-video-stream-controls button{min-width:24px;padding:3px 6px!important;font-size:11px;line-height:14px;background:rgba(10,17,23,.85)!important}.ds-h3-video-stream-controls button.active{background:rgba(125,82,188,.9)!important;border-color:#d4b3ff;color:#fff}.ds-h3-lock-icon{position:absolute;right:4px;top:4px;z-index:8;font-size:13px;color:#f3c67a;text-shadow:0 0 4px rgba(0,0,0,.9);pointer-events:none}.ds-h3-edit-btn{position:absolute;right:5px;bottom:5px;z-index:7;font-size:14px;padding:3px 6px!important;border-radius:3px!important;background:rgba(20,35,45,.8)!important;border-color:rgba(100,150,180,.5)!important}.ds-h3-empty-slot{display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#7eeba7;border:1.5px dashed #7eeba7;border-radius:6px;cursor:pointer;box-sizing:border-box;background:rgba(126,235,167,.06);transition:all .15s ease;pointer-events:auto}.ds-h3-empty-slot:hover{background:rgba(126,235,167,.18);border-color:#bff3d0;color:#d4ffe1;box-shadow:0 0 14px rgba(126,235,167,.45);transform:scale(1.02)}.ds-h3-timeline-lane.disabled .ds-h3-empty-slot{opacity:.15;cursor:not-allowed;pointer-events:none}.ds-h3-timeline-lane.audio .ds-h3-empty-slot{color:#5b8dd9;border-color:#5b8dd9;background:rgba(91,141,217,.06)}.ds-h3-timeline-lane.audio .ds-h3-empty-slot:hover{background:rgba(91,141,217,.18);border-color:#8bb4f0;color:#b3d4fc;box-shadow:0 0 14px rgba(91,141,217,.45)}.ds-h3-prompt-panel.disabled{opacity:.5;filter:grayscale(.55)}.ds-h3-prompt-panel.disabled textarea,.ds-h3-prompt-panel.disabled input,.ds-h3-prompt-panel.disabled button{pointer-events:none}.ds-h3-ext-note{font-weight:600;color:#7ec8f0}.ds-h3-prompt-mode-btn{white-space:nowrap}.ds-h3-pp-btn{display:inline-flex;align-items:center;gap:5px;padding:4px 10px!important;border-radius:999px!important;background:transparent!important;color:#9fb3c2!important;font-size:11px!important;font-family:inherit;border:1px solid #40515e!important;cursor:pointer;white-space:nowrap;transition:all .16s ease}.ds-h3-pp-btn:hover{background:rgba(74,144,217,.12)!important;border-color:rgba(74,144,217,.5)!important}.ds-h3-pp-btn.active{background:rgba(74,144,217,.18)!important;color:#a8d4ff!important;border-color:rgba(74,144,217,.85)!important;box-shadow:0 0 12px rgba(74,144,217,.5);font-weight:700}.ds-h3-pp-summary{color:#6fa8e0;font-size:10px!important;font-weight:400!important}.ds-h3-pp-burger{display:inline-flex;align-items:center;justify-content:center;min-width:16px;min-height:16px;padding:0 4px!important;font-size:9px!important;background:rgba(74,144,217,.22)!important;border:1px solid rgba(74,144,217,.45)!important;border-radius:3px!important;color:#cfe6ff!important;cursor:pointer;white-space:nowrap;margin-left:2px}.ds-h3-pp-btn.active .ds-h3-pp-burger{background:rgba(74,144,217,.35)!important;border-color:rgba(74,144,217,.7)!important}.ds-h3-modebar .ds-h3-prompt-mode-btn.active{background:rgba(126,235,167,.14)!important;color:#d7ffe3!important;border-color:rgba(126,235,167,.9)!important;box-shadow:0 0 8px rgba(126,235,167,.45);font-weight:700}.ds-h3-global-prompt{min-height:140px}
  `;
  style.textContent += `.ds-h3-res-field{display:flex;flex-direction:column;gap:3px;min-width:150px;font-size:10px;font-weight:600;letter-spacing:.4px;color:#8fb3d6;text-transform:uppercase}.ds-h3-res-control{display:flex;position:relative}.ds-h3-res-select{position:absolute;inset:0;width:100%;opacity:0;pointer-events:none}.ds-h3-res-btn{width:100%;display:flex;align-items:center;gap:8px;box-sizing:border-box;min-height:30px;padding:0 9px;background:#16283a;border:1px solid #2f5478;border-radius:5px;color:#d6ebff;font:12px system-ui,sans-serif;cursor:pointer;text-align:left;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}.ds-h3-res-btn:hover:not(:disabled){background:#1d3550;border-color:#3f79b4;box-shadow:0 0 9px rgba(74,144,217,.28)}.ds-h3-res-btn:focus-visible{outline:none;border-color:#4f97d6;box-shadow:0 0 0 2px rgba(74,144,217,.35)}.ds-h3-res-btn:disabled{opacity:.4;cursor:not-allowed}.ds-h3-res-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ds-h3-res-caret{flex:none;color:#6fa8e0;font-size:9px;transition:transform .18s ease}.ds-h3-res-btn[aria-expanded="true"] .ds-h3-res-caret{transform:rotate(180deg)}.ds-h3-res-swatch{flex:none;display:flex;align-items:center;justify-content:center;width:20px;height:20px}.ds-h3-res-swatch-box{background:#3f79b4;border:1px solid #9fd0ff;border-radius:1px;box-shadow:inset 0 0 4px rgba(0,0,0,.4);transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}.ds-h3-res-btn:hover:not(:disabled) .ds-h3-res-swatch-box,.ds-h3-res-btn[aria-expanded="true"] .ds-h3-res-swatch-box{background:#5b9be0;border-color:#c4e4ff}.ds-h3-res-menu{position:absolute;top:calc(100% + 4px);left:0;z-index:2500;min-width:100%;max-height:290px;overflow-y:auto;padding:4px;background:#101c28;border:1px solid #35618f;border-radius:6px;box-shadow:0 10px 30px rgba(0,0,0,.65);display:none}.ds-h3-res-menu.open{display:block}.ds-h3-res-menu.grid.open{display:grid;gap:2px}.ds-h3-res-menu.grid .ds-h3-res-item-label{white-space:normal;overflow-wrap:anywhere}.ds-h3-res-menu[data-place="up"]{top:auto;bottom:calc(100% + 4px)}.ds-h3-res-item{display:flex;align-items:center;gap:8px;width:100%;box-sizing:border-box;padding:6px 9px;background:transparent;border:0;border-radius:4px;color:#cfe3f7;font:12px system-ui,sans-serif;cursor:pointer;text-align:left;transition:background .12s ease,color .12s ease,box-shadow .12s ease}.ds-h3-res-item:hover{background:rgba(74,144,217,.24);color:#fff;box-shadow:inset 0 0 0 1px rgba(96,168,232,.35)}.ds-h3-res-item.active{background:rgba(74,144,217,.34);color:#fff;font-weight:600;box-shadow:inset 0 0 0 1px rgba(120,190,255,.55)}.ds-h3-res-item.active:hover{background:rgba(74,144,217,.44)}.ds-h3-res-item-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ds-h3-res-num{width:100%;box-sizing:border-box;height:30px;padding:0 8px;background:#16283a;border:1px solid #2f5478;border-radius:5px;color:#d6ebff;font:12px system-ui,sans-serif;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}.ds-h3-res-num:hover:not(:disabled){background:#1d3550;border-color:#3f79b4}.ds-h3-res-num:focus{outline:none;border-color:#4f97d6;box-shadow:0 0 0 2px rgba(74,144,217,.3)}.ds-h3-res-num:disabled{opacity:.4;cursor:not-allowed}.ds-h3-res-menu.cols.open{display:flex;align-items:flex-start;gap:7px}.ds-h3-res-col{display:flex;flex-direction:column;gap:2px;min-width:84px}.ds-h3-res-col-title{font-size:9px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:#6fa8e0;padding:1px 9px 3px;border-bottom:1px solid #2f5478;margin-bottom:2px}.ds-h3-res-menu.cols .ds-h3-res-item{white-space:nowrap}`;
  document.head.appendChild(style);
}

// Built-in live step preview: the backend OUTER_SAMPLE wrapper ships per-step
// frames to the Director's own Preview & Output panel. Route each event to the
// matching node's live slot (node_id === ComfyUI node id === unique_id).
api.addEventListener("dasiwa_director_v2_preview", e => {
  const d = e.detail || e.data || {};
  const n = app.graph?._nodes?.find(x => String(x.id) === String(d.node_id));
  n?.__dasiwaH3ShowStepPreview?.({ src: `data:${d.mime};base64,${d.image}`, mime: d.mime, w: d.w, h: d.h, step: d.step, total: d.total, fps: d.fps ?? 12, frames: d.frames });
});

function parseState(value) { try { const s = JSON.parse(value || "{}"); return { ...DEFAULT_STATE, ...s, items: Array.isArray(s.items) ? s.items : [], prompt_blocks: Array.isArray(s.prompt_blocks) ? s.prompt_blocks : [] }; } catch { return structuredClone(DEFAULT_STATE); } }
function textValue(value) { return typeof value === "string" ? value.trim() : ""; }
function builderHasContent(builder) {
  if (!builder || typeof builder !== "object") return false;
  if (textValue(builder.simple_prompt) || textValue(builder.imd) || textValue(builder.soundscape)) return true;
  const ref = builder.ref;
  return !!(ref && typeof ref === "object" && (
    ["subject_definitions", "summary", "retention_analysis", "detailed_description", "soundscape"].some(key => textValue(ref[key])) ||
    (Array.isArray(ref.subject_defs) && ref.subject_defs.length) || (Array.isArray(ref.retention) && ref.retention.length)
  ));
}
function legacyPrompt(state, widgetPrompt) {
  for (const candidate of [state?.resolved_prompt, state?.full_prompt]) if (textValue(candidate)) return textValue(candidate);
  const global = textValue(widgetPrompt) || textValue(state?.prompt);
  const blocks = Array.isArray(state?.prompt_blocks) ? [...state.prompt_blocks].sort((a, b) => (Number(a?.start) || 0) - (Number(b?.start) || 0) || (Number(a?.order) || 0) - (Number(b?.order) || 0)) : [];
  return [global, ...blocks.filter(block => block?.enabled !== false).map(block => textValue(block?.text)).filter(Boolean)].filter(Boolean).join("\n");
}
function viewUrl(path) { return api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`); }
function savedViewUrl(entry) {
  const params = new URLSearchParams({ filename: entry.filename, subfolder: entry.subfolder || "", type: entry.type || "output" });
  return api.apiURL(`/view?${params}`);
}
function count(state, type) { return state.items.filter(i => i.enabled !== false && i.type === type).length; }
function idFor(type, n) { return `${type}-${Date.now()}-${n}`; }
function mediaTypeFor(file) {
  const mimeType = String(file.type || "").toLowerCase();
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("video/")) return "video";
  if (mimeType.startsWith("audio/")) return "audio";
  const extension = String(file.name || "").split(".").pop().toLowerCase();
  if (IMAGE_EXTENSIONS.has(extension)) return "image";
  if (VIDEO_EXTENSIONS.has(extension)) return "video";
  return AUDIO_EXTENSIONS.has(extension) ? "audio" : null;
}
function wavDurationFromBuffer(buffer) {
  const data = new DataView(buffer);
  if (data.byteLength < 12 || data.getUint32(0, false) !== 0x52494646 || data.getUint32(8, false) !== 0x57415645) return null;
  let byteRate = null;
  let dataSize = null;
  for (let offset = 12; offset + 8 <= data.byteLength;) {
    const chunk = data.getUint32(offset, false);
    const size = data.getUint32(offset + 4, true);
    const contentOffset = offset + 8;
    if (contentOffset + size > data.byteLength) return null;
    if (chunk === 0x666d7420 && size >= 16) byteRate = data.getUint32(contentOffset + 8, true);
    if (chunk === 0x64617461) dataSize = size;
    offset = contentOffset + size + (size & 1);
  }
  return byteRate && dataSize !== null ? dataSize / byteRate : null;
}
function mediaLabel(item) {
  const name = String(item.value || "media").split("/").pop();
  const prompt = String(item.prompt || "").trim().replace(/\s+/g, " ");
  const promptPreview = prompt ? ` · “${prompt.slice(0, 42)}${prompt.length > 42 ? "…" : ""}”` : "";
  if (item.type === "image") return `${name}${promptPreview}`;
  const duration = Number(item.duration);
  const durationText = Number.isFinite(duration) ? ` · ${duration.toFixed(2)}s` : "";
  const trim = (item.type === "video" || item.type === "audio") && (Number(item.trim_start) > 0 || item.trim_end != null)
    ? ` · crop L ${Number(item.trim_start || 0).toFixed(2)}s / R ${item.trim_end == null ? "end" : `${Number(item.trim_end).toFixed(2)}s`}` : "";
  return `${name}${durationText}${trim}${promptPreview}`;
}

const mediaReferenceName = type => type === "image" ? "Picture" : type === "video" ? "Video" : type === "audio" ? "Audio" : "Media";

async function uploadFile(file, status) {
  const form = new FormData();
  form.append("image", file, file.name);
  form.append("type", "input");
  form.append("overwrite", "false");
  status.textContent = `Uploading ${file.name}…`;
  const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
  if (!response.ok) throw new Error(`upload failed (${response.status})`);
  const result = await response.json();
  return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart || 0;
  const end = textarea.selectionEnd || 0;
  textarea.value = textarea.value.slice(0, start) + text + textarea.value.slice(end);
  textarea.selectionStart = textarea.selectionEnd = start + text.length;
  textarea.focus();
  textarea.dispatchEvent(new Event("change"));
}

function createBuilderField(label, value, opts = {}, fieldHeights = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "ds-h3-prompt-field";
  wrapper.style.display = "flex"; wrapper.style.flexDirection = "column"; wrapper.style.gap = "2px"; wrapper.style.position = "relative";
  const key = opts.fieldKey || "";
  if (opts.dataAttr) wrapper.dataset.ref2vaTarget = opts.dataAttr;
  if (label) { const lbl = document.createElement("div"); lbl.className = "ds-h3-small"; lbl.textContent = label; wrapper.appendChild(lbl); }
  const area = document.createElement(opts.tag || "textarea");
  area.className = opts.class || "ds-h3-prompt";
  area.rows = opts.rows || 3;
  area.placeholder = opts.placeholder || "";
  area.value = value || "";
  if (opts.onChange) area.onchange = e => opts.onChange(e.target.value);
  wrapper.appendChild(area);
  const resizer = document.createElement("div");
  resizer.className = "ds-h3-prompt-field-resizer";
  let dragging = false, startY = 0, startH = 0;
  resizer.addEventListener("pointerdown", ev => { dragging = true; startY = ev.clientY; startH = area.clientHeight; area.setPointerCapture(ev.pointerId); });
  window.addEventListener("pointermove", ev => { if (!dragging) return; const height = Math.max(60, startH + (ev.clientY - startY)); area.style.height = `${height}px`; if (key) fieldHeights[key] = height; });
  window.addEventListener("pointerup", () => { dragging = false; });
  wrapper.appendChild(resizer);
  if (key) {
    const saved = Number(fieldHeights[key]);
    if (Number.isFinite(saved) && saved >= 60) area.style.height = `${saved}px`;
  }
  return wrapper;
}

function install(node) {
  if (node.__dasiwaH3Installed) return;
  node.__dasiwaH3Installed = true;
  installStyles();
  if (!window.__dasiwaH3MenuCloseInstalled) { window.__dasiwaH3MenuCloseInstalled = true; window.addEventListener("pointerdown", event => { const open = document.querySelector(".ds-h3-res-menu.open"); if (open && !open.parentElement.contains(event.target)) open.classList.remove("open"); }, true); }
  // preview_tiny_vae ships as a bare-list combo (the VAELoader look). As an
  // optional socket LiteGraph would still draw a hollow ring on the node edge
  // (core's required VAELoader.vae_name gets none). Null the socket shape so
  // the combo reads as a plain model selector; the socket stays present and
  // connectable, only the ring is gone. Runs on nodeCreated + loadedGraphNode.
  for (const input of node.inputs || []) if (input.name === "preview_tiny_vae") { input.shape = null; break; }
  const dataWidget = node.widgets?.find(w => w.name === "timeline_data");
  if (!dataWidget) return;
  dataWidget.hidden = true; dataWidget.options = { ...(dataWidget.options || {}), hidden: true };
  dataWidget.draw = () => {}; dataWidget.computeSize = () => [0, -4];
  let state = parseState(dataWidget.value);
  // Resized prompt text-box heights, kept per node so a re-render (media add,
  // mode/resolution change, workflow reload) restores the user's layout.
  const fieldHeights = {};
  for (const [key, value] of Object.entries(state.field_heights || {})) {
    const height = Number(value);
    if (Number.isFinite(height) && height >= 60) fieldHeights[key] = height;
  }
  const DEFAULT_POSTPROCESS_RECIPE = [{ id: "frame_interpolation", enabled: false, factor: 2, model: "rife_v4.26.safetensors" }, { id: "torch_resize", enabled: false, size_mode: "Multiplier", scale_multiplier: 2, interpolation: "Lanczos" }, { id: "model_upscale", enabled: false, model_name: "2x-AnimeSharpV4_RCAN.safetensors" }, { id: "rtx_refiner", enabled: false, upscale: "VSR", upscale_quality: "Ultra" }, { id: "watermark", enabled: false, watermark_path: "", position: "bottom-right" }];
  state.postprocess_recipe = Array.isArray(state.postprocess_recipe) ? state.postprocess_recipe : DEFAULT_POSTPROCESS_RECIPE;
  state.seed_control = { mode: "random", last_seed: null, recent: [], ...(state.seed_control && typeof state.seed_control === "object" ? state.seed_control : {}) };
  state.seed_control.mode = state.seed_control.mode === "fixed" ? "fixed" : "random";
  state.seed_control.recent = Array.isArray(state.seed_control.recent) ? state.seed_control.recent.map(String).filter(value => /^\d+$/.test(value)).slice(0, 10) : [];
  state.internal_execution = { sampler: "res_multistep", scheduler: "simple", steps: 25, shift_video: 11, shift_audio: 4, comfy_kitchen_attention: false, cache: { enabled: false, reuse_threshold: 0.05, start_percent: 0.15, end_percent: 0.90, max_steps: 2, device: "auto", verbose: false }, ...(state.internal_execution && typeof state.internal_execution === "object" ? state.internal_execution : {}) };
  state.internal_execution.cache = { enabled: false, reuse_threshold: 0.05, start_percent: 0.15, end_percent: 0.90, max_steps: 2, device: "auto", verbose: false, ...(state.internal_execution.cache && typeof state.internal_execution.cache === "object" ? state.internal_execution.cache : {}) };
  state.internal_execution.save = { output_kind: "image", filename_prefix: "DaSiWa_MiniMaxH3_%date:yyyyMMdd%_%seed%", file_format: "webp", compression: 15, save_output: true, save_workflow: true, codec: "Auto", container: "Auto", bit_depth: "Auto", quality: 20, pingpong: false, crop_to_audio: false, audio_codec: "Auto", audio_bitrate: "192k", save_first_frame: false, save_last_frame: false, model_hash: "", text_positive: "", text_negative: "", text_steps: 0, text_cfg: 0, text_sampler: "", text_scheduler: "", text_seed: 0, text_model: "", ...(state.internal_execution.save || {}) };
  let selectedModeOverride = null;
  const mode = () => selectedModeOverride ?? (node.widgets?.find(w => w.name === "mode")?.value || "FL2VA");
  let builderState = state.builder_state && typeof state.builder_state === "object" ? { ...DEFAULT_BUILDER_STATE(mode()), ...state.builder_state, ref: { ...DEFAULT_BUILDER_STATE(mode()).ref, ...(state.builder_state.ref || {}) } } : DEFAULT_BUILDER_STATE(mode());
  builderState.mode = mode();
  const isNaSentinel = v => typeof v === "string" && v.trim() === "N/A";
  if (isNaSentinel(builderState.music)) builderState.music = "";
  if (builderState.ref && isNaSentinel(builderState.ref.music)) builderState.ref.music = "";
  const builderWidget = node.widgets?.find(w => w.name === "builder_state");
  if (builderWidget) { builderWidget.hidden = true; builderWidget.draw = () => {}; builderWidget.computeSize = () => [0, -4]; }
  const modeWidget = node.widgets?.find(w => w.name === "mode");
  if (modeWidget) { modeWidget.hidden = true; modeWidget.draw = () => {}; modeWidget.computeSize = () => [0, -4]; }
  const widthWidget = node.widgets?.find(w => w.name === "width");
  const heightWidget = node.widgets?.find(w => w.name === "height");
  for (const widget of [widthWidget, heightWidget]) { if (widget) { widget.hidden = true; widget.draw = () => {}; widget.computeSize = () => [0, -4]; } }
  const promptWidget = node.widgets?.find(w => w.name === "prompt"); if (promptWidget) { promptWidget.hidden = true; promptWidget.draw = () => {}; promptWidget.computeSize = () => [0, -4]; }
  // Legacy execution opt-in remains readable for old workflows but is no longer
  // a Director control: normal H3 execution follows the native graph path.
  const internalExecuteWidget = node.widgets?.find(w => w.name === "internal_execute");
  if (internalExecuteWidget) { internalExecuteWidget.hidden = true; internalExecuteWidget.draw = () => {}; internalExecuteWidget.computeSize = () => [0, -4]; }
  function migrateLegacyExternalPromptInput() {
    const legacyIndex = node.inputs?.findIndex(input => input.name === "external_prompt") ?? -1;
    if (legacyIndex < 0) return;
    const overwriteIndex = node.inputs?.findIndex(input => input.name === "external_prompt_overwrite") ?? -1;
    const legacy = node.inputs[legacyIndex];
    if (overwriteIndex >= 0) {
      const overwrite = node.inputs[overwriteIndex];
      if (legacy.link != null && overwrite.link == null) {
        node.removeInput(overwriteIndex);
        legacy.name = "external_prompt_overwrite";
        return;
      }
      node.removeInput(legacyIndex);
      return;
    }
    legacy.name = "external_prompt_overwrite";
  }
  migrateLegacyExternalPromptInput();
  const externalPromptWidget = () => node.widgets?.find(w => w.name === "external_prompt_overwrite");
  const externalPromptInput = () => node.inputs?.find(i => i.name === "external_prompt_overwrite");
  const externalCanvasInput = name => node.inputs?.find(i => i.name === name);
  const hasExternalCanvas = () => externalCanvasInput("external_width_overwrite")?.link != null && externalCanvasInput("external_height_overwrite")?.link != null;
  const seedWidget = () => node.widgets?.find(w => w.name === "seed");
  if (!seedWidget()) { const widget = node.addWidget("number", "seed", 0, null, { min: 0, max: 0xffffffffffffffff, step: 1 }); widget.hidden = true; widget.draw = () => {}; widget.computeSize = () => [0, -4]; }
  const hasExternalSeed = () => node.inputs?.find(i => i.name === "seed")?.link != null;
  const hasExternalPrompt = () => {
    if (externalPromptInput()?.link != null) return true;
    const w = externalPromptWidget();
    return Boolean(w && String(w.value || "").trim().length > 0);
  };
  const hasExternalSampling = () => ["external_sampler", "external_scheduler", "external_steps", "external_shift_video", "external_shift_audio"].some(n => node.inputs?.find(i => i.name === n)?.link != null);
  let lastExternalPromptLinked = hasExternalPrompt();
  let lastExternalCanvasLinked = hasExternalCanvas();
  let lastExternalSeedLinked = hasExternalSeed();
  let lastExternalSamplingLinked = hasExternalSampling();
  const status = document.createElement("div"); status.className = "ds-h3-status ds-h3-info-field"; status.textContent = ""; status.style.display = "none";

  const timeline = document.createElement("div"); timeline.className = "ds-h3 ds-h3-root"; timeline.tabIndex = 0;
  timeline.addEventListener("wheel", event => {
    const openMenu = event.target instanceof Element ? event.target.closest(".ds-h3-res-menu.open") : null;
    if (openMenu && openMenu.scrollHeight > openMenu.clientHeight) return;
    const canvas = app.canvas?.canvas;
    if (!canvas) return;
    event.preventDefault();
    canvas.dispatchEvent(new WheelEvent("wheel", {
      bubbles: true, cancelable: true, clientX: event.clientX, clientY: event.clientY,
      deltaX: event.deltaX, deltaY: event.deltaY, deltaZ: event.deltaZ,
      deltaMode: event.deltaMode, ctrlKey: event.ctrlKey, shiftKey: event.shiftKey,
      altKey: event.altKey, metaKey: event.metaKey,
    }));
  }, { passive: false });
  const setStatus = (message, isError = false) => { status.textContent = message; status.classList.toggle("error", isError); };
  const emit = () => { builderState.mode = mode(); const duration = Number(node.widgets?.find(w => w.name === "duration")?.value); if (Number.isFinite(duration)) builderState.duration = duration; const resolved = builderPromptForWidget(builderState, mode()); if (promptWidget && promptWidget.value !== resolved) { promptWidget.value = resolved; promptWidget.callback?.(resolved); } state.field_heights = { ...fieldHeights }; state.builder_state = builderState; state.resolved_prompt = resolved; dataWidget.value = JSON.stringify(state); dataWidget.callback?.(dataWidget.value); if (builderWidget) { builderWidget.value = JSON.stringify(builderState); builderWidget.callback?.(builderWidget.value); } node.graph?.setDirtyCanvas(true, true); };
  const promptStyle = () => { const v = builderState?.prompt_mode; return v === "simple" || v === "structured" ? v : "structured"; };
  function legacySimplePromptFor(m) {
    if (m === "REF2VA") {
      const r = builderState.ref || {};
      return [
        r.subject_definitions ? `subject_definitions: ${r.subject_definitions}` : "",
        r.summary ? `summary: ${r.summary}` : "",
        r.retention_analysis ? `retention_analysis: ${r.retention_analysis}` : "",
        r.detailed_description ? `detailed_description: ${r.detailed_description}` : "",
        r.soundscape ? `overall_soundscape: ${r.soundscape}` : "",
        `non_diegetic_music: ${r.music || "N/A"}`,
      ].filter(Boolean).join("\n");
    }
    return [
      builderState.imd ? `integrated_multimodal_description: ${builderState.imd}` : "",
      builderState.soundscape ? `overall_soundscape: ${builderState.soundscape}` : "",
      `non_diegetic_music: ${builderState.music || "N/A"}`,
    ].filter(Boolean).join("\n");
  }
  function builderPromptForWidget(builder, m) {
    if (builder?.prompt_mode === "simple" && typeof builder.simple_prompt === "string") return builder.simple_prompt;
    const r = builder?.ref || {};
    if (m === "REF2VA") return `subject_definitions:\n${r.subject_definitions || ""}\n\nsummary:\n${r.summary || ""}\n\nretention_analysis:\n${r.retention_analysis || ""}\n\ndetailed_description:\n${r.detailed_description || ""}\n\noverall_soundscape:\n${r.soundscape || ""}\n\nnon_diegetic_music:\n${r.music || "N/A"}`;
    const music = builder?.music || "N/A";
    let head = "";
    if (m === "I2VA") head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
    else if (m === "FL2VA") head = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the end mark of the target video.";
    else if (m === "L2VA") head = "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the end mark of the target video.";
    const body = `integrated_multimodal_description: ${builder?.imd || ""}\n\noverall_soundscape: ${builder?.soundscape || ""}\n\nnon_diegetic_music: ${music}`;
    return head ? `${head}\n\n${body}` : body;
  }
  function previewTextFor(m, external) {
    if (external) {
      const t = String(externalPromptWidget()?.value || "").trim();
      return t || "(External prompt detected — the prompt is supplied by the upstream node at execution time.)";
    }
    if (promptStyle() === "simple") return typeof builderState.simple_prompt === "string" ? builderState.simple_prompt : legacySimplePromptFor(m);
    const areas = timeline.querySelectorAll(".ds-h3-prompt-panel .ds-h3-prompt");
    if (m === "REF2VA") {
      const vals = []; areas.forEach(a => vals.push(String(a.value || "").trim()));
      const subj = vals[0] || ""; const summ = vals[1] || ""; const ret = vals[2] || ""; const detail = vals[3] || ""; const sound = vals[4] || ""; const music = vals[5] || "N/A";
      if (promptStyle() === "simple") {
        const parts = [
          subj ? `subject_definitions: ${subj}` : "",
          summ ? `summary: ${summ}` : "",
          ret ? `retention_analysis: ${ret}` : "",
          detail ? `detailed_description: ${detail}` : "",
          sound ? `overall_soundscape: ${sound}` : "",
          `non_diegetic_music: ${music || "N/A"}`,
        ].filter(Boolean);
        return parts.join("\n");
      }
      return `subject_definitions:\n${subj}\n\nsummary:\n${summ}\n\nretention_analysis:\n${ret}\n\ndetailed_description:\n${detail}\n\noverall_soundscape:\n${sound}\n\nnon_diegetic_music:\n${music}`;
    }
    const imd = String(areas[0]?.value || "").trim(); const sound = String(areas[1]?.value || "").trim(); const music = String(areas[2]?.value || "N/A").trim();
    let head = "";
    if (m === "I2VA") head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
    else if (m === "FL2VA") head = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the end mark of the target video.";
    else if (m === "L2VA") head = "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the end mark of the target video.";
    if (promptStyle() === "simple") {
      const parts = [
        imd ? `integrated_multimodal_description: ${imd}` : "",
        sound ? `overall_soundscape: ${sound}` : "",
        `non_diegetic_music: ${music || "N/A"}`,
      ].filter(Boolean);
      return parts.join("\n");
    }
    const body = `integrated_multimodal_description: ${imd}\n\noverall_soundscape: ${sound}\n\nnon_diegetic_music: ${music}`;
    return head ? `${head}\n\n${body}` : body;
  }
  const allowNativeTextEditing = element => { ["pointerdown","mousedown","keydown","keypress","keyup","copy","cut","paste"].forEach(type => element.addEventListener(type, event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") return; event.stopPropagation(); })); };

  function buildBaseForm(panel) {
    panel.replaceChildren();
    const label = document.createElement("div"); label.className = "ds-h3-small"; label.textContent = `${mode()} prompt builder`; panel.appendChild(label);
    const imdArea = createBuilderField("integrated_multimodal_description", builderState.imd, { rows: 6, placeholder: "[Shot 1] Start your scene description here...", onChange: val => { builderState.imd = val; emit(); }, fieldKey: "imd" }, fieldHeights);
    allowNativeTextEditing(imdArea.querySelector("textarea"));
    const helpers = document.createElement("div"); helpers.className = "ds-h3-actions";
    const shotBtn = document.createElement("button"); shotBtn.textContent = "Insert [Shot N]"; shotBtn.onclick = () => { const n = window.prompt("Shot number:", "1"); if (n) insertAtCursor(imdArea.querySelector("textarea"), `[Shot ${n}] `); }; helpers.appendChild(shotBtn);
    panel.appendChild(helpers); panel.appendChild(imdArea);
    const soundscape = createBuilderField("overall_soundscape", builderState.soundscape, { rows: 3, placeholder: "Describe ambient sounds, dialogue, effects...", onChange: val => { builderState.soundscape = val; emit(); }, fieldKey: "soundscape" }, fieldHeights); allowNativeTextEditing(soundscape.querySelector("textarea")); panel.appendChild(soundscape);
    const music = createBuilderField("non_diegetic_music", builderState.music, { rows: 2, placeholder: 'N/A or describe background score...', onChange: val => { builderState.music = val; emit(); }, fieldKey: "music" }, fieldHeights); allowNativeTextEditing(music.querySelector("textarea")); panel.appendChild(music);
  }

  function buildSimpleForm(panel) {
    panel.replaceChildren();
    const label = document.createElement("div"); label.className = "ds-h3-small"; label.textContent = `${mode()} simple prompt`; panel.appendChild(label);
    const simplePrompt = createBuilderField("Prompt", builderState.simple_prompt, { rows: 10, placeholder: "Write the complete MiniMax H3 prompt...", onChange: val => { builderState.simple_prompt = val; emit(); }, fieldKey: "simple_prompt" }, fieldHeights);
    allowNativeTextEditing(simplePrompt.querySelector("textarea"));
    panel.appendChild(simplePrompt);
  }

  function buildRefForm(panel) {
    panel.replaceChildren();
    const r = builderState.ref || {};
    const label = document.createElement("div"); label.className = "ds-h3-small"; label.textContent = "REF2VA prompt builder — write freely; headers are added automatically"; panel.appendChild(label);
    const helpers = document.createElement("div"); helpers.className = "ds-h3-actions";
    const shotBtn = document.createElement("button"); shotBtn.textContent = "Insert [Shot N]"; shotBtn.onclick = () => { const wrapper = panel.querySelector("[data-ref2va-target='detail']"); const ta = wrapper?.querySelector("textarea") ?? wrapper; const n = window.prompt("Shot number:", "1"); if (n && ta) insertAtCursor(ta, `[Shot ${n}] `); }; helpers.appendChild(shotBtn);
    const prefillBtn = document.createElement("button"); prefillBtn.textContent = "Prefill Labels & Summary"; prefillBtn.title = "Generate Subject/Picture/Video/Audio labels from inserted media and fill summary template"; prefillBtn.onclick = () => { generateRefLabelsAndSummary(panel); }; helpers.appendChild(prefillBtn);
    const previewBtn = document.createElement("button"); previewBtn.textContent = "Preview Prompt"; previewBtn.title = "Show the full prompt that will be sent upstream"; previewBtn.onclick = () => { showPromptPreview(); }; helpers.appendChild(previewBtn);
    panel.appendChild(helpers);
    const subjField = createBuilderField("subject_definitions:", r.subject_definitions, { rows: 4, placeholder: "<Subject 1> is the woman in <Picture 1>, with short dark hair and a red coat.\n<Picture 1> is the opening-frame anchor for [Shot 1].\n<Subject 2> is the walking motion taken from <Video 1>.\n<Video 1> provides the camera path and pacing structure.\n<Audio 1> is the voice-timbre reference for <Subject 1>.", onChange: val => { r.subject_definitions = val; emit(); }, fieldKey: "ref_subject_definitions" }, fieldHeights); allowNativeTextEditing(subjField.querySelector("textarea")); panel.appendChild(subjField);
    const summaryField = createBuilderField("summary:", r.summary, { rows: 3, placeholder: "[reference generation + audio reference] Use <Subject 1> from <Picture 1>, the motion and pacing of <Video 1>, and the voice character of <Audio 1>.", onChange: val => { r.summary = val; emit(); }, fieldKey: "ref_summary" }, fieldHeights); allowNativeTextEditing(summaryField.querySelector("textarea")); panel.appendChild(summaryField);
    const retField = createBuilderField("retention_analysis:", r.retention_analysis, { rows: 4, placeholder: "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity and clothing remain consistent.\n<Picture 1> ([Shot 1] first frame): fully_preserved - opening composition anchor.\n<Subject 2> (motion transferred to <Subject 1>): attribute_transfer - walk rhythm is applied to <Subject 1>.\n<Video 1> (pacing structure): weak_reference - general timing and camera rhythm are retained.\n<Audio 1>: reference - timbre and delivery are followed without copying the signal.", onChange: val => { r.retention_analysis = val; emit(); }, fieldKey: "ref_retention_analysis" }, fieldHeights); allowNativeTextEditing(retField.querySelector("textarea")); panel.appendChild(retField);
    const detailField = createBuilderField("detailed_description:", r.detailed_description, { rows: 5, placeholder: "[Shot 1] ... [Shot 2] At 00:04.500, ...", dataAttr: "detail", onChange: val => { r.detailed_description = val; emit(); }, fieldKey: "ref_detailed_description" }, fieldHeights); allowNativeTextEditing(detailField.querySelector("textarea")); panel.appendChild(detailField);
    const rsoundField = createBuilderField("overall_soundscape:", r.soundscape, { rows: 2, placeholder: "Audio environment description...", onChange: val => { r.soundscape = val; emit(); }, fieldKey: "ref_soundscape" }, fieldHeights); allowNativeTextEditing(rsoundField.querySelector("textarea")); panel.appendChild(rsoundField);
    const rmusicField = createBuilderField("non_diegetic_music:", r.music, { rows: 2, placeholder: "N/A or background score...", onChange: val => { r.music = val; emit(); }, fieldKey: "ref_music" }, fieldHeights); allowNativeTextEditing(rmusicField.querySelector("textarea")); panel.appendChild(rmusicField);
  }

  function generateRefLabelsAndSummary(panel) {
    const items = activeItems().sort((a, b) => {
      const typeOrder = { image: 0, video: 1, audio: 2 };
      return (typeOrder[a.type] ?? 3) - (typeOrder[b.type] ?? 3) || a.slot - b.slot;
    });
    const pictures = []; const videos = []; const audios = [];
    items.forEach(item => {
      if (item.type === "image") pictures.push(item);
      else if (item.type === "video") videos.push(item);
      else if (item.type === "audio") audios.push(item);
    });
    const subjLines = [];
    let pictureIdx = 1; let videoIdx = 1; let audioIdx = 1;
    pictures.forEach(item => {
      subjLines.push(`<Picture ${pictureIdx}> is the opening-frame anchor.`);
      pictureIdx++;
    });
    videos.forEach(item => {
      subjLines.push(`<Video ${videoIdx}> provides the camera path and pacing structure.`);
      videoIdx++;
    });
    audios.forEach(item => {
      subjLines.push(`<Audio ${audioIdx}> is the voice-timbre and audio reference.`);
      audioIdx++;
    });
    const subjArea = panel.querySelector("[data-ref2va-target='subj']")?.querySelector("textarea") || panel.querySelectorAll(".ds-h3-prompt")[0]?.querySelector("textarea") || panel.querySelectorAll(".ds-h3-prompt")[0];
    if (subjArea && subjLines.length > 0) {
      subjArea.value = subjLines.join("\n");
      builderState.ref.subject_definitions = subjArea.value;
    }
    const summaryArea = panel.querySelectorAll(".ds-h3-prompt")[1]?.querySelector("textarea") || panel.querySelectorAll(".ds-h3-prompt")[1];
    if (summaryArea) {
      const refs = [];
      pictures.forEach((_, i) => refs.push(`<Picture ${i + 1}>`));
      videos.forEach((_, i) => refs.push(`<Video ${i + 1}>`));
      audios.forEach((_, i) => refs.push(`<Audio ${i + 1}>`));
      const taskPrefixes = [];
      if (pictures.length > 0) taskPrefixes.push("reference generation");
      if (videos.length > 0) taskPrefixes.push("video editing");
      if (audios.length > 0) taskPrefixes.push("audio reference");
      const prefix = taskPrefixes.length ? `[${taskPrefixes.join(" + ")}] ` : "";
      const summaryLine = prefix + "Use " + refs.map(r => r).join(", ") + ".";
      summaryArea.value = summaryLine;
      builderState.ref.summary = summaryArea.value;
    }
    emit();
  }

  function showPromptPreview() {
    const m = mode();
    const promptText = previewTextFor(m, hasExternalPrompt());
    let overlay = document.createElement("div"); overlay.style.cssText = "position:fixed;inset:0;z-index:10002;display:flex;align-items:center;justify-content:center;background:rgba(8,10,14,.7);";
    overlay.onclick = event => { if (event.target === overlay) overlay.remove(); };
    const panel = document.createElement("div"); panel.style.cssText = "width:min(720px,90vw);max-height:85vh;display:flex;flex-direction:column;background:#111820;border:1px solid #40515e;border-radius:10px;overflow:hidden;box-shadow:0 8px 32px #000;";
    const header = document.createElement("div"); header.style.cssText = "padding:8px 12px;background:#0d1217;color:#dbe7f0;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #344452;";
    header.innerHTML = `<span>Prompt Preview (${m} · ${promptStyle() === "simple" ? "simple" : "structured"}${hasExternalPrompt() ? " · external" : ""})</span><button title="Copy">📋</button>`;
    const copyBtn = header.querySelector("button");
    copyBtn.onclick = () => { navigator.clipboard.writeText(promptText).then(() => { copyBtn.textContent = "✓"; setTimeout(() => copyBtn.textContent = "📋", 1200); }).catch(() => { copyBtn.textContent = "×"; }); };
    const closeBtn = document.createElement("button"); closeBtn.textContent = "×"; closeBtn.style.cssText = "background:none;border:none;color:#fff;font-size:18px;cursor:pointer;padding:0 4px;"; closeBtn.onclick = () => overlay.remove(); header.appendChild(closeBtn);
    const body = document.createElement("div"); body.style.cssText = "padding:10px 12px;overflow:auto;";
    const textarea = document.createElement("textarea"); textarea.readOnly = true; textarea.value = promptText; textarea.style.cssText = "width:100%;min-height:300px;max-height:60vh;resize:vertical;background:#090d11;color:#e5eef4;border:1px solid #40515e;border-radius:4px;padding:7px;font-family:monospace;font-size:11px;line-height:1.4;box-sizing:border-box;"; body.appendChild(textarea);
    const footer = document.createElement("div"); footer.style.cssText = "padding:8px 12px;background:#0d1217;display:flex;justify-content:flex-end;gap:6px;border-top:1px solid #344452;";
    const doneBtn = document.createElement("button"); doneBtn.textContent = "Done"; doneBtn.onclick = () => overlay.remove(); footer.appendChild(doneBtn);
    panel.append(header, body, footer); overlay.appendChild(panel); document.body.appendChild(overlay);
    window.addEventListener("keydown", event => { if (event.key === "Escape") overlay.remove(); }, { once: true });
  }

  let previewOverlay = null;
  let previewCloseFn = null;
  let previewItemRef = null;
  function closePreview(discard = false) {
    if (!discard && previewOverlay && previewItemRef) {
      applyPreviewChanges(previewOverlay, previewItemRef);
    }
    previewOverlay?.remove();
    previewOverlay = null;
    previewItemRef = null;
    if (previewCloseFn) window.removeEventListener("keydown", previewCloseFn);
    previewCloseFn = null;
  }
  function applyPreviewChanges(overlay, item) {
    const tsInput = overlay.querySelector(".ds-h3-ts-input");
    const teInput = overlay.querySelector(".ds-h3-te-input");
    const sourceDuration = Number(item.source_duration) || Number(item.duration) || 0;
    if ((item.type === "video" || item.type === "audio") && sourceDuration > 0) {
      const ts = parseFloat(tsInput?.value) ?? 0;
      const te = parseFloat(teInput?.value) ?? sourceDuration;
      mutate(s => {
        const x = s.items.find(i => i.id === item.id);
        if (x) {
          x.trim_start = Math.min(sourceDuration, Math.max(0, ts));
          x.trim_end = Math.min(sourceDuration, Math.max(x.trim_start + 2, te));
          x.duration = x.trim_end - x.trim_start;
        }
      });
    }
  }
  function openPreview(item) {
    closePreview();
    previewItemRef = item;
    const sourceDuration = Number(item.source_duration) || Number(item.duration) || 0;
    const trimStart = Number(item.trim_start || 0);
    const trimEnd = Number(item.trim_end ?? sourceDuration);
    previewOverlay = document.createElement("div"); previewOverlay.className = "ds-h3-preview-overlay";
    previewOverlay.onclick = event => { if (event.target === previewOverlay) closePreview(); };
    const panel = document.createElement("div"); panel.className = "ds-h3-preview-panel";
    const header = document.createElement("div"); header.className = "ds-h3-preview-header";
    const title = document.createElement("span"); title.textContent = `${item.type}: ${String(item.value || "").split("/").pop()}`;
    const close = document.createElement("button"); close.textContent = "×"; close.title = "Close and save (Escape or click outside)"; close.onclick = () => closePreview(false); header.append(title, close);
    const body = document.createElement("div"); body.className = "ds-h3-preview-body";
    const media = document.createElement(item.type === "image" ? "img" : item.type === "audio" ? "audio" : "video"); media.className = "ds-h3-preview-media"; media.src = viewUrl(item.value); if (item.type !== "image") media.controls = true; body.append(media);
    const controls = document.createElement("div"); controls.className = "ds-h3-preview-controls";
    if ((item.type === "video" || item.type === "audio") && sourceDuration > 0) {
      const row1 = document.createElement("div"); row1.className = "ds-h3-preview-row";
      const lblTs = document.createElement("span"); lblTs.textContent = "Start:"; row1.appendChild(lblTs);
      const tsInput = document.createElement("input"); tsInput.type = "number"; tsInput.step = "0.25"; tsInput.min = "0"; tsInput.max = String(sourceDuration.toFixed(2)); tsInput.value = String(trimStart.toFixed(2)); tsInput.className = "ds-h3-ts-input"; row1.appendChild(tsInput);
      const lblTe = document.createElement("span"); lblTe.textContent = "End:"; row1.appendChild(lblTe);
      const teInput = document.createElement("input"); teInput.type = "number"; teInput.step = "0.25"; teInput.min = "0"; teInput.max = String(sourceDuration.toFixed(2)); teInput.value = String(trimEnd.toFixed(2)); teInput.className = "ds-h3-te-input"; row1.appendChild(teInput);
      const lblDur = document.createElement("span"); lblDur.textContent = `/ ${sourceDuration.toFixed(2)}s`; row1.appendChild(lblDur);
      const playCropBtn = document.createElement("button"); playCropBtn.textContent = "▶ Play crop"; playCropBtn.title = "Play only the current crop range"; playCropBtn.className = "ds-h3-play-crop-btn"; row1.appendChild(playCropBtn);
      controls.appendChild(row1);
      const rowSlider = document.createElement("div"); rowSlider.className = "ds-h3-preview-row"; rowSlider.style.flexDirection = "column"; rowSlider.style.alignItems = "stretch";
      const minRef = Math.min(2, sourceDuration);
      const updateFromInputs = () => {
        let ts = parseFloat(tsInput.value) ?? 0;
        let te = parseFloat(teInput.value) ?? sourceDuration;
        ts = Math.min(Math.max(0, ts), sourceDuration - minRef);
        te = Math.max(ts + minRef, Math.min(sourceDuration, te));
        tsInput.value = ts.toFixed(2);
        teInput.value = te.toFixed(2);
        const sPct = (ts / sourceDuration) * 100;
        const ePct = (te / sourceDuration) * 100;
        rangeTs.value = sPct; rangeTe.value = ePct;
        syncMarkers();
      };
      const trackBg = document.createElement("div"); trackBg.title = "Drag the highlighted crop to move it; drag either end marker to resize it."; trackBg.style.cssText = "position:relative;height:18px;background:#111a21;border-radius:3px;margin-top:4px;margin-bottom:4px;cursor:grab;";
      const filledBar = document.createElement("div"); filledBar.style.cssText = "position:absolute;top:0;left:0;right:0;bottom:0;border-radius:3px;background:rgba(126,210,157,.25);pointer-events:none;";
      const markerS = document.createElement("div"); markerS.style.cssText = "position:absolute;top:-2px;width:8px;height:22px;background:#f3c67a;border-radius:2px;z-index:3;cursor:pointer;transform:translateX(-50%);";
      const markerE = document.createElement("div"); markerE.style.cssText = "position:absolute;top:-2px;width:8px;height:22px;background:#8dd7ff;border-radius:2px;z-index:3;cursor:pointer;transform:translateX(-50%);";
      const rangeTs = document.createElement("input"); rangeTs.type = "range"; rangeTs.min = "0"; rangeTs.max = "100"; rangeTs.step = "0.1"; rangeTs.value = ((trimStart / sourceDuration) * 100).toFixed(1); rangeTs.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;pointer-events:none;z-index:0;";
      const rangeTe = document.createElement("input"); rangeTe.type = "range"; rangeTe.min = "0"; rangeTe.max = "100"; rangeTe.step = "0.1"; rangeTe.value = ((trimEnd / sourceDuration) * 100).toFixed(1); rangeTe.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;pointer-events:none;z-index:0;";
      const MARKER_GRAB_RADIUS_PX = 12;
      const pointerPct = pointerX => {
        const rect = trackBg.getBoundingClientRect();
        return ((pointerX - rect.left) / rect.width) * 100;
      };
      const syncMarkers = () => {
        const sPct = parseFloat(rangeTs.value);
        const ePct = parseFloat(rangeTe.value);
        markerS.style.left = sPct + "%";
        markerE.style.left = ePct + "%";
        filledBar.style.left = sPct + "%";
        filledBar.style.right = (100 - ePct) + "%";
      };
      const clampSliders = () => {
        let s = parseFloat(rangeTs.value);
        let e = parseFloat(rangeTe.value);
        const gap = (minRef / sourceDuration) * 100;
        if (e - s < gap) {
          if (dragging === "left") e = s + gap;
          else s = e - gap;
        }
        rangeTs.value = Math.max(0, Math.min(s, 100 - gap));
        rangeTe.value = Math.max(gap, Math.min(e, 100));
        syncMarkers();
        const tsSec = (parseFloat(rangeTs.value) / 100) * sourceDuration;
        const teSec = (parseFloat(rangeTe.value) / 100) * sourceDuration;
        tsInput.value = tsSec.toFixed(2);
        teInput.value = teSec.toFixed(2);
      };
      let dragging = null;
      let cropDragOffset = 0;
      const onStart = pointerX => {
        const pct = pointerPct(pointerX);
        const sPct = parseFloat(rangeTs.value);
        const ePct = parseFloat(rangeTe.value);
        const markerGrabRadius = (MARKER_GRAB_RADIUS_PX / trackBg.getBoundingClientRect().width) * 100;
        if (Math.abs(pct - sPct) <= markerGrabRadius && Math.abs(pct - sPct) <= Math.abs(pct - ePct)) {
          dragging = "left";
        } else if (Math.abs(pct - ePct) <= markerGrabRadius) {
          dragging = "right";
        } else if (pct > sPct && pct < ePct) {
          dragging = "range";
          cropDragOffset = pct - sPct;
        } else {
          dragging = null;
        }
      };
      const onMove = pointerX => {
        if (!dragging) return;
        const pct = pointerPct(pointerX);
        if (dragging === "left") {
          rangeTs.value = Math.max(0, Math.min(100, pct));
        } else if (dragging === "right") {
          rangeTe.value = Math.max(0, Math.min(100, pct));
        } else {
          const width = parseFloat(rangeTe.value) - parseFloat(rangeTs.value);
          const start = Math.max(0, Math.min(100 - width, pct - cropDragOffset));
          rangeTs.value = start;
          rangeTe.value = start + width;
        }
        clampSliders();
      };
      const onEnd = () => { dragging = null; trackBg.style.cursor = "grab"; };
      const onPointerStart = pointerX => { onStart(pointerX); if (dragging) trackBg.style.cursor = "grabbing"; };
      trackBg.addEventListener("mousedown", event => { onPointerStart(event.clientX); });
      window.addEventListener("mousemove", event => { onMove(event.clientX); });
      window.addEventListener("mouseup", () => { onEnd(); });
      trackBg.addEventListener("touchstart", event => { onPointerStart(event.touches[0].clientX); }, { passive: false });
      trackBg.addEventListener("touchmove", event => { event.preventDefault(); onMove(event.touches[0].clientX); }, { passive: false });
      trackBg.addEventListener("touchend", () => { onEnd(); });
      tsInput.onchange = updateFromInputs;
      teInput.onchange = updateFromInputs;
      let cropPlayback = false;
      const stopCropPlayback = () => { cropPlayback = false; playCropBtn.textContent = "▶ Play crop"; };
      playCropBtn.onclick = async () => {
        const start = Number(tsInput.value);
        const end = Number(teInput.value);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
        media.currentTime = start;
        cropPlayback = true;
        playCropBtn.textContent = "Playing crop…";
        try { await media.play(); }
        catch (error) { console.warn("[MiniMax H3 Director] Crop playback failed", error); stopCropPlayback(); }
      };
      media.addEventListener("timeupdate", () => {
        if (cropPlayback && media.currentTime >= Number(teInput.value)) {
          media.pause();
          media.currentTime = Number(teInput.value);
          stopCropPlayback();
        }
      });
      media.addEventListener("pause", () => {
        if (cropPlayback && media.currentTime < Number(teInput.value)) stopCropPlayback();
      });
      media.addEventListener("ended", stopCropPlayback);
      updateFromInputs();
      trackBg.append(filledBar, markerS, markerE, rangeTs, rangeTe);
      rowSlider.appendChild(trackBg);
      controls.appendChild(rowSlider);
    }
    const row2 = document.createElement("div"); row2.className = "ds-h3-preview-row";
    const saveBtn = document.createElement("button"); saveBtn.textContent = "Save"; saveBtn.className = "ds-h3-save-btn"; saveBtn.onclick = () => { applyPreviewChanges(previewOverlay, item); closePreview(false); }; row2.appendChild(saveBtn);
    const cancelBtn = document.createElement("button"); cancelBtn.textContent = "Cancel"; cancelBtn.onclick = () => { closePreview(true); }; cancelBtn.style.background = "#2c2c2c"; row2.appendChild(cancelBtn);
    controls.appendChild(row2);
    const meta = document.createElement("div"); meta.className = "ds-h3-preview-meta"; meta.textContent = sourceDuration && (item.type === "video" || item.type === "audio") ? `Source: ${sourceDuration.toFixed(2)}s · Current crop: ${trimStart.toFixed(2)}s – ${trimEnd.toFixed(2)}s` : "Full media preview";
    panel.append(header, body, controls, meta); previewOverlay.append(panel); document.body.append(previewOverlay);
    previewCloseFn = event => { if (event.key === "Escape") closePreview(); };
    window.addEventListener("keydown", previewCloseFn);
  }
  const syncAttachedPrompts = () => { const attached = state.items.filter(item => String(item.prompt || "").trim()).map((item, index) => ({ id: `attached-${item.id}`, text: String(item.prompt).trim(), enabled: item.enabled !== false, start: Number(item.start) || 0, duration: Number(item.duration) || 1, order: index })); if (attached.length || state.items.every(item => !item.prompt)) state.prompt_blocks = attached; };
  const mutate = fn => { fn(state); syncAttachedPrompts(); state.items.forEach((x, i) => { x.order = i; }); state.prompt_blocks.forEach((x, i) => { x.order = i; }); applyResolution?.(); emit(); render(); };
  const activeItems = () => state.items.filter(x => x.enabled !== false);
  const isReferenceMode = () => mode() === "REF2VA";
  const allowsType = type => mode() === "Image Inpaint" ? type === "image" : isReferenceMode() || ((mode() === "I2VA" || mode() === "FL2VA" || mode() === "L2VA") && type === "image");
  const frameSlots = () => mode() === "T2VA" ? [] : (mode() === "I2VA" || mode() === "Image Inpaint") ? [0] : mode() === "FL2VA" ? [0, 1] : mode() === "L2VA" ? [0, 1] : [];
  const imageCapacity = () => frameSlots().length;
  const isLockedSlot = (item) => mode() === "L2VA" && laneForItem(item) === "visual" && item.slot === 0;
  const displayedItems = () => isReferenceMode() ? activeItems() : activeItems().filter(x => x.type === "image").sort((a, b) => a.slot - b.slot).slice(0, imageCapacity());
  const ensureLayout = () => { let cursor = 0; state.items.forEach(item => { if (!Number.isFinite(item.start)) item.start = cursor; if (!Number.isFinite(item.duration)) item.duration = item.type === "image" ? 1 : 2; cursor = Math.max(cursor, item.start + item.duration + 0.25); }); };
  let insertAt = 0;
  let selectedId = null;
  let selectedLane = "visual";
  let lastTimelineLength = null;
  let mediaPromptHeight = 120;
  let globalPromptHeight = 120;
  const hiddenLanes = new Set();
  const laneForItem = item => item.type === "audio" ? "audio" : "visual";
  const availableSlots = lane => lane === "audio" ? [...Array(isReferenceMode() ? MAX.audio : 0).keys()] : isReferenceMode() ? [...Array(MAX.image + MAX.video).keys()] : frameSlots();
  const addItem = item => mutate(s => { const lane = laneForItem(item); const occupied = new Set(s.items.filter(x => laneForItem(x) === lane).map(x => x.slot).filter(Number.isInteger)); const slot = availableSlots(lane).find(index => !occupied.has(index)); if (slot == null) throw new Error(`No free ${lane} slot is available.`); s.items.push({ id: idFor(item.type, s.items.length), enabled: true, order: s.items.length, slot, start: slot, duration: item.type === "image" ? 1 : 2, ...item }); });
  const remove = id => mutate(s => { s.items = s.items.filter(x => x.id !== id); if (selectedId === id) selectedId = null; });
  const resetBuilderState = () => { builderState = DEFAULT_BUILDER_STATE(mode()); builderState.mode = mode(); };
  const hasBuilderContent = () => [builderState.imd, builderState.soundscape, builderState.simple_prompt, ...Object.values(builderState.ref || {})].some(value => typeof value === "string" && value !== "N/A" && value.trim());
  const clearAll = () => { selectedId = null; resetBuilderState(); if (promptWidget) { promptWidget.value = ""; promptWidget.callback?.(promptWidget.value); } mutate(s => { s.items = []; s.prompt_blocks = []; }); setStatus("All media and prompts cleared."); };
  timeline.addEventListener("keydown", event => { if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return; if ((event.key === "Delete" || event.key === "Backspace") && selectedId) { const sel = state.items.find(x => x.id === selectedId); if (sel && isLockedSlot(sel)) { setStatus(`${mediaReferenceName(sel.type)} ${sel.slot + 1} is locked in L2VA mode`, true); return; } event.preventDefault(); event.stopPropagation(); remove(selectedId); } });
  const move = (id, delta) => mutate(s => { const i = s.items.findIndex(x => x.id === id); const j = i + delta; if (i >= 0 && j >= 0 && j < s.items.length) [s.items[i], s.items[j]] = [s.items[j], s.items[i]]; });
  const replace = (id, value) => mutate(s => { const x = s.items.find(i => i.id === id); if (x) x.value = value; });
  const resolutionState = () => ({ aspect: "auto", resolution: "auto", input_scaling: "Auto", custom_aspect_w: 16, custom_aspect_h: 9, custom_mp: 1, custom_width: 1344, custom_height: 768, ...(state.resolution || {}) });
  const snap16 = value => Math.max(MINIMAX_MULTIPLE, Math.round(Number(value) / MINIMAX_MULTIPLE) * MINIMAX_MULTIPLE);
  const sourceDimensions = () => activeItems().filter(item => (item.type === "image" || item.type === "video") && Number(item.source_width) > 0 && Number(item.source_height) > 0).sort((a, b) => a.order - b.order)[0];
  const selectedAspect = settings => {
    if (settings.aspect === "auto") { const source = sourceDimensions(); return source ? Number(source.source_width) / Number(source.source_height) : 4 / 3; }
    if (settings.aspect === "custom") return Math.max(1, Number(settings.custom_aspect_w) || 16) / Math.max(1, Number(settings.custom_aspect_h) || 9);
    const [w, h] = String(settings.aspect).split(":").map(Number); return w > 0 && h > 0 ? w / h : null;
  };
  const setCanvasWidgets = (width, height) => { for (const [widget, value] of [[widthWidget, width], [heightWidget, height]]) { if (widget) { widget.value = value; widget.callback?.(value); } } node.setDirtyCanvas?.(true, true); };
  const resolveCanvas = settings => {
    if (settings.resolution === "custom" && settings.custom_mode === "fixed") return [snap16(settings.custom_width), snap16(settings.custom_height)];
    const aspect = selectedAspect(settings); if (!aspect) return null;
    if (settings.resolution === "auto") { const shortSide = 768; return aspect >= 1 ? [snap16(shortSide * aspect), shortSide] : [shortSide, snap16(shortSide / aspect)]; }
    if (settings.resolution === "custom") { const pixels = Math.max(.01, Number(settings.custom_mp) || 1) * 1024 * 1024; const h = Math.sqrt(pixels / aspect); return [snap16(h * aspect), snap16(h)]; }
    const pixels = Number(RESOLUTION_PRESETS[settings.resolution]) * 1024 * 1024; const h = Math.sqrt(pixels / aspect); return [snap16(h * aspect), snap16(h)];
  };
  const applyResolution = () => { if (hasExternalCanvas()) return; const canvas = resolveCanvas(resolutionState()); if (canvas) setCanvasWidgets(...canvas); };

  const fileInput = document.createElement("input"); fileInput.type = "file"; fileInput.multiple = true; fileInput.accept = "image/*,video/*,audio/*"; fileInput.hidden = true;
  fileInput.onchange = async () => { for (const file of fileInput.files || []) await acceptFile(file); fileInput.value = ""; };
  timeline.addEventListener("paste", async event => {
    if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
    const files = [...(event.clipboardData?.files || [])];
    if (!files.length) files.push(...[...(event.clipboardData?.items || [])].filter(item => item.kind === "file").map(item => item.getAsFile()).filter(Boolean));
    if (!files.length) return;
    event.preventDefault(); event.stopPropagation();
    const selected = state.items.find(item => item.id === selectedId);
    if (selected) {
      const replacementFile = files.find(file => { const type = mediaTypeFor(file); return type && allowsType(type) && laneForItem(selected) === (type === "audio" ? "audio" : "visual"); });
      if (replacementFile) {
        if (isLockedSlot(selected)) { setStatus(`${mediaReferenceName(selected.type)} ${selected.slot + 1} is locked in L2VA mode`, true); return; }
        await replaceSelectedFile(replacementFile, selected);
        return;
      }
    }
    if (mode() === "FL2VA" && selectedLane === "audio") { setStatus("FL2VA supports image references only; select the Image/Video lane.", true); return; }
    setStatus(`Pasting ${files.length} file${files.length === 1 ? "" : "s"} into the ${selectedLane === "audio" ? "Audio" : "Image/Video"} lane.`);
    for (const file of files) await acceptFile(file, selectedLane);
  });
  async function probeWavDuration(value) {
    try { return wavDurationFromBuffer(await (await fetch(viewUrl(value))).arrayBuffer()); }
    catch (error) { console.warn("[MiniMax H3 Director] WAV metadata probe failed", error); return null; }
  }
  async function probeDuration(value, type) {
    if (type === "image") return null;
    const media = document.createElement(type === "video" ? "video" : "audio");
    media.preload = "metadata";
    media.src = viewUrl(value);
    const duration = await new Promise(resolve => {
      const done = result => { media.remove(); resolve(Number.isFinite(result) ? result : null); };
      media.onloadedmetadata = () => done(media.duration);
      media.onerror = () => done(null);
    });
    return duration ?? (type === "audio" ? await probeWavDuration(value) : null);
  }
  async function probeDimensions(value, type) {
    if (type !== "image" && type !== "video") return null;
    const media = document.createElement(type === "video" ? "video" : "img");
    media.preload = "metadata";
    media.src = viewUrl(value);
    return await new Promise(resolve => {
      const done = () => { const width = type === "video" ? media.videoWidth : media.naturalWidth; const height = type === "video" ? media.videoHeight : media.naturalHeight; media.remove(); resolve(width > 0 && height > 0 ? { source_width: width, source_height: height } : null); };
      if (type === "video") media.onloadedmetadata = done; else media.onload = done;
      media.onerror = () => { media.remove(); resolve(null); };
    });
  }
  async function captureFirstFrame(videoUrl) {
    return await new Promise(resolve => {
      const vid = document.createElement("video");
      vid.preload = "auto"; vid.muted = true; vid.crossOrigin = "";
      const fail = () => { vid.remove(); resolve(null); };
      const draw = () => {
        try {
          const c = document.createElement("canvas");
          c.width = vid.videoWidth || 160; c.height = vid.videoHeight || 96;
          const ctx = c.getContext("2d");
          ctx.drawImage(vid, 0, 0, c.width, c.height);
          vid.remove();
          resolve(c.toDataURL("image/jpeg", 0.7));
        } catch { fail(); }
      };
      vid.onloadeddata = () => { vid.currentTime = 0; };
      vid.onseeked = draw;
      vid.onerror = fail;
      vid.src = videoUrl;
    });
  }
  async function extractWaveform(value, id) {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const response = await fetch(viewUrl(value));
      const buffer = await audioContext.decodeAudioData(await response.arrayBuffer());
      const channel = buffer.getChannelData(0); const peakCount = 240; const step = Math.max(1, Math.floor(channel.length / peakCount)); const peaks = [];
      for (let index = 0; index < peakCount; index += 1) { let peak = 0; for (let sample = index * step; sample < Math.min(channel.length, (index + 1) * step); sample += 1) peak = Math.max(peak, Math.abs(channel[sample])); peaks.push(peak); }
      audioContext.close?.();
      mutate(s => { const item = s.items.find(x => x.id === id); if (item) item.waveform_peaks = peaks; });
    } catch (error) { console.warn("[MiniMax H3 Director] Audio waveform decode failed", error); }
  }
  async function replaceSelectedFile(file, selected) {
    const type = mediaTypeFor(file);
    const lane = type === "audio" ? "audio" : "visual";
    if (!type || !allowsType(type) || laneForItem(selected) !== lane) { setStatus(`Paste a compatible ${laneForItem(selected) === "audio" ? "audio" : "image or video"} file to replace the selected media.`, true); return; }
    if (count(state, type) - (selected.type === type ? 1 : 0) >= MAX[type]) { setStatus(`Limit reached: ${MAX[type]} ${type}s.`, true); return; }
    try {
      const value = await uploadFile(file, status);
      const [sourceDuration, dimensions] = await Promise.all([probeDuration(value, type), probeDimensions(value, type)]);
      if (sourceDuration !== null && sourceDuration < 2) { setStatus(`${file.name}: MiniMax references must be at least 2 seconds.`, true); return; }
      const duration = sourceDuration === null ? null : Math.min(sourceDuration, 15);
      const thumbnail = type === "video" ? await captureFirstFrame(viewUrl(value)) : null;
      const { thumbnail: oldThumbnail, source_width, source_height, duration: oldDuration, source_duration, trim_start, trim_end, waveform_peaks, ...stable } = selected;
      const item = { ...stable, type, value, thumbnail, ...dimensions, ...(duration !== null ? { duration, source_duration: sourceDuration } : {}), ...((type === "video" || type === "audio") ? { trim_start: 0, trim_end: duration } : {}) };
      mutate(s => { const index = s.items.findIndex(existing => existing.id === selected.id); if (index >= 0) s.items[index] = item; });
      setStatus(`Replaced selected ${mediaReferenceName(type)}.`);
      if (type === "audio") void extractWaveform(value, selected.id);
    } catch (error) { console.error(error); setStatus(`Upload failed: ${error.message || error}`, true); }
  }
  async function acceptFile(file, targetLane = null) { const type = mediaTypeFor(file); const validLane = targetLane === "visual" ? type === "image" || type === "video" : targetLane === "audio" ? type === "audio" : true; const modeSupportsType = !!type && allowsType(type); const lane = type === "audio" ? "audio" : "visual"; if (!type || !validLane || !modeSupportsType) { const requirement = mode() === "FL2VA" ? "FL2VA supports image references only; video and audio are unavailable." : targetLane ? `Drop ${targetLane === "audio" ? "audio" : "image or video"} files on this lane.` : "This media type is not available in the selected MiniMax mode."; setStatus(requirement, true); return; } if (count(state, type) >= MAX[type] || activeItems().length >= MAX.total) { setStatus(`Limit reached: ${MAX[type]} ${type}s / ${MAX.total} files.`, true); return; } const laneAvail = availableSlots(lane); const laneOccupied = new Set(activeItems().filter(x => laneForItem(x) === lane).map(x => x.slot).filter(Number.isInteger)); const laneFree = laneAvail.some(slot => !laneOccupied.has(slot)); if (!laneFree) { setStatus(`No free ${lane} slot is available.`, true); return; } try { const value = await uploadFile(file, status); const [sourceDuration, dimensions] = await Promise.all([probeDuration(value, type), probeDimensions(value, type)]); if (sourceDuration !== null && sourceDuration < 2) { setStatus(`${file.name}: MiniMax references must be at least 2 seconds.`, true); return; } const duration = sourceDuration === null ? null : Math.min(sourceDuration, 15); let thumbnail = null; if (type === "video") { thumbnail = await captureFirstFrame(viewUrl(value)); } const item = { type, value, thumbnail, ...dimensions, ...(duration !== null ? { duration, source_duration: sourceDuration } : {}), ...((type === "video" || type === "audio") ? { trim_start: 0, trim_end: duration } : {}) }; addItem(item); applyResolution(); const added = state.items[state.items.length - 1]; if (type === "audio") void extractWaveform(value, added.id); setStatus(sourceDuration > 15 ? `${file.name} added; cropped to the first 15 seconds.` : `${file.name} added.`); } catch (error) { setStatus(error.message || "Upload failed", true); } }
  function selectDirectorMode(value) {
    if (!modeWidget) return;
    selectedModeOverride = value;
    modeWidget.value = value;
    modeWidget.callback?.call(modeWidget, value);
    builderState.mode = value;
    node.graph?.setDirtyCanvas(true, true);
  }

  const render = () => {
    if (!Array.isArray(state.postprocess_recipe)) state.postprocess_recipe = DEFAULT_POSTPROCESS_RECIPE;
    timeline.replaceChildren();
    const selected = state.items.find(item => item.id === selectedId);
    const modeGroup = document.createElement("div"); modeGroup.className = "ds-h3-mode-group"; modeGroup.style.display = "flex"; modeGroup.style.flexDirection = "column"; modeGroup.style.alignItems = "flex-start"; modeGroup.style.gap = "4px"; modeGroup.style.padding = "6px"; modeGroup.style.background = "#0d1217"; modeGroup.style.border = "1px solid #344452"; modeGroup.style.borderRadius = "6px";
    const topRow = document.createElement("div"); topRow.className = "ds-h3-modebar"; topRow.style.flexWrap = "wrap"; topRow.style.gap = "8px"; topRow.style.padding = "0"; topRow.style.border = "0"; topRow.style.background = "transparent"; topRow.style.width = "100%";
    const controlGroup = () => { const group = document.createElement("span"); group.className = "ds-h3-actions"; group.style.cssText = "gap:4px;flex-wrap:nowrap;white-space:nowrap"; return group; };
    const modesSide = controlGroup(); const modeLabel = document.createElement("span"); modeLabel.textContent = "Model Mode:"; modeLabel.style.cssText = "color:#9fb3c2;font-weight:600"; modesSide.append(modeLabel); ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA", "Image Inpaint"].forEach(value => { const button = document.createElement("button"); button.textContent = value; button.title = value === "Image Inpaint" ? "One image reference; output exactly one frame through Get Image from Batch." : `Use MiniMax H3 ${value} mode`; button.classList.toggle("active", mode() === value); button.onclick = () => { const previousMode = mode(); if (value === "Image Inpaint") { const images = activeItems().filter(item => item.type === "image"); if (!images.length) setStatus("Image Inpaint selected — add one image reference to continue."); state.items = state.items.filter(item => item.enabled === false || item.type === "image").map((item, index) => index === 0 ? { ...item, slot: 0, order: 0 } : { ...item, enabled: false }); state.image_inpaint_saved_duration ??= Number(node.widgets?.find(w => w.name === "duration")?.value) || 5; const durationWidget = node.widgets?.find(w => w.name === "duration"); if (durationWidget) { durationWidget.value = 1; durationWidget.callback?.call(durationWidget, 1); } } else if (previousMode === "Image Inpaint" && Number.isFinite(Number(state.image_inpaint_saved_duration))) { const durationWidget = node.widgets?.find(w => w.name === "duration"); if (durationWidget) { durationWidget.value = state.image_inpaint_saved_duration; durationWidget.callback?.call(durationWidget, durationWidget.value); } delete state.image_inpaint_saved_duration; } selectDirectorMode(value); if (selectedLane === "audio" && value !== "REF2VA") selectedLane = "visual"; emit(); render(); }; modesSide.append(button); });
    const promptSide = controlGroup(); const promptLabel = document.createElement("span"); promptLabel.textContent = "Prompt Mode:"; promptLabel.style.cssText = "color:#9fb3c2;font-weight:600"; promptSide.append(promptLabel); const styleLabel = promptStyle(); [["simple", "Simple"], ["structured", "Structured"]].forEach(([value, label]) => { const promptButton = document.createElement("button"); promptButton.className = "ds-h3-prompt-mode-btn"; promptButton.textContent = label; promptButton.classList.toggle("active", styleLabel === value); promptButton.title = `Use the ${label.toLowerCase()} prompt editor`; promptButton.onclick = () => { if (styleLabel === value) return; if (value === "simple") builderState.simple_prompt = previewTextFor(mode(), false); builderState.prompt_mode = value; emit(); render(); }; promptSide.append(promptButton); });
    const actionsSide = controlGroup(); actionsSide.style.cssText += ";padding-left:8px;border-left:1px solid #344452"; const hasContent = state.items.length || state.prompt_blocks?.length || hasBuilderContent() || String(promptWidget?.value || "").trim(); if (selected) { if (!isLockedSlot(selected)) { const removeButton = document.createElement("button"); removeButton.className = "ds-h3-remove-btn"; removeButton.textContent = "Remove"; removeButton.title = `Remove selected ${selected.type}`; removeButton.onclick = () => remove(selected.id); actionsSide.append(removeButton); } else { setStatus(`${mediaReferenceName(selected.type)} ${selected.slot + 1} is locked in L2VA mode`, true); } } if (hasContent) { const clearButton = document.createElement("button"); clearButton.className = "ds-h3-clear-btn"; clearButton.textContent = "Clear"; clearButton.title = "Remove all media and prompts"; clearButton.onclick = clearAll; actionsSide.append(clearButton); } else { const clearButton = document.createElement("button"); clearButton.className = "ds-h3-clear-btn ds-h3-clear-btn-empty"; clearButton.textContent = "Clear"; clearButton.title = "Nothing to clear yet"; clearButton.onclick = () => setStatus("Nothing to clear."); actionsSide.append(clearButton); }
    const docsButton = document.createElement("button"); docsButton.className = "ds-h3-docs"; docsButton.textContent = "?"; docsButton.title = "Open MiniMax H3 Director documentation on GitHub"; docsButton.onclick = () => window.open(REPOSITORY_URL, "_blank", "noopener,noreferrer");
    topRow.append(modesSide, actionsSide, docsButton); modeGroup.append(topRow);
    const promptRow = document.createElement("div"); promptRow.className = "ds-h3-modebar"; promptRow.style.cssText = "padding:0;border:0;background:transparent;width:100%"; promptRow.append(promptSide); modeGroup.append(promptRow);
    const modeHint = { T2VA: "T2VA · no input frame", I2VA: "I2VA · one opening-frame slot", FL2VA: "FL2VA · opening and closing-frame slots", L2VA: "L2VA · one closing-frame slot", "Image Inpaint": "Image Inpaint · exactly one image · output frame count 1 · no video/audio" }; const hint = document.createElement("div"); hint.className = "ds-h3-mode-hint"; hint.style.fontSize = "11px"; hint.style.color = "#9fb3c2"; hint.style.margin = "0"; hint.textContent = modeHint[mode()] || `REF2VA · up to ${MAX.image} image, ${MAX.video} video, and ${MAX.audio} audio slots · ${MAX.total} combined files maximum`; modeGroup.append(hint);
    // Prompt controls belong with the mode pills; save configuration has one
    // owner: the dedicated Save node panel.
    const controlRow = document.createElement("div"); controlRow.className = "ds-h3-control-row"; controlRow.style.cssText = "width:100%;box-sizing:border-box;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:6px;margin-top:6px";
    const postprocessPanel = document.createElement("div"); postprocessPanel.className = "ds-h3-postprocess-row"; postprocessPanel.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-wrap:wrap;align-items:center;gap:5px;padding:4px 0 0;border-top:1px solid #344452;margin-top:2px";
    const saveNodePanel = document.createElement("section"); saveNodePanel.className = "ds-h3-save-node"; saveNodePanel.style.cssText = "min-width:0;min-height:420px;height:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:6px;padding:8px;background:#0d1217;border:1px solid #344452;border-radius:6px";
    let optimizationPanel = null; let postprocessInfo = null; let samplingPanel = null;
    {

      const openStageSettings = (label, target, fieldList) => {
        const fields = fieldList || (target.id ? (PP_STAGE_FIELDS[target.id] || []) : []);
        const overlay = document.createElement("div"); overlay.style.cssText = "position:fixed;inset:0;z-index:10020;display:grid;place-items:center;background:rgba(0,0,0,.66)";
        const modal = document.createElement("div"); modal.style.cssText = "width:min(520px,calc(100vw - 32px));max-height:80vh;overflow:auto;padding:14px;background:#111a21;border:1px solid #4c6b80;border-radius:8px;color:#d5e6f2";
        const heading = document.createElement("h3"); heading.textContent = label; heading.style.margin = "0 0 10px"; modal.append(heading);
        const body = document.createElement("div"); modal.append(body);
        const inputStyle = "padding:3px 6px;background:#16283a;border:1px solid #2f5478;border-radius:4px;color:#d6ebff;font:12px system-ui,sans-serif";
        const ensureValue = (fspec) => { if (!(fspec.key in target)) target[fspec.key] = fspec.default; };
        const makeCombo = (fspec, options, selected) => { const select = document.createElement("select"); select.style.cssText = inputStyle; const chosen = options.includes(selected) ? selected : (fspec.default && options.includes(fspec.default) ? fspec.default : options[0]); for (const opt of options) { const o = document.createElement("option"); o.value = opt; o.textContent = opt; if (opt === chosen) o.selected = true; select.append(o); } if (selected != null && selected !== "" && !options.includes(selected)) { const o = document.createElement("option"); o.value = selected; o.textContent = `${selected} (custom)`; o.selected = true; select.append(o); } return select; };
        for (const fspec of fields) {
          ensureValue(fspec);
          const value = target[fspec.key];
          const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;margin:7px 0;font-size:11px";
          const title = document.createElement("span"); title.textContent = fspec.key.replaceAll("_", " "); title.style.cssText = "color:#8fb3d6"; if (fspec.type !== "boolean") field.append(title);
          if (fspec.type === "imagePicker") {
            ppImagePicker(field, fspec.key, target, inputStyle, String(target[fspec.key] || ""));
            body.append(field);
          } else if (fspec.type === "modelFolder") {
            const select = document.createElement("select"); select.style.cssText = inputStyle; select.disabled = true;
            const placeholder = document.createElement("option"); placeholder.value = "(loading…)"; placeholder.textContent = "(loading…)"; placeholder.selected = true; select.append(placeholder);
            field.append(select); body.append(field);
            void ppLoadModelList(fspec.folder).then((options) => {
              if (!select.isConnected) return;
              select.innerHTML = "";
              const opts = options.length ? options : ["(no models in " + fspec.folder + ")"];
              const current = String(target[fspec.key] || "");
              for (const opt of opts) { const o = document.createElement("option"); o.value = opt; o.textContent = opt; if (opt === current) o.selected = true; select.append(o); }
              if (current && !opts.includes(current)) { const o = document.createElement("option"); o.value = current; o.textContent = `${current} (custom)`; o.selected = true; select.append(o); }
              select.disabled = false; select.onchange = () => { target[fspec.key] = select.value; };
            }).catch(() => {});
          } else if (fspec.type === "combo") {
            const select = makeCombo(fspec, fspec.options, value); select.onchange = () => { target[fspec.key] = select.value; }; field.append(select);
            body.append(field);
          } else if (fspec.type === "boolean") {
            field.style.cssText = "display:flex;flex-direction:row;align-items:center;gap:6px;margin:7px 0;font-size:11px"; const input = document.createElement("input"); input.type = "checkbox"; input.checked = Boolean(value); input.onchange = () => { target[fspec.key] = input.checked; }; field.append(input, title);
            body.append(field);
          } else {
            const input = document.createElement("input"); input.type = fspec.type === "number" ? "number" : "text"; input.value = String(value ?? "");
            if (fspec.type === "number") { if (fspec.min != null) input.min = fspec.min; if (fspec.max != null) input.max = fspec.max; if (fspec.step != null) input.step = fspec.step; }
            input.onchange = () => { target[fspec.key] = fspec.type === "number" ? Number(input.value) : input.value; };
            field.append(input);
            body.append(field);
          }
        }
        const actions = document.createElement("div"); actions.style.cssText = "display:flex;justify-content:flex-end;gap:6px;margin-top:12px";
        const cancel = document.createElement("button"); cancel.textContent = "Cancel"; cancel.onclick = () => overlay.remove();
        const save = document.createElement("button"); save.textContent = "Save"; save.onclick = () => { emit(); overlay.remove(); render(); };
        actions.append(cancel, save); modal.append(actions);
        overlay.onclick = event => { if (event.target === overlay) overlay.remove(); };
        overlay.append(modal); document.body.append(overlay);
      };
      for (const stage of state.postprocess_recipe) { const label = ppStageLabel(stage.id); const pill = document.createElement("div"); pill.className = "ds-h3-pp-btn"; pill.setAttribute("role", "button"); pill.setAttribute("tabindex", "0"); pill.classList.toggle("active", Boolean(stage.enabled)); pill.title = `Toggle ${label}`; pill.append(document.createTextNode(label)); const toggleStage = () => { stage.enabled = !stage.enabled; emit(); render(); }; pill.onclick = (event) => { if (event.target.closest(".ds-h3-pp-burger")) return; toggleStage(); }; pill.onkeydown = (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (event.target.closest(".ds-h3-pp-burger")) return; toggleStage(); } }; const burger = document.createElement("button"); burger.type = "button"; burger.className = "ds-h3-pp-burger"; burger.textContent = "☰"; burger.title = `Advanced options for ${label}`; burger.onclick = (event) => { event.stopPropagation(); openStageSettings(label, stage); }; pill.append(burger); postprocessPanel.append(pill); }
      const ppToken = (s) => {
        switch (s.id) {
          case "frame_interpolation": return `Frame Interpolation ×${Number(s.factor) || 2}`;
          case "torch_resize": {
            const target = (s.size_mode === "Target resolution" && s.width && s.height) ? `${s.width}×${s.height}` : `×${s.scale_multiplier || 1}`;
            return `Resize ${s.interpolation || "Lanczos"} ${target}`;
          }
          case "model_upscale": return `Model Upscale ${(String(s.model_name || "").split("/").pop()) || "model"}`;
          case "rtx_refiner": {
            const mode = String(s.upscale || "VSR");
            return mode === "Off" ? "Refiner (no upscale)" : `Refiner ${mode}${s.upscale_quality && s.upscale_quality !== "Ultra" ? " " + s.upscale_quality.toLowerCase() : ""}`;
          }
          case "watermark": return `Watermark ${s.position || "bottom-right"}`;
          default: return s.id.replaceAll("_", " ");
        }
      };
      const ppActive = state.postprocess_recipe.filter(s => s.enabled);
      const ppSummary = document.createElement("span"); ppSummary.className = "ds-h3-pp-summary-line"; ppSummary.style.cssText = "width:100%;box-sizing:border-box;text-align:left;font-size:11px;color:#7ee19d;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px";
      if (ppActive.length) { ppSummary.textContent = ppActive.map(ppToken).join(" → "); } else { ppSummary.style.color = "#5a6b78"; ppSummary.textContent = "post-processing off"; }
      postprocessInfo = ppSummary;
      const postprocessTitle = document.createElement("span"); postprocessTitle.textContent = "Post processing:"; postprocessTitle.style.cssText = "font-size:11px;color:#9fb3c2;font-weight:600;margin-right:2px"; postprocessPanel.prepend(postprocessTitle);
      // Optimization pills: Comfy Kitchen INT8 attention + MiniMax H3 residual cache.
      // Both patch the H3 model inside the Director's internal (Image Inpaint) executor,
      // so they are applied at runtime rather than wired as standalone graph nodes.
      const exec = state.internal_execution; exec.cache = exec.cache && typeof exec.cache === "object" ? exec.cache : { enabled: false, reuse_threshold: 0.05, start_percent: 0.15, end_percent: 0.90, max_steps: 2, device: "auto", verbose: false };
      optimizationPanel = document.createElement("div"); optimizationPanel.className = "ds-h3-optimization-row"; optimizationPanel.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-wrap:wrap;align-items:center;gap:5px;padding:4px 0 0;border-top:1px solid #344452;margin-top:2px";
      const optimizationTitle = document.createElement("span"); optimizationTitle.textContent = "Optimizations:"; optimizationTitle.style.cssText = "font-size:11px;color:#9fb3c2;font-weight:600;margin-right:2px"; optimizationPanel.append(optimizationTitle);
      const makeOptPill = (label, active, onToggle, onSettings) => { const pill = document.createElement("div"); pill.className = "ds-h3-pp-btn"; pill.setAttribute("role", "button"); pill.setAttribute("tabindex", "0"); pill.classList.toggle("active", active); pill.title = `Toggle ${label}`; pill.append(document.createTextNode(label)); const toggleOpt = () => { onToggle(); emit(); render(); }; pill.onclick = (event) => { if (event.target.closest(".ds-h3-pp-burger")) return; toggleOpt(); }; pill.onkeydown = (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (event.target.closest(".ds-h3-pp-burger")) return; toggleOpt(); } }; if (onSettings) { const burger = document.createElement("button"); burger.type = "button"; burger.className = "ds-h3-pp-burger"; burger.textContent = "☰"; burger.title = `Configure ${label}`; burger.onclick = (event) => { event.stopPropagation(); onSettings(); }; pill.append(burger); } optimizationPanel.append(pill); };
      makeOptPill("Comfy Kitchen Attention", Boolean(exec.comfy_kitchen_attention), () => { exec.comfy_kitchen_attention = !exec.comfy_kitchen_attention; }, null);
      makeOptPill("MiniMax H3 Cache", Boolean(exec.cache.enabled), () => { exec.cache.enabled = !exec.cache.enabled; }, () => { openStageSettings("MiniMax H3 Cache", exec.cache, PP_H3_CACHE_FIELDS); });
      // Sampling panel: sampler/scheduler dropdowns + steps/shift number inputs,
      // all bound to state.internal_execution (the backend already honors these).
      const updateExec = (patch) => mutate(s => { const next = { ...s.internal_execution, ...patch }; if ("steps" in next) next.steps = Math.max(1, Math.min(1000, Math.trunc(next.steps || 25))); if ("shift_video" in next) next.shift_video = Math.max(0, Number(next.shift_video) || 0); if ("shift_audio" in next) next.shift_audio = Math.max(0, Number(next.shift_audio) || 0); s.internal_execution = next; });
      samplingPanel = document.createElement("div"); samplingPanel.className = "ds-h3-sampling-row"; samplingPanel.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-wrap:wrap;align-items:end;gap:8px;padding:6px 0 0;border-top:1px solid #344452;margin-top:4px";
      const samplingTitle = document.createElement("span"); samplingTitle.textContent = "Sampling:"; samplingTitle.style.cssText = "font-size:11px;color:#9fb3c2;font-weight:600;width:100%"; samplingPanel.append(samplingTitle);
      const makeSamplingSelect = (label, value, options) => { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;font-weight:600;letter-spacing:.4px;color:#8fb3d6;text-transform:uppercase"; const cap = document.createElement("span"); cap.textContent = label; field.append(cap); const select = document.createElement("select"); select.style.cssText = "min-width:160px;padding:3px 6px;background:#16283a;border:1px solid #2f5478;border-radius:4px;color:#d6ebff;font:12px system-ui,sans-serif"; options.forEach(opt => { const o = document.createElement("option"); o.value = opt; o.textContent = opt; if (opt === value) o.selected = true; select.append(o); }); if (!options.includes(value)) { const o = document.createElement("option"); o.value = value; o.textContent = `${value} (current)`; o.selected = true; select.append(o); } select.onchange = () => updateExec({ [label === "SAMPLER" ? "sampler" : "scheduler"]: select.value }); field.append(select); samplingPanel.append(field); return select; };
      const samplerSelect = makeSamplingSelect("SAMPLER", exec.sampler, SAMPLER_OPTIONS);
      const schedulerSelect = makeSamplingSelect("SCHEDULER", exec.scheduler, SCHEDULER_OPTIONS);
      const makeSamplingNumber = (label, value, key, step, min, max, fallback) => { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;font-weight:600;letter-spacing:.4px;color:#8fb3d6;text-transform:uppercase"; const cap = document.createElement("span"); cap.textContent = label; field.append(cap); const input = document.createElement("input"); input.type = "number"; input.step = step; input.min = min; input.max = max; input.value = value; input.style.cssText = "min-width:74px;padding:3px 6px;background:#16283a;border:1px solid #2f5478;border-radius:4px;color:#d6ebff;font:12px system-ui,sans-serif"; input.onchange = () => updateExec({ [key]: Number(input.value) || fallback }); field.append(input); samplingPanel.append(field); return input; };
      makeSamplingNumber("STEPS", exec.steps, "steps", 1, 1, 1000, 25);
      makeSamplingNumber("SHIFT V", exec.shift_video, "shift_video", 0.5, 0, 100, 11);
      makeSamplingNumber("SHIFT A", exec.shift_audio, "shift_audio", 0.5, 0, 100, 4);
      const samplingDisabled = hasExternalSampling();
      if (samplingDisabled) {
        samplingPanel.classList.add("disabled");
        samplingPanel.querySelectorAll("select, input, button").forEach(el => { el.disabled = true; });
        const note = document.createElement("div"); note.className = "ds-h3-small ds-h3-ext-note";
        note.style.cssText = "width:100%";
        note.textContent = "External sampling connected — local sampling fields are disabled (external values win).";
        samplingPanel.append(note);
      }
      modeGroup.append(samplingPanel);
      const saveSettings = state.internal_execution.save;
      const outputKind = mode() === "Image Inpaint" ? "image" : "video";
      saveSettings.output_kind = outputKind;
      const saveTitle = document.createElement("strong"); saveTitle.textContent = "Preview & Output"; saveTitle.style.cssText = "font-size:12px;color:#d5e6f2";
      const outputSettingsHint = () => { const flag = value => value ? "on" : "off"; const parts = [`prefix ${saveSettings.filename_prefix || "Director"}`, `save ${flag(saveSettings.save_output !== false)}`, `workflow ${flag(saveSettings.save_workflow !== false)}`]; if (outputKind === "image") parts.push(String(saveSettings.file_format || "webp").toUpperCase(), `compression ${saveSettings.compression ?? 15}`); else parts.push(saveSettings.codec || "Auto", saveSettings.container || "Auto", saveSettings.bit_depth || "Auto", `quality ${saveSettings.quality ?? 20}`, `pingpong ${flag(saveSettings.pingpong)}`, `crop to audio ${flag(saveSettings.crop_to_audio)}`, `${saveSettings.audio_codec || "Auto"} ${saveSettings.audio_bitrate || "192k"}`, `first frame ${flag(saveSettings.save_first_frame)}`, `last frame ${flag(saveSettings.save_last_frame)}`); return parts.join(" · "); };
      const saveHint = document.createElement("span"); saveHint.textContent = outputSettingsHint(); saveHint.style.cssText = "font-size:11px;line-height:1.35;color:#7ee19d;overflow-wrap:anywhere";
      const saveOptionFields = [...(outputKind === "image" ? PP_IMAGE_SAVE_FIELDS : PP_VIDEO_SAVE_FIELDS), ...PREVIEW_OPTION_FIELDS, { key: "save_output", type: "boolean", default: true }, { key: "save_workflow", type: "boolean", default: true }];
      const saveHeader = document.createElement("div"); saveHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px";
      const saveMenu = document.createElement("button"); saveMenu.type = "button"; saveMenu.className = "ds-h3-res-btn"; saveMenu.textContent = "☰"; saveMenu.title = "Preview & Output options"; saveMenu.style.cssText = "width:30px;min-height:26px;padding:0;justify-content:center;gap:0"; saveMenu.onclick = () => openStageSettings("Preview & Output options", saveSettings, saveOptionFields); saveHeader.append(saveTitle, saveMenu);
      const savedPreview = node.__dasiwaH3SavedPreview;
      // Live step-preview slot: shows per-step frames streamed by the backend
      // OUTER_SAMPLE wrapper (dasiwa_director_v2_preview events) until the
      // final published output replaces it. Hidden until a run streams frames.
      const stepPreviewSlot = document.createElement("div"); stepPreviewSlot.className = "ds-h3-step-preview"; stepPreviewSlot.style.cssText = "display:none;flex-direction:column;gap:4px;padding:6px;background:#080d11;border:1px solid #2d4658;border-radius:4px";
      const stepPreviewMedia = document.createElement("img"); stepPreviewMedia.style.cssText = "display:block;width:100%;max-height:220px;object-fit:contain;background:#000"; stepPreviewMedia.setAttribute("alt", "Live step preview");
      const stepPreviewCaption = document.createElement("span"); stepPreviewCaption.style.cssText = "font-size:10px;color:#7ee19d"; stepPreviewCaption.textContent = "";
      stepPreviewSlot.append(stepPreviewMedia, stepPreviewCaption);
      node.__dasiwaH3ShowStepPreview = (payload) => {
        stepPreviewSlot.style.display = "flex";
        stepPreviewSlot.parentElement.style.order = "-1";
        const isVideo = payload.mime === "video/mp4";
        const isAnimWebp = payload.mime === "image/webp" && (payload.frames > 1 || payload.fps);
        let target = stepPreviewMedia;
        if (isVideo || isAnimWebp) {
          if (target.tagName !== "VIDEO") { const v = document.createElement("video"); v.muted = true; v.loop = true; v.playsInline = true; v.controls = isVideo; v.style.cssText = "display:block;width:100%;max-height:220px;object-fit:contain;background:#000"; target.replaceWith(v); target = v; }
        } else if (target.tagName === "VIDEO") { const i = document.createElement("img"); i.alt = "Live step preview"; i.style.cssText = "display:block;width:100%;max-height:220px;object-fit:contain;background:#000"; target.replaceWith(i); target = i; }
        target.src = `data:${payload.mime};base64,${payload.image}`;
        if (target.tagName === "VIDEO") target.play?.();
        stepPreviewCaption.textContent = payload.total ? `step ${payload.step}/${payload.total}` : `step ${payload.step}`;
        stepPreviewSlot.parentElement?.appendChild(stepPreviewSlot);
      };
      saveNodePanel.append(saveHeader, saveHint, stepPreviewSlot);
      if (savedPreview?.filename) { const isVideo = String(savedPreview.format || "").startsWith("video/") || savedPreview.container; const media = document.createElement(isVideo ? "video" : "img"); media.src = savedViewUrl(savedPreview); media.title = "Latest Director output"; media.style.cssText = "display:block;width:100%;max-height:180px;object-fit:contain;background:#000;cursor:pointer"; if (isVideo) { media.controls = true; media.muted = true; media.loop = true; media.playsInline = true; } else { media.alt = savedPreview.filename; } media.onclick = () => window.open(media.src, "_blank", "noopener,noreferrer"); const previewFrame = document.createElement("div"); previewFrame.className = "ds-h3-save-preview"; previewFrame.style.cssText = "min-height:220px;flex:1;display:grid;place-items:center;background:#080d11;border:1px solid #2d4658;border-radius:4px;overflow:hidden"; previewFrame.append(media); saveNodePanel.append(previewFrame); } else if (!savedPreview) { const previewFrame = document.createElement("div"); previewFrame.className = "ds-h3-save-preview"; previewFrame.style.cssText = "min-height:220px;flex:1;display:grid;place-items:center;background:#080d11;border:1px solid #2d4658;border-radius:4px;overflow:hidden"; const empty = document.createElement("span"); empty.textContent = "Preview appears after the Director publishes an image or video"; empty.style.cssText = "padding:8px;text-align:center;font-size:11px;color:#9fb3c2"; previewFrame.append(empty); saveNodePanel.append(previewFrame); }
      controlRow.append(modeGroup, saveNodePanel);
      timeline.append(controlRow);
    }

    const resolutionPanel = document.createElement("div"); resolutionPanel.className = "ds-h3-resolution-panel"; resolutionPanel.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-wrap:wrap;align-items:end;gap:6px;padding:2px 0 0;border-top:1px solid #344452;margin-top:2px";
    const settings = resolutionState(); const externalCanvas = hasExternalCanvas(); const currentCanvas = externalCanvas ? ["external", "external"] : (resolveCanvas(settings) || [Number(widthWidget?.value) || 1344, Number(heightWidget?.value) || 768]);
    const updateResolution = patch => mutate(s => { s.resolution = { ...resolutionState(), ...patch }; });
    const closeAllMenus = () => resolutionPanel.querySelectorAll(".ds-h3-res-menu.open").forEach(menu => menu.classList.remove("open"));
    const aspectSwatch = id => { if (!String(id).includes(":")) return null; const [w, h] = String(id).split(":").map(Number); if (!w || !h) return null; const boxSize = 15; const box = document.createElement("span"); box.className = "ds-h3-res-swatch-box"; box.style.width = `${Math.round(w >= h ? boxSize : boxSize * w / h)}px`; box.style.height = `${Math.round(w >= h ? boxSize * h / w : boxSize)}px`; const wrap = document.createElement("span"); wrap.className = "ds-h3-res-swatch"; wrap.title = `Aspect ${id}`; wrap.append(box); return wrap; };
    const inputScalingHint = id => ({ Off: "Skip input scaling — references are sent to MiniMax at their original size.", Auto: "Scale references down only when their short edge exceeds 2048 px; smaller inputs pass through untouched.", Target: "Resize references to the selected Aspect & Resolution box (stretch, aspect not preserved).", Fit: "Scale references to fit inside the target box while preserving their aspect ratio.", "Fill and crop": "Scale references to cover the target box, preserving aspect, then crop the overflow.", "Fit and pad": "Scale references to fit inside the target box, preserving aspect, then pad the empty edges.", "Long side with divisible crop": "Scale so the long side matches the target, then crop to an exact, divisible target size." }[id] || "");
    const addDropdown = (label, value, options, onChange, withAspectSwatches = false, withHints = false, groups = null) => { const field = document.createElement("div"); field.className = "ds-h3-res-field"; field.textContent = label; const control = document.createElement("span"); control.className = "ds-h3-res-control"; const select = document.createElement("select"); select.className = "ds-h3-res-select"; select.setAttribute("aria-hidden", "true"); options.forEach(([id, text]) => { const option = document.createElement("option"); option.value = id; option.textContent = text; select.append(option); }); select.value = value; select.oninput = event => { onChange(event.target.value); syncButton(); syncMenu(); }; const button = document.createElement("button"); button.type = "button"; button.className = "ds-h3-res-btn"; button.setAttribute("aria-haspopup", "listbox"); button.setAttribute("aria-expanded", "false"); const labelSpan = document.createElement("span"); labelSpan.className = "ds-h3-res-label"; const caret = document.createElement("span"); caret.className = "ds-h3-res-caret"; caret.textContent = "▾"; button.append(labelSpan, caret); const menu = document.createElement("div"); const hasGroups = Array.isArray(groups) && groups.length > 0; menu.className = hasGroups ? "ds-h3-res-menu cols" : (options.length >= 6 ? "ds-h3-res-menu grid" : "ds-h3-res-menu"); if (menu.classList.contains("grid")) { const cols = Math.max(3, Math.min(6, Math.round(Math.sqrt(options.length)))); menu.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`; menu.style.width = `${cols * 148}px`; menu.style.maxHeight = "420px"; } if (hasGroups) menu.style.maxHeight = "420px"; menu.setAttribute("role", "listbox"); const syncButton = () => { const current = options.find(([id]) => id === select.value); labelSpan.textContent = current ? current[1] : select.value; const swatch = withAspectSwatches ? aspectSwatch(select.value) : null; const existing = button.querySelector(".ds-h3-res-swatch"); if (existing) existing.remove(); if (swatch) button.insertBefore(swatch, labelSpan); button.title = withHints ? inputScalingHint(select.value) : ""; }; const syncMenu = () => menu.querySelectorAll(".ds-h3-res-item").forEach(item => item.classList.toggle("active", item.dataset.id === select.value)); const makeItem = ([id, text]) => { const item = document.createElement("button"); item.type = "button"; item.className = "ds-h3-res-item"; item.setAttribute("role", "option"); item.dataset.id = id; if (withAspectSwatches) { const swatch = aspectSwatch(id); if (swatch) item.append(swatch); } const itemLabel = document.createElement("span"); itemLabel.className = "ds-h3-res-item-label"; itemLabel.textContent = text; item.append(itemLabel); if (withHints) item.title = inputScalingHint(id); item.onclick = () => { select.value = id; onChange(id); syncButton(); syncMenu(); closeAllMenus(); button.setAttribute("aria-expanded", "false"); }; return item; }; if (hasGroups) { groups.forEach(group => { const col = document.createElement("div"); col.className = "ds-h3-res-col"; if (group.title) { const title = document.createElement("span"); title.className = "ds-h3-res-col-title"; title.textContent = group.title; col.append(title); } (group.items || []).forEach(pair => col.append(makeItem(pair))); menu.append(col); }); } else { options.forEach(pair => menu.append(makeItem(pair))); } button.onclick = () => { const willOpen = !menu.classList.contains("open"); closeAllMenus(); menu.classList.toggle("open", willOpen); button.setAttribute("aria-expanded", String(willOpen)); if (willOpen) { const margin = 8; const rect = control.getBoundingClientRect(); const menuRect = menu.getBoundingClientRect(); const spaceBelow = window.innerHeight - rect.bottom - margin; const spaceAbove = rect.top - margin; menu.dataset.place = menuRect.height > spaceBelow && spaceAbove > spaceBelow ? "up" : "down"; let left = rect.left; if (left + menuRect.width > window.innerWidth - margin) left = Math.max(margin, window.innerWidth - margin - menuRect.width); menu.style.left = `${left - rect.left}px`; } }; syncButton(); syncMenu(); control.append(select, button, menu); field.append(control); resolutionPanel.append(field); };
    const disableWhenExternal = () => { if (externalCanvas) resolutionPanel.querySelectorAll("select,input,button").forEach(control => { control.disabled = true; }); };
    const ratioOf = id => { const [w, h] = String(id).split(":").map(Number); return w && h ? w / h : 0; };
    const aspectItems = { general: [], horizontal: [], square: [], vertical: [] };
    ASPECT_OPTIONS.forEach(pair => { const [id] = pair; if (id === "auto" || id === "custom" || !id.includes(":")) aspectItems.general.push(pair); else { const [w, h] = id.split(":").map(Number); aspectItems[w > h ? "horizontal" : w < h ? "vertical" : "square"].push(pair); } });
    aspectItems.horizontal.sort((a, b) => ratioOf(a[0]) - ratioOf(b[0]));
    aspectItems.vertical.sort((a, b) => ratioOf(a[0]) - ratioOf(b[0]));
    const resNames = Object.keys(RESOLUTION_PRESETS);
    const pVal = n => Number(String(n).replace(/K$/i, "000").replace(/p$/i, ""));
    const mpVal = n => Number((String(n).match(/^([\d.]+)\s*MP/) || [])[1] || 0);
    addDropdown("ASPECT", settings.aspect, ASPECT_OPTIONS, aspect => updateResolution({ aspect }), true, false, [
      { title: "General", items: aspectItems.general },
      { title: "Horizontal", items: aspectItems.horizontal },
      { title: "Square", items: aspectItems.square },
      { title: "Vertical", items: aspectItems.vertical },
    ]);
    addDropdown("RESOLUTION", settings.resolution, [["auto", "Native (ShortEdge 768px)"], ...resNames.map(name => [name, name]), ["custom", "CUSTOM"]], resolution => updateResolution({ resolution }), false, false, [
      { title: "General", items: [["auto", "Native (ShortEdge 768px)"], ["custom", "CUSTOM"]] },
      { title: "###p", items: resNames.filter(n => !/MP/.test(n)).sort((a, b) => pVal(a) - pVal(b)).map(n => [n, n]) },
      { title: "MP", items: resNames.filter(n => /MP/.test(n)).sort((a, b) => mpVal(a) - mpVal(b)).map(n => [n, n]) },
    ]);
    addDropdown("INPUT SCALING", settings.input_scaling, [["Off", "Off"], ["Auto", "Native (ShortEdge 2048px)"], ["Target", "Target · Selected Aspect & Resolution"], ["Fit", "Fit"], ["Fill and crop", "Fill and crop"], ["Fit and pad", "Fit and pad"], ["Long side with divisible crop", "Long side with divisible crop"]], input_scaling => updateResolution({ input_scaling }), false, true);
    const seedControl = document.createElement("div"); seedControl.className = "ds-h3-seed-control"; seedControl.style.cssText = "display:flex;align-items:flex-end;gap:5px;margin-left:8px;padding-left:10px;border-left:1px solid #344452";
    const externalSeed = hasExternalSeed();
    if (externalSeed) { const external = document.createElement("span"); external.textContent = "External seed connected"; external.style.cssText = "font-size:11px;color:#9fb3c2;white-space:nowrap;padding-bottom:4px"; seedControl.append(external); } else {
      const seed = seedWidget(); const controlState = state.seed_control = { mode: "random", last_seed: null, recent: [], ...(state.seed_control || {}) }; const maxSeed = 0xffffffffffffffffn;
      const setSeed = raw => { let value; try { value = BigInt(String(raw).trim()); } catch { return false; } if (value < 0n || value > maxSeed) return false; const text = value.toString(); if (seed) { seed.value = text; seed.callback?.(text); } node.setDirtyCanvas?.(true, true); return text; };
      const currentSeed = () => String(seed?.value ?? 0); if (controlState.mode === "random" && currentSeed() === "0") { const words = crypto.getRandomValues(new Uint32Array(2)); setSeed((BigInt(words[0]) << 32n) | BigInt(words[1])); } const remember = value => { controlState.last_seed = value; controlState.recent = [value, ...controlState.recent.filter(entry => entry !== value)].slice(0, 10); };
      const toggle = document.createElement("div"); toggle.style.cssText = "display:flex;gap:2px";
      for (const modeName of [["random", "Random"], ["fixed", "Fixed"]]) { const button = document.createElement("button"); button.type = "button"; button.className = "ds-h3-res-btn"; button.textContent = modeName[1]; button.style.cssText = `width:58px;padding:3px 0;justify-content:center;gap:0;text-align:center;${controlState.mode === modeName[0] ? "border-color:#7ee19d;color:#7ee19d" : ""}`; button.onclick = () => { controlState.mode = modeName[0]; emit(); render(); }; toggle.append(button); }
      const seedField = document.createElement("label"); seedField.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:150px;font-size:10px;font-weight:600;letter-spacing:.4px;color:#8fb3d6;text-transform:uppercase"; seedField.textContent = "SEED";
      const input = document.createElement("input"); input.type = "text"; input.inputMode = "numeric"; input.className = "ds-h3-res-num"; input.style.textAlign = "center"; input.value = currentSeed(); input.title = "Unsigned 64-bit Director seed";
      input.onchange = event => { if (!setSeed(event.target.value)) event.target.value = currentSeed(); else { event.target.value = currentSeed(); emit(); } }; seedField.append(input);
      const roll = document.createElement("button"); roll.type = "button"; roll.className = "ds-h3-res-btn"; roll.textContent = "New"; roll.title = "Roll a new seed and retain the selected mode"; roll.style.cssText = "width:68px;padding:3px 0;justify-content:center;gap:0;text-align:center;white-space:nowrap"; roll.onclick = () => { const words = crypto.getRandomValues(new Uint32Array(2)); const value = ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString(); setSeed(value); remember(value); emit(); render(); };
      const last = document.createElement("button"); last.type = "button"; last.className = "ds-h3-res-btn"; last.textContent = "Use Last"; last.disabled = !controlState.last_seed; last.style.cssText = "width:68px;padding:3px 0;justify-content:center;gap:0;text-align:center;white-space:nowrap"; last.onclick = () => { if (setSeed(controlState.last_seed)) { controlState.mode = "fixed"; emit(); render(); } };
      const history = document.createElement("details"); history.style.cssText = "position:relative"; const summary = document.createElement("summary"); summary.className = "ds-h3-res-btn"; summary.textContent = "Last 10 seeds"; summary.style.cssText = "padding:3px 6px;justify-content:center;gap:0;text-align:center;white-space:nowrap;cursor:pointer"; history.append(summary); const list = document.createElement("div"); list.style.cssText = "position:absolute;z-index:10;right:0;top:24px;min-width:160px;max-height:180px;overflow:auto;padding:5px;background:#111b23;border:1px solid #344452;border-radius:4px"; for (const value of controlState.recent) { const row = document.createElement("div"); row.style.cssText = "display:flex;gap:4px;align-items:center"; const item = document.createElement("button"); item.type = "button"; item.textContent = value; item.style.cssText = "flex:1;padding:3px 5px;border:0;background:transparent;color:#d5e6f2;text-align:right;cursor:pointer"; item.onclick = () => { if (setSeed(value)) { controlState.mode = "fixed"; emit(); render(); } }; const copy = document.createElement("button"); copy.type = "button"; copy.textContent = "Copy"; copy.title = "Copy seed"; copy.style.cssText = "padding:2px 5px;font-size:10px"; copy.onclick = event => { event.stopPropagation(); navigator.clipboard.writeText(value).then(() => { copy.textContent = "✓"; setTimeout(() => { copy.textContent = "Copy"; }, 900); }); }; row.append(item, copy); list.append(row); } if (!controlState.recent.length) list.textContent = "No previous seeds"; history.append(list);
      seedControl.append(seedField, toggle, roll, last, history);
    }
    resolutionPanel.append(seedControl);
    if (settings.aspect === "custom") { for (const [name, key] of [["W", "custom_aspect_w"], ["H", "custom_aspect_h"]]) { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#9fb3c2;width:56px"; field.textContent = `RATIO ${name}`; const input = document.createElement("input"); input.type = "number"; input.min = "1"; input.step = "1"; input.className = "ds-h3-res-num"; input.value = settings[key]; input.onchange = event => updateResolution({ [key]: Math.max(1, Number(event.target.value) || 1) }); field.append(input); resolutionPanel.append(field); } }
    if (settings.resolution === "custom") { addDropdown("CUSTOM", settings.custom_mode || "mp", [["mp", "Megapixels"], ["fixed", "Fixed pixels"]], custom_mode => updateResolution({ custom_mode })); const customKey = (settings.custom_mode || "mp") === "mp" ? "custom_mp" : "custom_width"; if (customKey === "custom_mp") { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#9fb3c2;width:76px"; field.textContent = "MP"; const input = document.createElement("input"); input.type = "number"; input.min = ".01"; input.step = ".01"; input.className = "ds-h3-res-num"; input.value = settings.custom_mp; input.onchange = event => updateResolution({ custom_mp: Number(event.target.value) || settings.custom_mp }); field.append(input); resolutionPanel.append(field); } else { const dimensions = document.createElement("div"); dimensions.className = "ds-h3-fixed-dimensions"; dimensions.style.cssText = "display:flex;gap:6px;align-items:end"; const dimensionField = (label, value, onChange) => { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#9fb3c2;width:76px"; field.textContent = label; const input = document.createElement("input"); input.type = "number"; input.min = "16"; input.step = "16"; input.className = "ds-h3-res-num"; input.value = value; input.onchange = event => onChange(Number(event.target.value) || value); field.append(input); return field; }; const widthField = dimensionField("WIDTH", settings.custom_width, custom_width => updateResolution({ custom_width })); const heightField = dimensionField("HEIGHT", settings.custom_height, custom_height => updateResolution({ custom_height })); dimensions.append(widthField, heightField); resolutionPanel.append(dimensions); } }
    disableWhenExternal();
    const readout = document.createElement("span"); readout.style.cssText = "font-size:11px;color:#7ee19d;white-space:normal"; readout.textContent = externalCanvas ? "External width/height overwrite · Director sizing and input scaling disabled" : `${settings.aspect === "auto" ? "Auto 768px" : settings.aspect} · ${settings.resolution === "auto" ? "Auto 768px" : settings.resolution} · ${currentCanvas[0]} × ${currentCanvas[1]} · 32px H3 grid`; modeGroup.append(resolutionPanel, postprocessPanel, optimizationPanel); const infoFeed = document.createElement("div"); infoFeed.className = "ds-h3-info-feed"; infoFeed.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:5px;padding:8px;background:#0d1217;border:1px solid #344452;border-radius:6px"; infoFeed.append(readout, postprocessInfo); timeline.append(infoFeed);

    const lengthWidget = node.widgets?.find(w => w.name === "duration"); const timelineSeconds = Math.max(1, Number(lengthWidget?.value) || 5); lastTimelineLength = timelineSeconds;

    ensureLayout();
    const track = document.createElement("div"); track.className = "ds-h3-track"; track.addEventListener("pointerdown", () => timeline.focus());
    const trackInner = document.createElement("div"); trackInner.className = "ds-h3-track-inner";
    trackInner.onclick = event => { if (event.target !== trackInner) return; const rect = trackInner.getBoundingClientRect(); insertAt = Math.max(0, Math.round(((event.clientX - rect.left) / scale) * 4) / 4); status.textContent = `Insert cursor: ${insertAt.toFixed(2)}s · choose + image, + video, + audio, or + text.`; trackInner.style.setProperty("--insert-x", `${insertAt * scale}px`); };
    const laneTypes = ["Image/Video", "audio"]; const laneHeight = 120; const trackEntries = displayedItems(); const scale = 48; const audioSlotWidth = sourceDuration => Math.round(Math.max(180, Math.min(420, 96 * Math.log2(Math.max(2, sourceDuration) + 1)))); const slotWidthFor = item => item && (item.type === "audio" || item.type === "video") ? audioSlotWidth(Number(item.source_duration) || item.duration) : 112; const laneNameFor = item => laneForItem(item) === "audio" ? "audio" : "Image/Video"; const slotItem = (lane, slot) => trackEntries.find(item => laneNameFor(item) === lane && item.slot === slot); const slotLeft = (lane, slot) => { let left = 6; for (let index = 0; index < slot; index += 1) left += slotWidthFor(slotItem(lane, index)) + 8; return left; }; const acceptLaneDrop = async (event, targetLane) => { event.preventDefault(); event.stopPropagation(); trackInner.classList.remove("over"); const rect = trackInner.getBoundingClientRect(); insertAt = Math.max(0, Math.round(((event.clientX - rect.left) / scale) * 4) / 4); trackInner.style.setProperty("--insert-x", `${insertAt * scale}px`); for (const file of event.dataTransfer.files || []) await acceptFile(file, targetLane); }; const visualSlotCount = () => mode() === "T2VA" ? 0 : (mode() === "I2VA" || mode() === "Image Inpaint") ? 1 : mode() === "L2VA" ? 2 : mode() === "FL2VA" ? 2 : MAX.image + MAX.video; const slotExtent = lane => Array.from({ length: lane === "audio" ? MAX.audio : visualSlotCount() }, (_, index) => slotWidthFor(slotItem(lane === "audio" ? "audio" : "Image/Video", index)) + 8).reduce((sum, width) => sum + width, 6); trackInner.style.width = `${Math.max(240, slotExtent("Image/Video"), slotExtent("audio"))}px`; trackInner.style.height = `${laneTypes.length * laneHeight}px`; trackInner.style.background = `repeating-linear-gradient(0deg, transparent 0, transparent ${laneHeight - 1}px, #344452 ${laneHeight - 1}px, #344452 ${laneHeight}px), repeating-linear-gradient(90deg,#111a21 0,#111a21 49px,#1b2933 50px)`; const ruler = document.createElement("div"); ruler.className = "ds-h3-ruler"; for (let second = 0; second <= timelineSeconds; second++) { if (second % 2 !== 0) continue; const mark = document.createElement("div"); mark.style.position = "absolute"; mark.style.left = `${second * scale}px`; mark.style.bottom = "0"; mark.style.fontSize = "9px"; mark.style.color = "#8fa3b2"; mark.textContent = `${second}s`; ruler.appendChild(mark); } track.append(ruler);
    const lanes = new Map(); laneTypes.forEach((type, index) => { const lane = document.createElement("div"); const targetLane = type === "audio" ? "audio" : "visual"; const m = mode(); const audioDisabledByMode = m === "T2VA" || m === "I2VA" || m === "FL2VA" || m === "L2VA" || m === "Image Inpaint"; const supported = !(targetLane === "audio" && audioDisabledByMode); const slotCount = targetLane === "audio" ? MAX.audio : visualSlotCount(); lane.className = `ds-h3-timeline-lane ${targetLane}${selectedLane === targetLane ? " selected" : ""}${supported ? "" : " disabled"}`; lane.style.top = `${index * laneHeight}px`; lane.onclick = () => { if (!supported) { setStatus(m === "T2VA" ? "T2VA has no input lanes." : audioDisabledByMode ? "This mode does not support audio references." : "", true); return; } selectedLane = targetLane; setStatus(`${type === "audio" ? "Audio" : "Image/Video"} lane selected. Paste files here with Ctrl+V.`); render(); }; lane.ondragover = event => { if ([...event.dataTransfer.items].some(item => item.kind === "file")) { event.preventDefault(); if (supported) trackInner.classList.add("over"); } }; lane.ondragleave = () => trackInner.classList.remove("over"); lane.ondrop = event => { if (!supported) { event.preventDefault(); event.stopPropagation(); setStatus(audioDisabledByMode ? "This mode does not support audio references." : "", true); return; } selectedLane = targetLane; acceptLaneDrop(event, targetLane); }; const label = document.createElement("span"); label.className = "ds-h3-lane-label"; label.textContent = `${type}${selectedLane === targetLane ? " · selected" : ""}`; label.style.display = hiddenLanes.has(type) ? "none" : "block"; lane.append(label); for (let slotIndex = 0; slotIndex < slotCount; slotIndex += 1) { if (slotItem(type, slotIndex)) continue; const slot = document.createElement("span"); slot.className = "ds-h3-empty-slot"; slot.textContent = "+"; slot.style.left = `${slotLeft(type, slotIndex)}px`; slot.style.width = `${slotWidthFor(null)}px`; if (m === "L2VA" && targetLane === "visual" && slotIndex === 0) { const lockIcon = document.createElement("span"); lockIcon.className = "ds-h3-lock-icon"; lockIcon.textContent = "🔒"; lockIcon.style.position = "absolute"; lockIcon.style.right = "4px"; lockIcon.style.top = "4px"; lockIcon.style.fontSize = "12px"; slot.style.position = "relative"; slot.appendChild(lockIcon); } if (!m.includes("T2VA") && supported) { slot.onclick = event => { event.stopPropagation(); selectedLane = targetLane; const acceptTypes = targetLane === "audio" ? ["audio"] : ["image","video"]; const accepts = acceptTypes.map(t => t === "image" ? "image/*" : t === "video" ? "video/*" : "audio/*").join(","); fileInput.accept = accepts; fileInput.click(); }; } lane.append(slot); } lanes.set(type, lane); trackInner.append(lane); });
    trackEntries.forEach(item => { const clip = document.createElement("div"); clip.className = `ds-h3-clip ${item.type} ${selectedId === item.id ? "selected" : ""}${isLockedSlot(item) ? " locked" : ""}`; clip.textContent = mediaLabel(item); clip.title = item.type === "text" ? String(item.value || "") : String(item.value || ""); clip.style.display = "block"; const bgSrc = item.type === "image" ? viewUrl(item.value) : item.thumbnail || null; if (bgSrc) { clip.style.backgroundImage = `linear-gradient(90deg, rgba(20,35,45,.78), rgba(20,35,45,.5)), url("${bgSrc}")`; clip.style.backgroundSize = "cover"; clip.style.backgroundPosition = "center"; } item.slot = Number.isInteger(item.slot) ? item.slot : trackEntries.filter(x => laneNameFor(x) === laneNameFor(item)).indexOf(item); item.start = item.slot; item.duration = Number.isFinite(item.duration) ? item.duration : 1; const visualStart = item.start; const sourceDuration = Number(item.source_duration) || item.duration; const clipWidth = slotWidthFor(item); clip.style.left = `${slotLeft(laneNameFor(item), item.slot)}px`; clip.style.top = "7px"; clip.style.width = `${clipWidth}px`; clip.style.setProperty("--clip-width", `${clipWidth}px`); clip.dataset.slot = String(item.slot); const identity = document.createElement("span"); identity.className = "ds-h3-clip-identity"; const typePosition = trackEntries.filter(entry => entry.type === item.type).sort((a, b) => a.slot - b.slot).indexOf(item) + 1; identity.textContent = `${mediaReferenceName(item.type)} ${typePosition}`; clip.append(identity); if (isLockedSlot(item)) { const lockIcon = document.createElement("span"); lockIcon.className = "ds-h3-lock-icon"; lockIcon.textContent = "🔒"; clip.append(lockIcon); } if (item.type === "video") { const streamControls = document.createElement("span"); streamControls.className = "ds-h3-video-stream-controls"; const currentMediaMode = ["video", "audio", "video_audio"].includes(item.media_mode) ? item.media_mode : "video"; [["video", "V", "Video only"], ["audio", "A", "Audio only"], ["video_audio", "V+A", "Video + embedded audio"]].forEach(([value, label, title]) => { const button = document.createElement("button"); button.textContent = label; button.title = title; button.classList.toggle("active", value === currentMediaMode); button.onpointerdown = event => event.stopPropagation(); button.onclick = event => { event.stopPropagation(); mutate(s => { const target = s.items.find(x => x.id === item.id); if (target) target.media_mode = value; }); }; streamControls.append(button); }); clip.append(streamControls); const videoScale = document.createElement("span"); videoScale.className = "ds-h3-video-scale"; videoScale.title = "Full source-duration scale"; clip.append(videoScale); } if (selectedId === item.id) { const close = document.createElement("button"); close.className = "ds-h3-clip-close"; close.textContent = "×"; close.title = `Remove selected ${item.type}`; close.onclick = event => { event.stopPropagation(); remove(item.id); }; clip.append(close); }
      const cropReadout = document.createElement("span"); cropReadout.className = "ds-h3-crop-readout"; if (item.type === "video" || item.type === "audio") { cropReadout.textContent = `crop ${Number(item.trim_start || 0).toFixed(2)}s–${item.trim_end == null ? "end" : Number(item.trim_end).toFixed(2) + "s"}`; clip.append(cropReadout); } if (item.type === "video") { for (const edge of ["start", "end"]) { const marker = document.createElement("span"); marker.className = `ds-h3-crop-marker ${edge}`; marker.style.left = `${(Number(edge === "start" ? item.trim_start || 0 : item.trim_end ?? sourceDuration) / sourceDuration) * 100}%`; clip.append(marker); } } if (item.type === "audio") { const peaks = Array.isArray(item.waveform_peaks) ? item.waveform_peaks : []; const waveform = document.createElement("canvas"); waveform.className = "ds-h3-waveform"; waveform.width = 720; waveform.height = 160; waveform.title = peaks.length ? "Full audio waveform; crop markers show the selected reference window" : "Decoding audio waveform…"; const context = waveform.getContext("2d"); if (context && peaks.length) { const center = waveform.height / 2; context.fillStyle = "rgba(126, 225, 157, .8)"; for (let x = 0; x < waveform.width; x += 1) { const peakIndex = Math.min(peaks.length - 1, Math.floor((x / waveform.width) * peaks.length)); const amplitude = (Math.log1p(9 * peaks[peakIndex]) / Math.log(10)) * (waveform.height - 16) * .45; context.fillRect(x, center - amplitude, 1, amplitude * 2); } } const cropStartMarker = document.createElement("span"); cropStartMarker.className = "ds-h3-audio-crop-marker start"; const cropEndMarker = document.createElement("span"); cropEndMarker.className = "ds-h3-audio-crop-marker end"; const updateAudioCropMarkers = () => { const startPercent = Math.max(0, Math.min(100, (Number(item.trim_start || 0) / sourceDuration) * 100)); const endPercent = Math.max(startPercent, Math.min(100, (Number(item.trim_end ?? sourceDuration) / sourceDuration) * 100)); cropStartMarker.style.left = `${startPercent}%`; cropEndMarker.style.left = `${endPercent}%`; }; updateAudioCropMarkers(); clip.append(waveform, cropStartMarker, cropEndMarker); }
      const resize = (edge, event) => { event.stopPropagation(); clip.setPointerCapture?.(event.pointerId); const origin = event.clientX; const start = item.start; const duration = item.duration; const sourceDuration = Number(item.source_duration) || duration; const minimumReferenceDuration = Math.min(2, sourceDuration); const cropStartAtDrag = Number(item.trim_start) || 0; const cropEndAtDrag = Number(item.trim_end) || sourceDuration; const onMove = moveEvent => { const delta = (item.type === "audio" || item.type === "video") ? ((moveEvent.clientX - origin) / clipWidth) * sourceDuration : (moveEvent.clientX - origin) / scale; if (edge === "left") { if (item.type === "video" || item.type === "audio") { item.trim_start = Math.min(cropEndAtDrag - minimumReferenceDuration, Math.max(0, Math.round((cropStartAtDrag + delta) * 4) / 4)); item.duration = Math.min(15, Math.max(minimumReferenceDuration, Math.round((cropEndAtDrag - item.trim_start) * 4) / 4)); item.trim_end = cropEndAtDrag; } else { const nextStart = Math.max(0, Math.round((start + delta) * 4) / 4); const end = start + duration; item.start = Math.min(nextStart, end - 0.25); item.duration = Math.max(0.25, Math.round((end - item.start) * 4) / 4); item.trim_start = item.start; } } else if (item.type === "video" || item.type === "audio") { item.trim_end = Math.min(sourceDuration, Math.max(cropStartAtDrag + minimumReferenceDuration, Math.round((cropEndAtDrag + delta) * 4) / 4)); item.duration = Math.min(15, Math.max(minimumReferenceDuration, Math.round((item.trim_end - cropStartAtDrag) * 4) / 4)); item.trim_end = cropStartAtDrag + item.duration; } else { item.duration = Math.max(0.25, Math.round((duration + delta) * 4) / 4); item.trim_end = item.start + item.duration; } if (item.type === "video" || item.type === "audio") { cropReadout.textContent = `crop ${Number(item.trim_start || 0).toFixed(2)}s–${Number(item.trim_end).toFixed(2)}s / ${sourceDuration.toFixed(2)}s`; setStatus(`${edge === "left" ? "Crop start" : "Crop end"}: ${cropReadout.textContent}`); } if (item.type === "video" || item.type === "audio") { const startPercent = Math.max(0, Math.min(100, (Number(item.trim_start || 0) / sourceDuration) * 100)); const endPercent = Math.max(startPercent, Math.min(100, (Number(item.trim_end ?? sourceDuration) / sourceDuration) * 100)); const [cropStartMarker, cropEndMarker] = clip.querySelectorAll(".ds-h3-crop-marker, .ds-h3-audio-crop-marker"); if (cropStartMarker) cropStartMarker.style.left = `${startPercent}%`; if (cropEndMarker) cropEndMarker.style.left = `${endPercent}%`; } clip.style.left = `${slotLeft(laneNameFor(item), item.slot)}px`; clip.style.width = `${clipWidth}px`; }; const onUp = () => { clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); if (item._block) { item._block.start = item.start; item._block.duration = item.duration; } mutate(() => {}); }; clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); };
      const leftGrip = document.createElement("span"); leftGrip.className = "ds-h3-grip left"; leftGrip.onpointerdown = event => resize("left", event); const rightGrip = document.createElement("span"); rightGrip.className = "ds-h3-grip right"; rightGrip.onpointerdown = event => resize("right", event); clip.append(leftGrip, rightGrip); if (item.type === "video" || item.type === "audio") { leftGrip.style.display = "none"; rightGrip.style.display = "none"; clip.querySelectorAll(".ds-h3-crop-marker, .ds-h3-audio-crop-marker").forEach(marker => { marker.onpointerdown = event => resize(marker.classList.contains("start") ? "left" : "right", event); }); }
      const editBtn = document.createElement("button"); editBtn.textContent = "☰"; editBtn.className = "ds-h3-edit-btn"; editBtn.title = "Open preview and details"; editBtn.onclick = event => { event.stopPropagation(); openPreview(item); }; clip.append(editBtn);
      clip.onclick = event => { if (event.target !== clip) return; selectedId = item.id; render(); }; clip.onpointerdown = event => { selectedId = item.id; if (event.target !== clip) return; event.stopPropagation(); clip.setPointerCapture?.(event.pointerId); const origin = event.clientX; const originalLeft = slotLeft(laneNameFor(item), item.slot); const lane = laneNameFor(item); const slotCount = lane === "audio" ? MAX.audio : visualSlotCount(); let dragged = false; const onMove = moveEvent => { dragged ||= Math.abs(moveEvent.clientX - origin) >= 4; if (dragged) clip.style.left = `${originalLeft + moveEvent.clientX - origin}px`; }; const onUp = moveEvent => { clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); if (!dragged) return; const rect = trackInner.getBoundingClientRect(); const x = moveEvent.clientX - rect.left; const targetSlot = Array.from({ length: slotCount }, (_, slot) => slot).reduce((nearest, slot) => Math.abs((slotLeft(lane, slot) + slotWidthFor(slotItem(lane, slot)) / 2) - x) < Math.abs((slotLeft(lane, nearest) + slotWidthFor(slotItem(lane, nearest)) / 2) - x) ? slot : nearest, 0); mutate(s => { const moved = s.items.find(x => x.id === item.id); if (!moved) return; const occupant = s.items.find(x => x.id !== moved.id && laneForItem(x) === laneForItem(moved) && x.slot === targetSlot); const previousSlot = moved.slot; moved.slot = targetSlot; moved.start = targetSlot; if (occupant) { occupant.slot = previousSlot; occupant.start = previousSlot; } }); }; clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); }; lanes.get(item.type === "audio" ? "audio" : "Image/Video").append(clip); }); track.append(trackInner); timeline.append(track);
    // Unified prompt-builder form replacing legacy per-item/global prompts
    const promptPanel = document.createElement("div"); promptPanel.className = "ds-h3-prompt-panel";
    if (promptStyle() === "simple") {
      buildSimpleForm(promptPanel);
    } else if (mode() === "REF2VA") {
      buildRefForm(promptPanel);
    } else {
      buildBaseForm(promptPanel);
    }
    if (hasExternalPrompt()) {
      promptPanel.classList.add("disabled");
      promptPanel.querySelectorAll("textarea, input, button").forEach(el => { el.disabled = true; });
      const note = document.createElement("div"); note.className = "ds-h3-small ds-h3-ext-note";
      note.textContent = "External prompt detected — builder fields are disabled.";
      promptPanel.prepend(note);
    }
    timeline.append(promptPanel);
    // Keep the authoring surface in a left column while Preview & Output stays beside it.
    saveNodePanel.remove(); controlRow.style.display = "block";
    const leftColumn = document.createElement("div"); leftColumn.className = "ds-h3-authoring-column"; leftColumn.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:6px"; leftColumn.append(controlRow, infoFeed, track, promptPanel);
    const contentLayout = document.createElement("div"); contentLayout.className = "ds-h3-content-layout"; contentLayout.style.cssText = "width:100%;box-sizing:border-box;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:8px;align-items:stretch"; contentLayout.append(leftColumn, saveNodePanel); timeline.append(contentLayout);
    // Prompt text is edited on the corresponding media row and synchronized to prompt_blocks.
    timeline.append(status); if (domWidget) domWidget.computeSize = () => [Math.max(420, node.size?.[0] || 520), uiHeight()];
  };
  let domWidget;
  const uiHeight = () => Math.max(850, 500 + mediaPromptHeight + globalPromptHeight);
  const minimumNodeSize = [440, 880];
  let syncingNodeBounds = false;
  const syncNodeBounds = () => {
    if (syncingNodeBounds) return;
    const [currentWidth = minimumNodeSize[0], currentHeight = minimumNodeSize[1]] = node.size || [];
    const [contentWidth = minimumNodeSize[0], contentHeight = minimumNodeSize[1]] = node.computeSize?.() || [];
    const width = Math.max(minimumNodeSize[0], contentWidth, currentWidth);
    const height = Math.max(minimumNodeSize[1], uiHeight(), contentHeight, currentHeight);
    if (width !== currentWidth || height !== currentHeight) {
      syncingNodeBounds = true;
      node.setSize?.([width, height]);
      syncingNodeBounds = false;
    }
    timeline.style.width = `${Math.max(1, (node.size?.[0] || width) - 20)}px`;
    timeline.style.height = `${uiHeight()}px`;
  };
  if (node.addDOMWidget) {
    domWidget = node.addDOMWidget("minimax_h3_director_ui", "custom", timeline, { serialize: false, hideOnZoom: false, getHeight: () => uiHeight() });
    domWidget.computeSize = () => [Math.max(420, node.size?.[0] || 520), uiHeight()];
  }
  const oldResize = node.onResize;
  node.onResize = function (...args) { oldResize?.apply(this, args); syncNodeBounds(); };
  const restorePersistedState = () => {
    state = parseState(dataWidget.value);
    for (const key of Object.keys(fieldHeights)) delete fieldHeights[key];
    for (const [key, value] of Object.entries(state.field_heights || {})) {
      const height = Number(value);
      if (Number.isFinite(height) && height >= 60) fieldHeights[key] = height;
    }
    state.postprocess_recipe = Array.isArray(state.postprocess_recipe) ? state.postprocess_recipe : DEFAULT_POSTPROCESS_RECIPE;
    state.internal_execution = { sampler: "res_multistep", scheduler: "simple", steps: 25, shift_video: 11, shift_audio: 4, comfy_kitchen_attention: false, cache: { enabled: false, reuse_threshold: 0.05, start_percent: 0.15, end_percent: 0.90, max_steps: 2, device: "auto", verbose: false }, ...(state.internal_execution && typeof state.internal_execution === "object" ? state.internal_execution : {}) };
    state.internal_execution.cache = { enabled: false, reuse_threshold: 0.05, start_percent: 0.15, end_percent: 0.90, max_steps: 2, device: "auto", verbose: false, ...(state.internal_execution.cache && typeof state.internal_execution.cache === "object" ? state.internal_execution.cache : {}) };
    state.internal_execution.save = { output_kind: "image", filename_prefix: "DaSiWa_MiniMaxH3_%date:yyyyMMdd%_%seed%", file_format: "webp", compression: 15, save_output: true, save_workflow: true, codec: "Auto", container: "Auto", bit_depth: "Auto", quality: 20, pingpong: false, crop_to_audio: false, audio_codec: "Auto", audio_bitrate: "192k", save_first_frame: false, save_last_frame: false, model_hash: "", text_positive: "", text_negative: "", text_steps: 0, text_cfg: 0, text_sampler: "", text_scheduler: "", text_seed: 0, text_model: "", ...(state.internal_execution.save || {}) };
    selectedId = null;
    const m = mode();
    const baseDefaults = DEFAULT_BUILDER_STATE(m);
    const rawBuilder = builderWidget?.value || state.builder_state;
    if (rawBuilder) {
      try {
        const parsed = typeof rawBuilder === "string" ? JSON.parse(rawBuilder) : rawBuilder;
        if (parsed && typeof parsed === "object") {
          builderState = { ...baseDefaults, ...parsed, ref: { ...baseDefaults.ref, ...(parsed.ref || {}) } };
          builderState.mode = m;
        }
      } catch { /* fall back to defaults */ }
    } else {
      builderState = baseDefaults;
      builderState.mode = m;
    }
    if (!builderHasContent(builderState)) {
      const preserved = legacyPrompt(state, promptWidget?.value);
      if (preserved) {
        builderState.prompt_mode = "simple";
        builderState.simple_prompt = preserved;
      }
    }
    emit();
    syncNodeBounds();
    render();
    void Promise.all(state.items.filter(item => item.type === "video" && !item.thumbnail).map(async item => {
      const thumb = await captureFirstFrame(viewUrl(item.value));
      if (thumb) mutate(s => { const x = s.items.find(i => i.id === item.id); if (x) x.thumbnail = thumb; });
    })).then(render);
  };
  const oldConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    oldConfigure?.apply(this, args);
    requestAnimationFrame(restorePersistedState);
  };
  const oldConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    oldConnectionsChange?.apply(this, args);
    const linked = hasExternalPrompt();
    if (linked !== lastExternalPromptLinked) {
      lastExternalPromptLinked = linked;
      requestAnimationFrame(render);
    }
    const externalCanvasLinked = hasExternalCanvas();
    if (externalCanvasLinked !== lastExternalCanvasLinked) {
      lastExternalCanvasLinked = externalCanvasLinked;
      requestAnimationFrame(render);
    }
    const externalSeedLinked = hasExternalSeed();
    if (externalSeedLinked !== lastExternalSeedLinked) {
      lastExternalSeedLinked = externalSeedLinked;
      requestAnimationFrame(render);
    }
    const externalSamplingLinked = hasExternalSampling();
    if (externalSamplingLinked !== lastExternalSamplingLinked) {
      lastExternalSamplingLinked = externalSamplingLinked;
      requestAnimationFrame(render);
    }
  };
  const oldOnExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    const result = oldOnExecuted?.apply(this, arguments);
    const asset = message?.gifs?.[0] ?? message?.videos?.[0] ?? message?.images?.[0];
    if (asset?.filename) {
      this.__dasiwaH3SavedPreview = asset;
      render();
    }
    return result;
  };
  node.__dasiwaH3RestorePersistedState = () => requestAnimationFrame(restorePersistedState);
  node.__dasiwaH3State = () => state; node.__dasiwaH3Render = render;
  node.__dasiwaH3PrepareSeed = () => { if (hasExternalSeed() || state.seed_control?.mode !== "random") return; const words = crypto.getRandomValues(new Uint32Array(2)); const value = ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString(); const widget = seedWidget(); if (widget) { widget.value = value; widget.callback?.(value); } state.seed_control.last_seed = value; state.seed_control.recent = [value, ...(state.seed_control.recent || []).filter(entry => entry !== value)].slice(0, 10); emit(); };
  if (modeWidget) { const old = modeWidget.callback; modeWidget.callback = value => { selectedModeOverride = value; old?.(value); render(); }; }
  const extWidget = externalPromptWidget();
  if (extWidget) { const old = extWidget.callback; extWidget.callback = value => { old?.(value); render(); }; }
  const lengthWidget = node.widgets?.find(w => w.name === "duration");
  if (lengthWidget) { const old = lengthWidget.callback; lengthWidget.callback = value => { old?.call(lengthWidget, value); render(); }; }
  const oldDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function (...args) { oldDrawForeground?.apply(this, args); const seconds = Math.max(1, Number(lengthWidget?.value) || 5); if (seconds !== lastTimelineLength) render(); };
  node.__dasiwaH3LengthPoll = window.setInterval(() => { const seconds = Math.max(1, Number(lengthWidget?.value) || 5); if (seconds !== lastTimelineLength) render(); }, 200);
  node.__dasiwaH3ExtPoll = window.setInterval(() => {
    const linked = hasExternalPrompt();
    if (linked !== lastExternalPromptLinked) {
      lastExternalPromptLinked = linked;
      render();
    }
    const externalCanvasLinked = hasExternalCanvas();
    if (externalCanvasLinked !== lastExternalCanvasLinked) {
      lastExternalCanvasLinked = externalCanvasLinked;
      render();
    }
    const externalSamplingLinked = hasExternalSampling();
    if (externalSamplingLinked !== lastExternalSamplingLinked) {
      lastExternalSamplingLinked = externalSamplingLinked;
      render();
    }
  }, 300);
  syncNodeBounds();
  requestAnimationFrame(syncNodeBounds);
  render();

}

const DIRECTOR_NODE_TYPES = new Set(["MiniMaxH3DirectorV2"]);
app.registerExtension({ name: "DaSiWa.MiniMaxH3DirectorV2", nodeCreated(node) { if (DIRECTOR_NODE_TYPES.has(node.comfyClass)) install(node); }, loadedGraphNode(node) { if (DIRECTOR_NODE_TYPES.has(node.comfyClass)) { install(node); node.__dasiwaH3RestorePersistedState?.(); } }, beforeQueued() { for (const node of app.graph?._nodes || []) if (DIRECTOR_NODE_TYPES.has(node.comfyClass)) node.__dasiwaH3PrepareSeed?.(); } });
