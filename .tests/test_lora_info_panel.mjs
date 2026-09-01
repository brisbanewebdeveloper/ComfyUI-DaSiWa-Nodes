import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

// Source-load the real UI module with the ComfyUI import stubbed, same
// pattern as test_enhanced_video_combine_widget_drift.mjs.
const sourcePath = new URL("../js/advanced_lora_loader_ui.js", import.meta.url);
let source = await readFile(sourcePath, "utf8");
source = source.replace(
    'import { app } from "../../scripts/app.js";',
    "const app = { graph: { setDirtyCanvas() {} }, canvas: undefined, registerExtension() {} };"
);
source += "\nexport { buildLoraInfoPanelHtml };\n";

const moduleUrl = new URL("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
const { buildLoraInfoPanelHtml } = await import(moduleUrl.href);
assert.equal(typeof buildLoraInfoPanelHtml, "function", "helper must be exported");

const theme = {
    btnBg: "#00FFCC18", btnBorder: "#00FFCC55", btnText: "#00FFCC",
    nameEmpty: "#444444", divider: "#2a2a2a", arrowColor: "#555555", onColor: "#4CAF50",
};

const info = {
    file: "sub/cool_lora.safetensors",
    name: "Cool LoRA - V2",
    sha256: "ab".repeat(32),
    type: "LoRA",
    baseModel: "SDXL",
    imageLocal: "/dasiwa/ltx2/loraimg?lora=sub/cool_lora.safetensors",
    civitaiFound: true,
    links: ["https://civitai.com/models/123?modelVersionId=456"],
    trainedWords: [
        { word: "sks", count: 12, civitai: true },
        { word: "myword", count: 3 },
    ],
    images: [
        { url: "https://civitai.com/image/1.png", type: "image", seed: 7, model: "SDXL" },
        { url: "https://civitai.com/image/2.mp4", type: "video" },
    ],
};

let html = buildLoraInfoPanelHtml(info, theme);
// civitai link, words, images, buttons
assert.match(html, /civitai\.com\/models\/123\?modelVersionId=456/);
assert.match(html, /Cool LoRA - V2/);
assert.match(html, /data-word="sks"/);
assert.match(html, /data-word="myword"/);
assert.match(html, /civitai\.com\/image\/1\.png/);
assert.match(html, /<video[^>]*civitai\.com\/image\/2\.mp4/);
assert.match(html, /loraimg\?lora=sub\/cool_lora\.safetensors/);
assert.match(html, /data-action="refresh"/);
assert.match(html, /data-action="close"/);
assert.match(html, /data-action="copy-words"/);
assert.match(html, /data-action="copy-selected"/);
assert.match(html, /seed 7/);
assert.match(html, /abababab/);

// civitai-missing state: error text, no model link, no word buttons
const missing = buildLoraInfoPanelHtml(
    { ...info, civitaiFound: false, civitaiError: "model not found on civitai", links: [], images: [], trainedWords: [] },
    theme,
);
assert.match(missing, /model not found on civitai/i);
assert.doesNotMatch(missing, /civitai\.com\/models/);
assert.doesNotMatch(missing, /data-action="copy-words"/);
assert.doesNotMatch(missing, /data-word=/);

// local sidecar image shows even without civitai
const localOnly = buildLoraInfoPanelHtml(
    { ...info, civitaiFound: false, links: [], trainedWords: [], images: [] },
    theme,
);
assert.match(localOnly, /loraimg\?lora=/);
assert.match(localOnly, /local/);

// XSS safety: file names with markup must be escaped
const xss = buildLoraInfoPanelHtml(
    { file: '<img src=x onerror=alert(1)>.safetensors', civitaiFound: false, links: [], trainedWords: [], images: [] },
    theme,
);
assert.doesNotMatch(xss, /<img src=x/);
assert.match(xss, /&lt;img src=x/);

// ── Trash button: geometry + behavior (ASCII-drawn, no emoji) ──────────────
const rawSource = await readFile(new URL("../js/advanced_lora_loader_ui.js", import.meta.url), "utf8");

// The trash button cell lives right after the (shifted-left) info cell.
assert.match(rawSource, /iX: 962 \* s, iW: 14 \* s/, "info button must shift left to x=962");
assert.match(rawSource, /tX: 976 \* s, tW: 14 \* s/, "trash button must sit at the old info x=976");
assert.doesNotMatch(rawSource, /iX: 976/, "the info cell must no longer occupy x=976");

// Behavior: a trash click resets the slot back to "None" (like selecting None).
assert.match(rawSource, /if \(x > C\.tX && x < C\.tX \+ C\.tW && data\[i\]\.lora !== "None"\)/, "trash hit-test must guard against empty slots");
assert.match(rawSource, /data\[i\]\.lora = "None"/, "trash must set the slot's LoRA back to None");

// Both buttons are drawn with plain canvas paths, not emoji glyphs.
assert.doesNotMatch(rawSource, /🗑/, "trash button must be ASCII-drawn, not an emoji");
assert.doesNotMatch(rawSource, /ⓘ/, "info button must be ASCII-drawn, not an emoji");

console.log("ok — test_lora_info_panel");
