import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const sourcePath = new URL("../js/enhanced_video_combine_preview.js", import.meta.url);
let source = await readFile(sourcePath, "utf8");
source = source.replace(
    'import { app } from "../../scripts/app.js";\nimport { api } from "../../scripts/api.js";',
    "const app = { registerExtension() {} }; const api = {};"
);
// Expose the pure helper plus a test-only setter for the module-local schema,
// so the test can drive the real implementation without a live graph.
source += "\nexport { sanitizeWidgetValues };\nexport function __setVideoCombineSchemaForTest(s) { videoCombineWidgetSchema = s; }";

const moduleUrl = new URL("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
const { sanitizeWidgetValues, __setVideoCombineSchemaForTest } = await import(moduleUrl.href);

// Frontend nodeData.input.required uses Python-style 2-tuple schemas.
__setVideoCombineSchemaForTest({
    audio_codec: [["Auto", "AAC", "Opus", "MP3"], { default: "Auto" }],
    codec: [["Auto", "AV1", "VP9", "H.264", "H.265 (HEVC)"], { default: "Auto" }],
    save_first_frame: ["BOOLEAN", { default: false }],
});

const calls = [];
function makeWidget(name, type, value, options) {
    return {
        name,
        type,
        value,
        options,
        callback: (v) => calls.push([name, v]),
    };
}

// Simulate a drifted workflow load: a boolean `true` landed on the audio_codec
// COMBO and an empty string landed on a BOOLEAN widget (old 17-slot layout).
const node = {
    widgets: [
        makeWidget("audio_codec", "COMBO", true, { values: ["Auto", "AAC", "Opus", "MP3"] }),
        makeWidget("codec", "COMBO", "AV1", { values: ["Auto", "AV1", "VP9", "H.264", "H.265 (HEVC)"] }),
        makeWidget("save_first_frame", "BOOLEAN", "", undefined),
    ],
};

sanitizeWidgetValues(node);

assert.equal(node.widgets[0].value, "Auto", "drifted combo value resets to schema default");
assert.equal(node.widgets[1].value, "AV1", "valid combo value is untouched");
assert.equal(node.widgets[2].value, false, "non-boolean on boolean widget resets to default");
assert.deepEqual(
    calls,
    [
        ["audio_codec", "Auto"],
        ["save_first_frame", false],
    ],
    "callbacks fire only for corrected widgets"
);

console.log("widget-drift self-heal: OK");
