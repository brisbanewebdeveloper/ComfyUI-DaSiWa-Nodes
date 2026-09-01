import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const sourcePath = new URL("../js/node_status_switch.js", import.meta.url);
let source = await readFile(sourcePath, "utf8");
source = source.replace(
    'import { app } from "../../scripts/app.js";\nimport { api } from "../../scripts/api.js";',
    "const app = globalThis.__testApp; const api = globalThis.__testApi;"
);
source += "\nexport { applyAllSwitches, readBoolFromNode, readEnabled, syncExternalToLocal };";

const rootGraph = {
    _nodes: [],
    _links: new Map(),
    getLink(linkId) {
        return this._links.get(linkId);
    },
};
const innerGraph = {
    rootGraph,
    _nodes: [],
    getLink(linkId) {
        return this._links.get(linkId);
    },
    _links: new Map(),
};
const switchNode = {
    id: 42,
    type: "DaSiWa_NodeStatusSwitch",
    graph: innerGraph,
    widgets: [
        { name: "enabled", type: "toggle", value: true },
        { name: "trigger_on", value: "true → active" },
        { name: "action", value: "bypass" },
    ],
    inputs: [
        { name: "enabled", widget: { name: "enabled" } },
        { name: "target_01", link: 2 },
    ],
};
const targetNode = { id: 99, mode: 4 };
const hostInput = {
    name: "enabled",
    widgetId: "root:7:enabled",
    _subgraphSlot: { linkIds: [1] },
};
const hostNode = {
    id: 7,
    graph: rootGraph,
    subgraph: innerGraph,
    inputs: [hostInput],
    getWidgetFromSlot(input) {
        return input === hostInput ? { value: true } : undefined;
    },
};
const outerBoolean = {
    id: 8,
    type: "PrimitiveBoolean",
    widgets: [{ name: "value", type: "toggle", value: false }],
};
innerGraph._nodes.push(switchNode, targetNode);
innerGraph._links.set(1, {
    resolve() {
        return { inputNode: switchNode, input: switchNode.inputs[0] };
    },
});
innerGraph._links.set(2, { origin_id: targetNode.id });
rootGraph._nodes.push(hostNode, outerBoolean);

globalThis.__testApp = {
    graph: rootGraph,
    registerExtension() {},
    graphToPrompt: async () => ({}),
};
globalThis.__testApi = { queuePrompt: async () => ({}) };
globalThis.requestAnimationFrame = () => 0;

const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { applyAllSwitches, readBoolFromNode, readEnabled, syncExternalToLocal } = await import(moduleUrl);

assert.equal(
    readEnabled(switchNode),
    true,
    "a promoted enabled widget must use its host's current value"
);

hostInput.link = 3;
rootGraph._links.set(3, { origin_id: outerBoolean.id });
assert.equal(
    readEnabled(switchNode),
    false,
    "an outer Boolean Primitive linked to a promoted input must override the host widget"
);

syncExternalToLocal(switchNode);
assert.equal(
    switchNode.widgets[0].value,
    false,
    "the live mirror must update a switch controlled through a promoted input"
);

applyAllSwitches();
assert.equal(
    targetNode.mode,
    4,
    "queue-time application must include status switches inside subgraphs"
);

assert.equal(
    readBoolFromNode({ widgets: [{ name: "value", type: "toggle", value: "true" }] }),
    true,
    "a Boolean Primitive string value must be read as true"
);

// ── Director request outputs ─────────────────────────────────────────────
// A switch driven by the MiniMax H3 Director's computed request outputs
// must follow the Director's mode widget, live and at queue time.
const director = {
    id: 55,
    type: "MiniMaxH3Director",
    widgets: [{ name: "mode", value: "FL2VA" }],
    outputs: [
        { name: "guide" }, { name: "duration" }, { name: "positive_prompt" },
        { name: "width" }, { name: "height" }, { name: "model" },
        { name: "fl2va_requested" }, { name: "inpaint_requested" }, { name: "frame_rate" },
    ],
};
const directorSwitch = {
    id: 56,
    type: "DaSiWa_NodeStatusSwitch",
    graph: rootGraph,
    widgets: [
        { name: "enabled", type: "toggle", value: true },
        { name: "trigger_on", value: "true → active" },
        { name: "action", value: "mute" },
    ],
    inputs: [
        { name: "enabled", link: 100 },
        { name: "target_01", link: 101 },
    ],
};
const directorTarget = { id: 57, mode: 0 };
rootGraph._nodes.push(director, directorSwitch, directorTarget);
rootGraph._links.set(100, { origin_id: director.id, origin_slot: 7 }); // inpaint_requested
rootGraph._links.set(101, { origin_id: directorTarget.id });

assert.equal(
    readEnabled(directorSwitch),
    false,
    "inpaint_requested must be false while the Director mode is FL2VA"
);

director.widgets[0].value = "Image Inpaint";
assert.equal(
    readEnabled(directorSwitch),
    true,
    "inpaint_requested must be true when the Director mode is Image Inpaint"
);

syncExternalToLocal(directorSwitch);
assert.equal(
    directorSwitch.widgets[0].value,
    true,
    "the live mirror must flip the switch when the mode pill switches to Image Inpaint"
);

applyAllSwitches();
assert.equal(
    directorTarget.mode,
    0,
    "queue-time application must keep the target active while inpaint is requested"
);

director.widgets[0].value = "FL2VA";
applyAllSwitches();
assert.equal(
    directorTarget.mode,
    2,
    "the target must be muted again when the mode leaves Image Inpaint"
);

// fl2va_requested: true for every image-to-video mode, false for REF2VA.
rootGraph._links.set(100, { origin_id: director.id, origin_slot: 6 }); // fl2va_requested
director.widgets[0].value = "L2VA";
assert.equal(readEnabled(directorSwitch), true, "fl2va_requested must be true for L2VA");
director.widgets[0].value = "Image Inpaint";
assert.equal(readEnabled(directorSwitch), true, "fl2va_requested must be true for Image Inpaint");
director.widgets[0].value = "REF2VA";
assert.equal(readEnabled(directorSwitch), false, "fl2va_requested must be false for REF2VA");
