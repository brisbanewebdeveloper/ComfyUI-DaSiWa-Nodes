import assert from "node:assert/strict";
import fs from "node:fs";

const sourceUrl = new URL("../js/minimax_h3_director.js", import.meta.url);
let source = fs.readFileSync(sourceUrl, "utf8");
source = source
  .replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
  .replace('import { api } from "../../scripts/api.js";', "const api = {};")
  .concat("\nexport { DIRECTOR_NODE_ID, isLegacyDaSiWaDirector, migrateLegacyDaSiWaDirectors };\n");
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const {
  DIRECTOR_NODE_ID,
  isLegacyDaSiWaDirector,
  migrateLegacyDaSiWaDirectors,
} = await import(moduleUrl);

function legacyNode(id = 1) {
  return {
    id,
    type: "MiniMaxH3Director",
    inputs: [
      { name: "fl2va_model" },
      { name: "ref2va_model" },
      { name: "mode" },
      { name: "timeline_data" },
    ],
    properties: { "Node name for S&R": "MiniMaxH3Director" },
    widgets_values: [
      "FL2VA",
      "",
      1344,
      768,
      5,
      "match",
      '{"version":1,"items":[],"prompt_blocks":[]}',
    ],
  };
}

{
  const graph = { nodes: [legacyNode()] };
  assert.equal(migrateLegacyDaSiWaDirectors(graph), 1);
  assert.equal(graph.nodes[0].type, DIRECTOR_NODE_ID);
  assert.equal(graph.nodes[0].properties["Node name for S&R"], DIRECTOR_NODE_ID);
}

{
  const nested = legacyNode(2);
  const graph = { nodes: [], extra: { subgraphs: [{ nodes: [nested] }] } };
  assert.equal(migrateLegacyDaSiWaDirectors(graph), 1);
  assert.equal(nested.type, DIRECTOR_NODE_ID);
}

{
  const integratedDirector = {
    type: "MiniMaxH3Director",
    inputs: [
      { name: "task_type" },
      { name: "global_prompt" },
      { name: "timeline_data" },
    ],
    widgets_values: [
      "t2v — Text to Video",
      "",
      '{"global":{"prompt":""},"segments":[]}',
    ],
  };
  assert.equal(isLegacyDaSiWaDirector(integratedDirector), false);
  assert.equal(migrateLegacyDaSiWaDirectors({ nodes: [integratedDirector] }), 0);
  assert.equal(integratedDirector.type, "MiniMaxH3Director");
}

{
  const conditioningDirector = {
    type: "MiniMaxH3DirectorCS",
    widgets_values: ["FL2VA", '{"version":1,"items":[]}'],
  };
  const currentDirector = {
    type: DIRECTOR_NODE_ID,
    widgets_values: ["FL2VA", '{"version":1,"items":[]}'],
  };
  assert.equal(migrateLegacyDaSiWaDirectors({
    nodes: [conditioningDirector, currentDirector],
  }), 0);
  assert.equal(conditioningDirector.type, "MiniMaxH3DirectorCS");
  assert.equal(currentDirector.type, DIRECTOR_NODE_ID);
}

console.log("MiniMax H3 Director workflow migration tests passed.");
