import { app } from "../../scripts/app.js";

// LLM Prompt Writer: start_with is one small fixed box, the idea box
// is about two and a half times it and takes any extra height the node gets.
const START_WITH_HEIGHT = 56;
// Widget heights include about 20 px of padding, so 56 and 110 draw text
// areas of roughly 36 and 90: the 2.5x asked for.
const IDEA_MIN_HEIGHT = 110;

function installModernTextLayout() {
  if (document.getElementById("dasiwa-prompt-writer-layout")) return;
  const style = document.createElement("style");
  style.id = "dasiwa-prompt-writer-layout";
  // Native Nodes 2.0 textareas forward options as attributes, but do not use
  // the legacy DOM widget's height callbacks. Scope layout to our marked fields.
  style.textContent = `
.lg-node [data-testid="node-widgets"]:has(textarea[data-dasiwa-prompt-field]) {
  display:flex; flex-direction:column; flex:1; min-height:0;
}
.lg-node [data-testid="node-widgets"]:has(textarea[data-dasiwa-prompt-field]) > [data-testid="node-widget"] {
  grid-template-columns:min-content minmax(80px,min-content) minmax(125px,1fr); flex-shrink:0;
}
.lg-node [data-testid="node-widget"]:has(textarea[data-dasiwa-prompt-field="start_with"]) {
  flex:0 0 ${START_WITH_HEIGHT}px; height:${START_WITH_HEIGHT}px;
}
.lg-node textarea[data-dasiwa-prompt-field="start_with"] { min-height:0; }
.lg-node [data-testid="node-widget"]:has(textarea[data-dasiwa-prompt-field="idea"]) {
  flex:1 0 ${IDEA_MIN_HEIGHT}px; min-height:${IDEA_MIN_HEIGHT}px;
}
`;
  document.head.append(style);
}

app.registerExtension({
  name: "DaSiWa.LLMPromptWriter",
  nodeCreated(node) {
    if (node.comfyClass !== "DaSiWa_LLMPromptWriter") return;
    const widget = name => node.widgets?.find(w => w.name === name);
    const startWith = widget("start_with");
    const idea = widget("idea");
    if (!startWith?.options || !idea?.options) return;
    installModernTextLayout();
    startWith.options["data-dasiwa-prompt-field"] = "start_with";
    idea.options["data-dasiwa-prompt-field"] = "idea";
    startWith.options.getMinHeight = () => START_WITH_HEIGHT;
    startWith.options.getMaxHeight = () => START_WITH_HEIGHT;
    idea.options.getMinHeight = () => IDEA_MIN_HEIGHT;
    // A new node opens at the size that fits; a loaded one keeps its own.
    requestAnimationFrame(() => {
      const [width, height] = node.computeSize();
      if (node.size[1] < height) node.setSize([Math.max(node.size[0], width), height]);
      node.setDirtyCanvas?.(true, true);
    });
  },
});
