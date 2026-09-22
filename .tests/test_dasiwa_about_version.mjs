import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const pyproject = await readFile(new URL("../pyproject.toml", import.meta.url), "utf8");
const version = pyproject.match(/^version = "([^"]+)"$/m)?.[1];
assert.ok(version, "pyproject.toml must declare a project version");

const sourcePath = new URL("../js/dasiwa_about_version.js", import.meta.url);
const source = await readFile(sourcePath, "utf8");
const moduleSource = source.replace(
    'import { app } from "../../scripts/app.js";',
    "const app = { registerExtension(value) { globalThis.__dasiwaAboutExtension = value; } };",
);

await import(new URL("data:text/javascript;base64," + Buffer.from(moduleSource).toString("base64")).href);
const extension = globalThis.__dasiwaAboutExtension;
delete globalThis.__dasiwaAboutExtension;

assert.equal(extension.name, "DaSiWa.AboutVersion");
assert.deepEqual(extension.aboutPageBadges, [{
    label: `DaSiWa Custom Nodes v${version}`,
    url: "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes",
    icon: "pi pi-github",
}]);

console.log("ok — test_dasiwa_about_version");
