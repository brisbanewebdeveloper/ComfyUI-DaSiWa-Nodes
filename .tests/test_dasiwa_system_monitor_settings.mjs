import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../js/dasiwa_system_monitor.js", import.meta.url), "utf8");

assert.match(source, /id: MONITOR_ENABLED_SETTING/);
assert.match(source, /category: \["DaSiWa", "System Monitor", "Enable System Monitor"\]/);
assert.match(source, /type: "boolean"/);
assert.match(source, /defaultValue: true/);
assert.match(source, /onChange: \(enabled\) => \{ void setMonitorEnabled\(enabled !== false\); \}/);
assert.match(source, /api\.removeEventListener\(EVENT_NAME, monitorEventListener\)/);
assert.match(source, /monitorRoot\?\.remove\(\)/);
assert.match(source, /MONITOR_ENABLED_ENDPOINT/);
assert.match(source, /await syncMonitorEnabled\(enabled\)/);
assert.match(source, /if \(enabled\) await mountMonitor\(\)/);
assert.doesNotMatch(source, /Show system monitor/);
assert.doesNotMatch(source, /settings\.enabled/);

console.log("ok — test_dasiwa_system_monitor_settings");
