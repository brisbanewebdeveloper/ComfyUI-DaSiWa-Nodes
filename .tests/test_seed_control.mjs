import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../js/dasiwa_seed_control.js", import.meta.url), "utf8");

// ---- Node wiring: standalone node type + persisted state restore + auto-roll ----
assert.match(source, /DaSiWa_SeedControl/, "seed control frontend must install for the standalone node class");
assert.match(source, /__dasiwaSeedRestorePersistedState/, "loaded graphs must re-read the persisted seed-control state");
assert.match(source, /__dasiwaSeedPrepareSeed/, "queuing must roll a seed in Random mode, like the Director");
assert.match(source, /beforeQueued/, "seed preparation must run before the graph is queued");
assert.match(source, /addDOMWidget\("dasiwa_seed_control_ui"/, "the panel must attach as the node's DOM widget");

// ---- External seed socket semantics (Director behaviour) ----
assert.match(source, /i\.name === "seed"/, "external seed detection must target the seed input socket");
assert.match(source, /External seed connected/, "a linked seed socket must replace the local controls with a note");
assert.match(source, /__dasiwaSeedExtPoll/, "connect/unconnect changes must re-evaluate external seeding");

// ---- Panel controls: same feature set as the Director seed panel ----
assert.match(source, /ds-seed-num/, "the seed input must reuse the Director's res-num visual family");
assert.match(source, /ds-seed-spin/, "the seed field must carry an attached up/down spinner on the input's right edge");
assert.match(source, /ds-seed-spinbtn/, "the spinner must expose stacked step buttons");
assert.match(source, /Seed \+1 \(wraps at 2\^64-1, locks Fixed\)\. Hold to repeat\./, "step-up must wrap at the 64-bit maximum, lock Fixed, and hold-repeat");
assert.match(source, /Seed -1 \(wraps at 0, locks Fixed\)\. Hold to repeat\./, "step-down must wrap at zero, lock Fixed, and hold-repeat");
assert.match(source, /ds-seed-switch/, "Random/Fixed must be a single switch control, not two separate buttons");
assert.match(source, /ds-seed-switch-seg/, "the switch must expose two flat segments");
assert.match(source, /\[\["random", "Random"\], \["fixed", "Fixed"\]\]/, "the switch segments must be Random and Fixed");
assert.match(source, /retain the selected mode/, "New must roll without changing the selected mode");
assert.match(source, /textContent = "Use Last"/, "the reuse action must stay compact");
assert.match(source, /Last 10 seeds/, "seed control must expose the last ten seeds");
assert.match(source, /Copy seed/, "each recent seed must expose a copy action");
assert.match(source, /No previous seeds/, "an empty history must show a placeholder");
assert.match(source, /BigInt/, "seed values must retain the full unsigned 64-bit range");
assert.match(source, /crypto\.getRandomValues\(new Uint32Array\(2\)\)/, "rolls must combine two Uint32s into a 64-bit seed");
assert.match(source, /ds-seed-control/, "the seed panel must render its own control container");
assert.match(source, /ds-seed-row/, "the panel must stack its controls in horizontal rows");
assert.match(source, /ds-seed-btn/, "the panel must reuse the Director's res-btn visual family");
assert.match(source, /seed_value/, "the panel must edit the hidden seed_value widget");
assert.match(source, /seed_control_state/, "the panel must persist state through the hidden seed_control_state widget");

// ---- Three-row layout: seed+spinner / switch+New / Use Last+Last 10 ----
assert.match(source, /numwrap\.append\(input, spin\)/, "row 1 must wrap the seed field with the attached spinner");
assert.match(source, /fieldRow\.append\(numwrap\)/, "row 1 must hold the numwrap only, no heading label");
assert.match(source, /modeRow\.append\(switchEl\)/, "row 2 must start with the Random|Fixed switch");
assert.match(source, /modeRow\.append\(roll\)/, "row 2 must end with the New roll button");
assert.match(source, /historyRow\.append\(last, history\)/, "row 3 must hold Use Last and the last-10 dropdown");
assert.match(source, /seedControl\.append\(fieldRow, modeRow, historyRow\)/, "the panel must stack the three rows in order");

// ---- Lossless seed text: display/step never read the native INT widget back ----
// The native INT widget coerces a 16-digit seed to a lossy JS number, so the
// panel keeps its own decimal string as the source of truth and only writes
// the widget as a mirror (same architecture as Pixaroma's properties-based seed).
assert.match(source, /let lastSeedText = controlState\.last_seed \|\| String\(dasiwaSeedControlSeedWidget\(\)\.value \?\? 0\)/, "the display seed must start from persisted state, with the widget only as an initial mirror");
assert.match(source, /const dasiwaSeedControlWriteSeedMirror = text => \{/, "the seed widget must be written through a dedicated mirror helper");
assert.match(source, /lastSeedText = value\.toString\(\); input\.value = lastSeedText/, "stepping must commit the lossless text into the field, not read the widget back");
assert.doesNotMatch(source, /dasiwaSeedControlCurrentSeed\(\) \?\? 0\)\) \+ delta/, "stepping must not base itself on the native widget value");

// ---- In-place updates: stepping/switching never rebuild the panel mid-hold ----
assert.match(source, /function.*SyncSwitch|dasiwaSeedControlSyncSwitch = \(\)/, "mode flips must sync the switch in place");
assert.match(source, /classList\.toggle\("active", button\.dataset\.mode === controlState\.mode\)/, "the active segment must be toggled in place, not re-rendered");
assert.match(source, /if \(controlState\.mode === modeName\) return;/, "clicking the active segment must be a no-op");

// ---- Unique internal function names (pack-wide uniqueness) ----
assert.match(source, /function dasiwaSeedControlInstall\(/, "install must carry the dasiwaSeedControl prefix");
assert.match(source, /function dasiwaSeedControlParseState\(/, "state parsing must carry the dasiwaSeedControl prefix");
assert.match(source, /function dasiwaSeedControlInstallStyles\(/, "style injection must carry the dasiwaSeedControl prefix");
assert.match(source, /DASIWASEED_MAX_SEED/, "constants must carry the DASIWASEED prefix");
assert.match(source, /DASIWASEED_NODE_TYPES/, "constants must carry the DASIWASEED prefix");
const topLevelFunctionNames = [...source.matchAll(/^(?:function |const )[A-Za-z0-9_]+/gm)]
  .map(entry => entry[0].replace(/^function |^const /, ""));
const nonUnique = topLevelFunctionNames.filter(name => !name.startsWith("dasiwaSeedControl") && !name.startsWith("DASIWASEED"));
assert.equal(nonUnique.length, 0, `top-level declarations must be uniquely prefixed, found: ${nonUnique.join(", ")}`);

// ---- Fixed panel column: fields keep the same width when the node resizes ----
assert.match(source, /DASIWASEED_PANEL_WIDTH/, "the panel must use a fixed column width constant");
assert.match(source, /width:\$\{DASIWASEED_PANEL_WIDTH\}px/, "the root panel must render at the fixed column width, not 100%");
assert.match(source, /domWidget\.computeSize = \(\) => \[DASIWASEED_COMPUTE_SIZE_WIDTH/, "the widget size must be fixed, not follow the node width");
assert.match(source, /flex:1;min-width:0/, "the last-10 history must flex to fill the fixed row width");
assert.match(source, /historyRow\.append\(last, history\)/, "row 3 must hold Use Last and the last-10 dropdown");
// Seed number must always display fully: the column is wide enough for a
// 16-digit monospace seed and the font auto-shrinks if it still overflows.
assert.match(source, /dasiwaSeedControlFitSeedFont/, "the seed font must auto-fit the field so a full seed never clips");
assert.match(source, /input\.scrollWidth > input\.clientWidth/, "the auto-fit must shrink the font while the digits overflow");
assert.match(source, /dasiwaSeedControlRefitFont/, "value changes must re-run the font auto-fit");

// ---- State persistence: same shape as the Director's seed_control block ----
assert.match(source, /mode: "random", last_seed: null, recent: \[\]/, "state defaults must mirror the Director's seed_control defaults");
assert.match(source, /controlState\.recent\.filter\(entry => entry !== value\)\]\.slice\(0, 10\)/, "the recent list must stay capped at ten entries");
