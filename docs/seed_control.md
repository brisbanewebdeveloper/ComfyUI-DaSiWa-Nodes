# Seed Control

**Category:** `DaSiWa`
**Display name:** `Seed Control`
**Class name:** `DaSiWa_SeedControl`
**File:** `nodes/nodes_seed_control.py`
**Frontend:** `js/dasiwa_seed_control.js`

---

## Overview

The Seed Control is the MiniMax H3 Director's seed panel extracted into a
standalone node. It gives every workflow the same seed handling the Director
provides — without requiring a Director node in the graph.

The node owns an unsigned 64-bit seed (the full H3 seed space,
`0..0xFFFFFFFFFFFFFFFF`), a Random/Fixed mode switch, a "New" roll that keeps
the selected mode, a "Use Last" restore, and a "Last 10 seeds" history with copy
actions. Linking the external `seed` socket disables the local controls and
passes the connected value through, exactly like the Director's external seed
input.

In Random mode with no local seed value set, the node rolls a fresh seed on
every queue, so it behaves identically through the panel and through the
headless API.

---

## Inputs

| Input | Type | Description |
|---|---|---|
| `seed_value` | INT (hidden) | Local seed value, `0..0xFFFFFFFFFFFFFFFF`. Edited by the panel; the native widget stays hidden. |
| `seed_control_state` | STRING (hidden) | Persisted JSON state `{mode, last_seed, recent[]}` — the same shape as the Director's `seed_control` block. |
| `seed` | INT (optional, force input) | External seed override. When linked, local controls are replaced by an "External seed connected" note and this value passes through. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `seed` | INT | The effective seed: the external value when linked, otherwise the local value (rolled in Random mode when it was 0). |
| `noise` | NOISE | A `NOISE`-compatible object wrapping the same seed, ready to feed any node that accepts a `NOISE` input (e.g. the LTX sampler's `noise` socket). |

---

## UI

The panel is a DOM widget that reuses the Director's `ds-h3-res-*` visual
family (`ds-seed-btn` / `ds-seed-num` classes) so it reads as the same
control set. It stacks three rows and keeps a fixed panel column
(`DASIWASEED_PANEL_WIDTH`, ~240 px), so resizing the node never stretches
or reflows the fields:

- **Row 1 — Seed field** — centred 64-bit numeric input with an attached ▲/▼ spinner on its right edge (like the Pixaroma seed control); stepping wraps at the 64-bit bounds (▲ up wraps at 2⁶⁴−1, ▼ down wraps at 0), both buttons repeat while held, and each step locks the seed as Fixed; the number auto-fits its font so a full 16-digit seed never clips; no heading, since the node title already says Seed Control.
- **Row 2 — Random|Fixed switch + New** — one segmented switch (a single control with two segments, Pixaroma style) that flips between Random (rolls a fresh seed every queue) and Fixed (keeps the current seed, repeatable), plus a "New" roll that keeps the selected mode. The switch and New flex to fill the column; flipping the switch re-marks the active segment in place, so a click that also commits the number field never destroys the spinner under the pointer.
- **Row 3 — Use Last / Last 10 seeds** — 68 px restore of the previous seed
  (flips to Fixed) and the collapsible history dropdown that flexes to the
  remaining column width, with per-entry copy actions.

Every row shares the same 42 px cell height so the columns align evenly.

All internal functions and constants in `js/dasiwa_seed_control.js` carry a
`dasiwaSeedControl` / `DASIWASEED` prefix so their names stay unique across
the pack's JS files.

When an external seed is linked the local controls are replaced by the
"External seed connected" note, matching the Director.

---

## State persistence

Mode, last seed, and the ten-entry history live in the hidden
`seed_control_state` widget, so they survive saving and reloading the
workflow. The backend normalizes any garbage payload back to safe defaults
(`{mode, last_seed, recent[]}`), so a stale or hand-edited value cannot crash
the node.

---

## Notes

- The panel keeps the seed as a **lossless decimal string** (`lastSeedText`),
  not the native INT widget value. The native INT widget coerces its value to
  a JS `Number`, which is lossy at 16 digits, so reading it back would desync
  the spinner. The panel only *writes* the widget as a mirror and never reads
  it back — the same architecture as Pixaroma's properties-based seed.
- The roll combines two `crypto.getRandomValues` Uint32 values into one
  64-bit seed, matching the Director's JS roll.
- Stepping, typing, and the mode switch all commit **in place** (no panel
  rebuild), so a long spinner hold or a mode click that also commits the
  number field can never destroy the control under the pointer.
- The backend independently rolls in Random mode when the local value is 0,
  so API clients (no DOM) still get a fresh seed every queue.
- Out-of-range seeds (below 0 or above `0xFFFFFFFFFFFFFFFF`) raise a
  `ValueError` in `execute`.
