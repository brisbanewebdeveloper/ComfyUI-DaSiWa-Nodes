import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

// Auto aspect must follow timeline slot order, not insertion order.
const slotSort = /sort\(\(a, b\) => \(Number\(a\.slot\) \|\| 0\) - \(Number\(b\.slot\) \|\| 0\) \|\| a\.order - b\.order\)\[0\]/;
for (const file of ["../js/minimax_h3_director.js"]) {
  const source = await readFile(new URL(file, import.meta.url), "utf8");
  assert.match(source, slotSort, `${file}: Auto aspect must resolve the first visual reference by timeline slot, falling back to insertion order`);
  assert.doesNotMatch(source, /sort\(\(a, b\) => a\.order - b\.order\)\[0\]/, `${file}: insertion-order-only Auto aspect resolution must be gone`);
}
