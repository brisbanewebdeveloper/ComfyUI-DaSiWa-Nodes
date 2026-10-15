const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'js/minimax_h3_director.js'), 'utf8');

test('Forge retains distinct saved member descriptions, including deliberately empty notes', () => {
  const start = source.indexOf('  const forgeRefs = () => {');
  const end = source.indexOf('  node.__dasiwaH3Continuity =', start);
  assert.ok(start >= 0 && end > start);
  const row = { slot: 1, name: 'bundle', description: 'Common Director description' };
  const entry = { name: 'bundle', description: 'Bundle overview', kinds: ['image', 'image', 'audio'],
    members: [{ kind: 'image', description: 'First picture' }, { kind: 'image', description: 'Second picture' }, { kind: 'audio', description: '' }] };
  const context = vm.createContext({ state: { refmods: [row] }, refModLibrary: { entries: [entry] },
    continuityContext: () => null, mode: () => 'REF2VA', node: {}, forgeItems: () => [],
    forgeReferences: () => [{ id: 'current', instructions: 'Current image' }] });
  vm.runInContext(source.slice(start, end) + '\nglobalThis.references = forgeRefs;', context);
  const refs = context.references();
  assert.deepEqual(Array.from(refs, ref => ref.instructions), ['Current image', 'First picture', 'Second picture', '']);
  assert.equal(row.description, 'Common Director description');
  assert.equal(entry.description, 'Bundle overview');
  delete entry.members;
  assert.deepEqual(Array.from(context.references().slice(1), ref => ref.instructions), Array(3).fill('Common Director description'));
  row.description = '';
  assert.deepEqual(Array.from(context.references().slice(1), ref => ref.instructions), Array(3).fill('Bundle overview'));
});

test('Director adds a separate namespaced export action directly after Save', () => {
  assert.match(source, /import \{ openDaSiWaRefModExport \} from "\.\/refmod_export\.js"/);
  const save = source.indexOf('addDropdown("", "save-picker"');
  const action = source.indexOf('exportButton.textContent = "Create RefMod"', save);
  assert.ok(save >= 0 && action > save);
  assert.ok(action < source.indexOf('const disableWhenExternal', save));
  assert.match(source, /activeRefModExportCleanup\?\.\(\)/);
});

test('registration uses only the DaSiWa exporter ID', () => {
  const init = fs.readFileSync(path.join(root, '__init__.py'), 'utf8');
  assert.match(init, /from \.nodes\.refmod_export import DaSiWaH3RefModExport/);
  assert.match(init, /"DaSiWaH3RefModExport": DaSiWaH3RefModExport/);
  assert.match(init, /"DaSiWaH3RefModExport": "MiniMaxH3 Create RefMod"/);
  assert.doesNotMatch(init, /"MiniMaxH3RefModExport":/);
});

test('overlay integration snapshots references, verifies library result and avoids auto-selection', async () => {
  const start = source.indexOf('  function openRefModExport()');
  const end = source.indexOf('  const onRefModReloadKey', start);
  assert.ok(start >= 0 && end > start);
  let options, emitted = 0, closed = 0;
  const messages = [];
  const state = { items: [{ id: 'image', type: 'image', value: 'a.png', enabled: false }], refmods: [] };
  const library = { loaded: false, entries: [] };
  const context = vm.createContext({
    state, node: { id: 7 }, nodeRemoved: false, app: {}, structuredClone,
    emit: () => emitted++, viewUrl: x => x, setStatus: (...args) => messages.push(args),
    render() {}, refModLibrary: library, api: { fetchApi: async () => ({ ok: true, json: async () => [{ name: 'hero' }] }) },
    openDaSiWaRefModExport: value => { options = value; return () => closed++; },
  });
  vm.runInContext('let activeRefModExportCleanup = null;\n' + source.slice(start, end) + '\nglobalThis.openExport = openRefModExport;', context);
  context.openExport();
  assert.equal(emitted, 1);
  assert.equal(JSON.stringify(options.snapshot), JSON.stringify(state));
  assert.notEqual(options.snapshot, state);
  options.snapshot.items[0].description = 'export only';
  assert.equal(state.items[0].description, undefined);
  await options.onSaved({ name: 'hero', message: 'Saved hero', warnings: [] });
  assert.equal(library.loaded, true);
  assert.equal(library.entries[0].name, 'hero');
  assert.equal(state.refmods.length, 0);
  context.openExport();
  assert.equal(closed, 1);
  context.nodeRemoved = true;
  context.openExport();
  assert.equal(emitted, 2);
});
