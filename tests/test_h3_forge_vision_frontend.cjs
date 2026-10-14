const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Minimal DOM host; all dialog behavior runs from the real extension.
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.style = {};
    this.listeners = {}; this.attributes = {}; this.dataset = {}; this.hidden = false;
    this.disabled = false; this.checked = false; this.value = ''; this._text = '';
    this.classList = { toggle() {} };
  }
  set textContent(value) { this._text = value; }
  get textContent() { return this._text + this.children.map(c => typeof c === 'string' ? c : c.textContent).join(''); }
  append(...children) {
    for (const child of children) { this.children.push(child); if (typeof child !== 'string') child.parentElement = this; }
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  insertBefore(child, before) { const i = this.children.indexOf(before); this.children.splice(i, 0, child); child.parentElement = this; }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(c => c !== this); }
  focus() {}
  getBoundingClientRect() { return { x: 0, y: 0, left: 0, top: 0, width: 640, height: 480 }; }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  async dispatch(type) {
    const event = { target: this };
    await this['on' + type]?.(event);
    for (let element = this; element; element = type === 'click' ? element.parentElement : null) {
      for (const callback of element.listeners[type] || []) await callback(event);
    }
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  querySelectorAll(selector) {
    const descendants = this.children.flatMap(c => typeof c === 'string' ? [] : [c, ...c.querySelectorAll('*')]);
    if (selector === '*') return descendants;
    return descendants.filter(c => selector.split(',').some(s => {
      const parts = s.trim().split(' '); const tag = parts.pop();
      if (tag.startsWith('.')) return c.className?.split(' ').includes(tag.slice(1));
      if (c.tagName !== tag.toUpperCase()) return false;
      if (!parts.length) return true;
      let parent = c.parentElement;
      for (const part of parts.reverse()) {
        while (parent && !parent.className?.split(' ').includes(part.slice(1))) parent = parent.parentElement;
        if (!parent) return false;
        parent = parent.parentElement;
      }
      return true;
    }));
  }
  closest(selector) {
    const root = this.parentElement?.parentElement?.parentElement;
    return root?.querySelectorAll(selector).includes(this) ? this : null;
  }
  get options() { return this.querySelectorAll('option'); }
  get selectedOptions() { return this.options.filter(o => o.value === this.value); }
}
const root = path.join(__dirname, '..');
const image = (role = 'character-1') => ({ kind: 'image', path: 'a.png', easy_role: role, role: 'subject', instructions: '', item: { id: 'a' } });
async function dialog({ refs = [image()], mode = 'REF2VA', continuity = null, history = [], prefs = {}, beforeReply = null, castReply = null, peopleReply = null } = {}) {
  const document = { head: new Element('head'), body: new Element('body'), createElement: t => new Element(t), getElementById: () => null, addEventListener() {}, removeEventListener() {} };
  const requests = [], applied = [], timers = new Map(); let timerId = 0;
  const node = { id: 1, properties: { dasiwaH3ForgeHistory: history }, graph: { setDirtyCanvas() {} }, __dasiwaH3Forge: {
    references: () => refs, mode: () => mode, continuity: () => continuity, contextKey: () => 'context', duration: () => 5,
    updateReference: () => true, apply: entry => { applied.push(entry); return true; }, setStatus() {},
  } };
  let stored = JSON.stringify(prefs);
  const context = vm.createContext({ document, window: {}, Image: class {}, app: { registerExtension() {} },
    localStorage: { getItem: () => stored, setItem: (_, value) => { stored = value; } },
    setInterval: () => 1, clearInterval() {}, setTimeout: callback => { timers.set(++timerId, callback); return timerId; }, clearTimeout: id => timers.delete(id),
    api: { apiURL: p => p, fetchApi: async (url, options) => {
      if (url.endsWith('/models')) return { ok: true, json: async () => ({ models: [{ id: 'local:test', label: 'Test' }], creativity: ['balanced'], default_creativity: 'balanced', detail_levels: {}, default_detail: 5, shot_counts: ['Auto'] }) };
      const request = JSON.parse(options.body); requests.push(request);
      if (url.endsWith('/cast') && castReply) return castReply;
      if (url.endsWith('/people') && peopleReply) return peopleReply;
      if (beforeReply) await beforeReply(document);
      return { ok: true, json: async () => ({ mode, continuity: !!continuity, source_id: continuity?.clip_id, simple_prompt: 'Generated draft', fields: {}, model: 'local:test', easy: request.easy, vision: true, saw_images: request.see_pictures ? 1 : 0, unloaded: true, stats: { seconds: 1 } }) };
    } },
  });
  const state = fs.readFileSync(path.join(root, 'js/minimax_h3_forge_state.js'), 'utf8').replace(/^export /gm, '');
  const extension = fs.readFileSync(path.join(root, 'js/minimax_h3_forge.js'), 'utf8').replace(/^import .*;\n/gm, '');
  vm.runInContext(state + '\n' + extension, context);
  await context.window.DaSiWaH3Forge.open(node);
  const all = document.body.querySelectorAll('*');
  const button = name => all.find(c => c.tagName === 'BUTTON' && c.textContent === name);
  const choice = all.find(c => c.tagName === 'LABEL' && c.textContent.includes('Let the model see the pictures'));
  const visible = element => !!element && !element.hidden && (!element.parentElement || visible(element.parentElement));
  return { node, refs, requests, applied, all, button, choice, checkbox: choice?.children[0], visible,
    autoCast: all.find(c => c.tagName === 'LABEL' && c.textContent.includes('Read Cast automatically'))?.children[0],
    async readTimer() { const pending = [...timers.values()]; timers.clear(); for (const callback of pending) await callback(); },
    brief: all.find(c => c.tagName === 'TEXTAREA'), historyButton: () => all.find(c => c.title === 'Show this prompt; Apply to node to use it'),
    close: () => context.window.DaSiWaH3Forge.close(node) };
}

test('renumbering during a Cast read preserves the edited card now occupying that number', async () => {
  let d;
  d = await dialog({ refs: [{ ...image(), boxes: [{ n:1, box:[0.1,0.1,0.4,0.9] }, { n:2, box:[0.6,0.1,0.9,0.9] }] }],
    castReply: { ok:true, json:async () => {
      const select = d.all.find(c => c.tagName === 'SELECT' && c.title?.startsWith('Character 1: pick'));
      d.autoCast.checked = false;
      select.value = '2'; await select.dispatch('change');
      return { text:'Old automatic description.' };
    } } });
  d.node.properties.dasiwaH3ForgeCast = { 2: { text:'Edited adult in a dark coat.', edited:true, sig:'old' } };
  await d.readTimer();
  assert.equal(d.node.properties.dasiwaH3ForgeCast[1].text, 'Edited adult in a dark coat.');
  assert.equal(d.node.properties.dasiwaH3ForgeCast[1].edited, true);
  assert.equal(d.node.properties.dasiwaH3ForgeCast[2], undefined);
  d.close();
});

test('Generate skips stale automatic cards but preserves explicitly edited cards', async () => {
  for (const edited of [false, true]) {
    const d = await dialog({ prefs: { auto_cast: false } });
    d.node.properties.dasiwaH3ForgeCast = { 1: { text: 'An adult in the previous outfit.', sig: 'stale', edited } };
    d.brief.value = 'A new outfit'; await d.button('Generate').dispatch('click');
    const request = d.requests.find(r => r.mode === 'REF2VA');
    assert.equal(request.cast['1'], edited ? 'An adult in the previous outfit.' : undefined);
    d.close();
  }
});

test('Find people serializes Generate and rejects answers after a role change', async () => {
  let finish;
  const response = new Promise(resolve => { finish = resolve; });
  const d = await dialog({ prefs: { auto_cast: false }, peopleReply: { ok: true, json: () => response } });
  const finding = d.button('Find people').dispatch('click');
  await Promise.resolve(); await Promise.resolve();
  await d.button('Add character').dispatch('click');
  assert.equal(d.refs[0].boxes, undefined);
  d.brief.value = 'An idea';
  const generating = d.button('Generate').dispatch('click');
  await Promise.resolve();
  assert.equal(d.requests.some(r => r.mode === 'REF2VA'), false);
  d.refs[0].picture_kinds = ['pose'];
  finish({ people: [{ box: [0.1,0.1,0.9,0.9] }], seconds: 1, unloaded: true });
  await finding;
  assert.equal(d.refs[0].boxes, undefined);
  await d.readTimer(); await generating;
  assert.equal(d.requests.filter(r => r.mode === 'REF2VA').length, 1);
  d.close();
});

test('Cast rejects stale replies after an Idea change and permits manual edits during a read', async () => {
  let d;
  d = await dialog({ castReply: { ok: true, json: async () => {
    const area = d.all.find(c => c.attributes['aria-label'] === 'Character 1: description');
    assert.equal(area.disabled, false);
    d.brief.value = 'A changed outfit';
    d.autoCast.checked = false;
    return { text: 'Old outfit.' };
  } } });
  await d.readTimer();
  assert.equal(d.node.properties.dasiwaH3ForgeCast, undefined);
  d.close();
});

test('Cast autoload is default on, independent of draft vision, with a remembered opt-out', async () => {
  const d = await dialog({ castReply: { ok: true, json: async () => ({ text: 'An adult with dark hair.' }) } });
  assert.equal(d.autoCast.checked, true);
  assert.equal(d.checkbox.checked, false);
  await d.readTimer();
  assert.equal(d.requests.filter(r => r.character === 1).length, 1);
  assert.equal(d.node.properties.dasiwaH3ForgeCast[1].text, 'An adult with dark hair.');
  d.close();
  const off = await dialog({ prefs: { auto_cast: false } });
  assert.equal(off.autoCast.checked, false);
  await off.readTimer();
  assert.equal(off.requests.length, 0);
  off.close();
});

test('a non-vision reader stops the automatic queue without blocking normal drafting', async () => {
  const d = await dialog({ refs: [image(), { ...image('character-2'), path: 'b.png', item: { id: 'b' } }],
    castReply: { ok: false, json: async () => ({ error: 'no_vision', message: 'Write their description yourself.' }) } });
  await d.readTimer(); await d.readTimer();
  assert.equal(d.requests.filter(r => r.character).length, 1);
  assert.equal(d.node.properties.dasiwaH3ForgeCast, undefined);
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  assert.equal(d.button('Apply to node').disabled, false);
  d.close();
});

test('picture button clicks invalidate drafts and repainted Who buttons disable during generation', async () => {
  const d = await dialog({ beforeReply: async document => {
    const people = document.body.querySelectorAll('.refs .who button');
    assert.ok(people.length);
    assert.ok(people.every(b => b.disabled));
  } });
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  assert.equal(d.button('Apply to node').disabled, false);
  await d.button('Place').dispatch('click');
  assert.equal(d.button('Apply to node').disabled, true);
  assert.equal(d.all.find(c => c.tagName === 'PRE').hidden, true);
  await d.button('Regenerate').dispatch('click');
  assert.equal(d.requests.length, 2);
  d.close();
});

test('eligibility accepts exactly the backend EASY_ROLES whitelist', async () => {
  const roles = [...Array.from({ length: 32 }, (_, i) => `character-${i + 1}`), 'place', 'style', 'first-frame', 'last-frame', 'pose', 'custom', 'group-12', 'group-21', 'group-13', 'group-31', 'group-23', 'group-32', 'group-123'];
  for (const role of [...roles, 'character-0', 'character-33', 'group-321', 'unknown', '']) {
    const d = await dialog({ refs: [image(role)] });
    assert.equal(d.visible(d.choice), roles.includes(role), role);
    d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
    assert.equal(d.requests[0].easy, roles.includes(role), role);
    d.close();
  }
});

test('no references, base modes and continuity cannot opt into easy writer vision', async () => {
  for (const options of [{ refs: [] }, { mode: 'I2VA' }, { mode: 'T2VA' }, { continuity: { clip_id: 'c', use_references: true } }]) {
    const d = await dialog({ ...options, prefs: { see_pictures: true } });
    assert.equal(d.visible(d.choice), false);
    d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
    assert.equal(d.requests[0].easy, false);
    assert.equal(d.requests[0].see_pictures, false);
    d.close();
  }
});

test('history restores per-draft vision options and missing see_pictures defaults blind', async () => {
  for (const vision of [undefined, false, true]) {
    const options = { model: 'local:test', detail: 5, creativity: 'balanced', shots: 'Auto' };
    if (vision !== undefined) options.see_pictures = vision;
    const history = [{ mode: 'REF2VA', simple_prompt: 'Saved draft', fields: {}, brief: 'An idea', draftOptions: options }];
    const d = await dialog({ history, prefs: { see_pictures: !vision } });
    await d.historyButton().dispatch('click');
    assert.equal(d.checkbox.checked, !!vision);
    assert.equal(d.button('Apply to node').disabled, false);
    await d.button('Apply to node').dispatch('click');
    assert.equal(d.applied.length, 1);
    d.close();
  }
});

test('a vision history entry is incompatible when the current references are mixed', async () => {
  const history = [{ mode: 'REF2VA', simple_prompt: 'Vision draft', fields: {}, brief: 'An idea', draftOptions: { see_pictures: true, shots: 'Auto' } }];
  const d = await dialog({ refs: [image(), { kind: 'audio', instructions: '' }], history });
  await d.historyButton().dispatch('click');
  assert.equal(d.button('Apply to node').disabled, true);
  await d.button('Apply to node').dispatch('click');
  assert.equal(d.applied.length, 0);
  d.close();
});

test('labelling an initially unlabelled image enables vision choice without reopening', async () => {
  const ref = image();
  delete ref.easy_role;
  const reopened = await dialog({ refs: [ref] });
  assert.equal(reopened.visible(reopened.choice), false);
  const selector = reopened.all.find(c => c.dataset.kind === 'place');
  assert.ok(selector, 'unlabelled image still has a label editor');
  await selector.dispatch('click');
  assert.equal(reopened.visible(reopened.choice), true);
  reopened.checkbox.checked = true;
  reopened.brief.value = 'A place'; await reopened.button('Generate').dispatch('click');
  assert.equal(reopened.requests[0].easy, true);
  assert.equal(reopened.requests[0].see_pictures, true);
  reopened.close();
});

test('apply rejects a vision mismatch even without a dispatched change event', async () => {
  const d = await dialog();
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  const saved = d.node.properties.dasiwaH3ForgeHistory[0];
  assert.equal(JSON.parse(saved.forgeInputKey).length, 4, 'legacy input key shape is unchanged');
  d.checkbox.checked = true;
  await d.button('Apply to node').dispatch('click');
  assert.equal(d.applied.length, 0);
  d.close();
});

test('legacy history without draftOptions restores blind despite a vision preference', async () => {
  const history = [{ mode: 'REF2VA', simple_prompt: 'Old draft', fields: {}, brief: 'An idea' }];
  const d = await dialog({ history, prefs: { see_pictures: true } });
  await d.historyButton().dispatch('click');
  assert.equal(d.checkbox.checked, false);
  assert.equal(d.button('Apply to node').disabled, false);
  d.close();
});

test('changing the vision checkbox invalidates a generated draft and blocks apply', async () => {
  const d = await dialog();
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  assert.equal(d.button('Apply to node').disabled, false);
  d.checkbox.checked = true; await d.checkbox.dispatch('change');
  assert.equal(d.button('Apply to node').disabled, true);
  assert.equal(d.all.find(c => c.tagName === 'PRE').hidden, true);
  await d.button('Apply to node').dispatch('click');
  assert.equal(d.applied.length, 0);
  d.close();
});

test('mixed references hide the opt-out while keeping label editors and using normal vision', async () => {
  const d = await dialog({ refs: [image(), { kind: 'video', path: 'v.mp4', stream: 'video', instructions: '' }], prefs: { see_pictures: true } });
  assert.equal(d.visible(d.choice), false);
  assert.ok(d.all.some(c => c.attributes['aria-label']?.endsWith(' is')));
  assert.ok(d.all.some(c => d.visible(c) && c.textContent.includes('Pictures are automatically sent to a model capable')));
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  assert.equal(d.requests[0].easy, false); assert.equal(d.requests[0].see_pictures, false);
  d.close();
});
