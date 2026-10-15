const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const file = path.join(__dirname, '../js/refmod_export.js');
const load = () => import('data:text/javascript;base64,' + Buffer.from(fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '').toString('base64'));
const graph = () => ({
  1: { class_type: 'MiniMaxH3Director', inputs: {} },
  2: { class_type: 'VAELoader', inputs: { vae_name: 'visual.safetensors' } },
  3: { class_type: 'VAELoader', inputs: { vae_name: 'audio.safetensors' } },
  4: { class_type: 'MiniMaxH3DirectorGuide', inputs: { guide: ['1', 0], vae: ['2', 0], audio_vae: ['3', 0], clip: ['8', 0] } },
  8: { class_type: 'CLIPLoader', inputs: {} },
  9: { class_type: 'KSampler', inputs: { positive: ['4', 0] } },
});

test('selects checked references in source order and edits descriptions on a clone', async () => {
  const { selectDaSiWaRefModSnapshot: select } = await load();
  assert.equal(typeof select, 'function');
  const source = { items: [{ type: 'text', value: 'prompt' }, { id: 'a', type: 'image', value: 'a.png', prompt: 'old', enabled: false }, { id: 'b', type: 'audio', value: 'b.wav' }], refmods: [{ name: 'bundle', description: 'old', strength: 0 }] };
  const original = structuredClone(source);
  const selected = select(source, [{ kind: 'refmods', index: 0, selected: true, description: 'saved edit' }, { kind: 'items', index: 2, selected: false }, { kind: 'items', index: 1, selected: true, description: '<edited>' }]);
  assert.deepEqual(selected.items, [{ ...source.items[1], description: '<edited>' }]);
  assert.deepEqual(selected.refmods, [{ ...source.refmods[0], description: 'saved edit' }]);
  assert.deepEqual(source, original);
});

test('selects only needed native modality loaders and rejects unsafe or empty inputs', async () => {
  const { buildDaSiWaRefModExportPrompt: build } = await load();
  const output = graph();
  const saved = { items: [], refmods: [{ name: 'bundle', enabled: false }] };
  assert.equal(Object.keys(build({}, 'missing', saved, 'saved').prompt).length, 1);
  for (const item of [{ type: 'audio', value: 'a.wav' }, { type: 'video', value: 'v.mp4', media_mode: 'audio' }]) {
    const result = build(output, '1', { items: [item] }, 'audio');
    assert.deepEqual(Object.keys(result.prompt).sort(), ['3', result.exportId].sort());
  }
  const both = build(output, '1', { items: [{ type: 'video', value: 'v.mp4', audio: 'a.wav' }] }, 'both');
  assert.deepEqual(Object.keys(both.prompt).sort(), ['2', '3', both.exportId].sort());
  assert.throws(() => build(output, '1', { items: [] }, 'empty'), /select|reference/i);
  assert.throws(() => build(output, '1', { items: 'bad' }, 'bad'), /timeline|items/i);
  assert.throws(() => build(output, '1', saved, 'bad', { concept_type: 'training' }), /concept/i);
  const raw = { items: [{ type: 'image', value: 'p.png' }] };
  assert.throws(() => build({}, '1', raw, 'x'), /exactly one Director Guide/i);
  const duplicate = graph(); duplicate[5] = structuredClone(duplicate[4]);
  assert.throws(() => build(duplicate, '1', raw, 'x'), /exactly one Director Guide/i);
  output[2].class_type = 'CustomLoader';
  assert.throws(() => build(output, '1', raw, 'x'), /direct native VAELoader/i);
  output[2] = graph()[2]; output[2].inputs.extra = ['9', 0];
  assert.throws(() => build(output, '1', raw, 'x'), /direct native VAELoader/i);
});

test('queues explicit isolated payload and waits through queued history for namespaced result', async () => {
  const { queueDaSiWaRefModExport: queue } = await load();
  assert.equal(typeof queue, 'function');
  let exportId, reads = 0;
  const calls = [];
  const result = { name: 'portrait', members: 2, counts: { image: 2 }, warnings: ['Grid snapped'], message: 'Saved' };
  const api = { clientId: 'client', fetchApi: async (url, options) => {
    calls.push(url);
    if (url === '/prompt') {
      const body = JSON.parse(options.body);
      assert.equal(options.method, 'POST');
      assert.equal(options.headers['Content-Type'], 'application/json');
      assert.equal(body.client_id, 'client');
      assert.deepEqual(body.extra_data.extra_pnginfo.workflow, { nodes: [] });
      assert.deepEqual(Object.values(body.prompt).map(n => n.class_type).sort(), ['DaSiWaH3RefModExport', 'VAELoader'].sort());
      exportId = Object.keys(body.prompt).find(id => body.prompt[id].class_type === 'DaSiWaH3RefModExport');
      return { ok: true, json: async () => ({ prompt_id: 'job/id' }) };
    }
    assert.equal(url, '/history/job%2Fid');
    return { ok: true, json: async () => ++reads < 3 ? {} : ({ 'job/id': { status: { completed: true }, outputs: { [exportId]: { dasiwa_refmod_export: [result] } } } }) };
  } };
  assert.deepEqual(await queue({ api, app: { graphToPrompt: async () => ({ output: graph(), workflow: { nodes: [] } }) }, node: { id: 1 }, snapshot: { items: [{ type: 'image', value: 'p.png' }] }, pollMs: 0 }), result);
  assert.equal(reads, 3);
  assert.equal(calls.filter(url => url === '/prompt').length, 1);
});

test('queue reports server and execution failures and aborts monitoring without interrupting', async () => {
  const { queueDaSiWaRefModExport: queue } = await load();
  const options = { app: { graphToPrompt: async () => ({ output: {}, workflow: {} }) }, node: { id: 1 }, snapshot: { items: [], refmods: [{ name: 'saved' }] }, pollMs: 0 };
  await assert.rejects(() => queue({ ...options, api: { fetchApi: async () => ({ ok: false, status: 400, json: async () => ({ error: { message: 'Invalid VAE' }, node_errors: { 2: 'unsupported' } }) }) } }), /Invalid VAE.*unsupported/);
  for (const entry of [{ status: { status_str: 'error', messages: [['execution_error', { exception_message: 'disk full' }]] } }, { status: { completed: true }, outputs: {} }]) {
    await assert.rejects(() => queue({ ...options, api: { fetchApi: async url => ({ ok: true, json: async () => url === '/prompt' ? { prompt_id: 'p' } : { p: entry } }) } }), /disk full|without.*result/i);
  }
  const controller = new AbortController(); const calls = [];
  await assert.rejects(() => queue({ ...options, signal: controller.signal, api: { fetchApi: async url => {
    calls.push(url);
    if (url !== '/prompt') controller.abort();
    return { ok: true, json: async () => url === '/prompt' ? { prompt_id: 'p' } : {} };
  } } }), { name: 'AbortError' });
  assert.deepEqual(calls, ['/prompt', '/history/p']);
});

const tick = () => new Promise(resolve => setImmediate(resolve));
const response = body => ({ ok: true, json: async () => body });

// A minimal DOM harness exercises dialog behavior without adding a test dependency.
class Element extends EventTarget {
  constructor(tag, doc) { super(); this.tagName = tag; this.doc = doc; this.children = []; this.attributes = {}; this.value = ''; this.disabled = false; this.checked = false; this.currentTime = 0; this.duration = 20; }
  append(...nodes) { for (const node of nodes) { this.children.push(node); node.parent = this; } }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  removeAttribute(key) { delete this.attributes[key]; if (key === 'src') this.src = ''; }
  querySelectorAll(selector) { const nodes = this.children.flatMap(n => [n, ...n.querySelectorAll(selector)]); return nodes.filter(n => selector.includes(n.tagName) && (!selector.includes(':not') || !n.disabled)); }
  focus() { this.doc.activeElement = this; }
  showModal() { this.open = true; }
  close() { this.open = false; this.dispatchEvent(new Event('close')); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(n => n !== this); }
  pause() { this.paused = true; }
  load() { this.unloaded = true; }
}
function dom() {
  const doc = { createElement: tag => new Element(tag, doc) };
  doc.body = new Element('body', doc); doc.head = new Element('head', doc);
  doc.activeElement = new Element('button', doc);
  doc.activeElement.isConnected = true;
  return doc;
}
function key(target, key, shiftKey = false) {
  const event = new Event('keydown', { cancelable: true });
  Object.assign(event, { key, shiftKey }); target.dispatchEvent(event); return event;
}

test('overlay lists hidden references, edits clone, bounds preview, traps focus and cleans up', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  assert.equal(typeof open, 'function');
  const doc = dom(); global.document = doc;
  const prior = doc.activeElement;
  const snapshot = { items: [{ type: 'text', value: 'skip' }, { type: 'video', value: '<movie>.mp4', enabled: false, trim_start: 2, trim_end: 4, prompt: 'old' }], refmods: [{ name: '<bundle>', strength: 0, enabled: false, description: 'saved' }] };
  const original = structuredClone(snapshot); let posted, saved;
  const cleanup = open({ snapshot, node: { id: 1 }, app: { graphToPrompt: async () => ({ output: graph(), workflow: {} }) }, viewUrl: value => '/view?' + value, onSaved: result => { saved = result; }, api: { clientId: 'c', fetchApi: async (url, options) => {
    if (url === '/dasiwa/refmods') return { ok: true, json: async () => [] };
    if (url === '/prompt') { posted = JSON.parse(options.body); return { ok: true, json: async () => ({ prompt_id: 'p' }) }; }
    const id = Object.keys(posted.prompt).find(id => posted.prompt[id].class_type === 'DaSiWaH3RefModExport');
    return { ok: true, json: async () => ({ p: { status: { completed: true }, outputs: { [id]: { dasiwa_refmod_export: [{ name: 'done', members: 2, counts: { video: 1 }, warnings: ['snapped'] }] } } } }) };
  } } });
  const dialog = doc.body.children[0];
  assert.equal(dialog.attributes['aria-modal'], 'true');
  const checkboxes = dialog.querySelectorAll('input').filter(n => n.type === 'checkbox');
  assert.equal(checkboxes.length, 2); assert.ok(checkboxes.every(n => n.checked));
  const areas = dialog.querySelectorAll('textarea'); areas[1].value = '<new description>';
  const video = dialog.querySelectorAll('video')[0];
  video.dispatchEvent(new Event('loadedmetadata')); assert.equal(video.currentTime, 2);
  video.currentTime = 5; video.dispatchEvent(new Event('timeupdate')); assert.equal(video.paused, true); assert.equal(video.currentTime, 4);
  let time = 4, seeks = 0;
  Object.defineProperty(video, 'currentTime', { get: () => time, set: value => { time = value; seeks++; } });
  video.dispatchEvent(new Event('timeupdate')); assert.equal(seeks, 0, 'a timeupdate at the crop end must not start another seek');
  await tick();
  const buttons = dialog.querySelectorAll('button');
  buttons.at(-1).focus(); assert.equal(key(dialog, 'Tab').defaultPrevented, true); assert.equal(doc.activeElement, buttons[0]);
  const create = buttons.find(n => n.textContent === 'Create RefMod');
  const click = new Event('click'); create.dispatchEvent(click); create.dispatchEvent(click);
  assert.equal(create.disabled, true);
  await new Promise(resolve => setImmediate(resolve));
  const exported = Object.values(posted.prompt).find(n => n.class_type === 'DaSiWaH3RefModExport');
  assert.equal(JSON.parse(exported.inputs.timeline_data).items[0].description, '<new description>');
  assert.equal(JSON.parse(exported.inputs.timeline_data).items[0].prompt, 'old');
  assert.equal(saved.name, 'done');
  assert.deepEqual(snapshot, original);
  key(dialog, 'Escape'); cleanup();
  assert.equal(doc.body.children.length, 0); assert.equal(doc.head.children.length, 0);
  assert.equal(video.unloaded, true); assert.equal(doc.activeElement, prior);
  delete global.document;
});

test('attached audio previews match the exported stream for A and V+A', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  for (const media_mode of ['audio', 'video_audio']) {
    const doc = dom(); global.document = doc;
    const cleanup = open({ api: {}, app: {}, node: { id: 1 },
      snapshot: { items: [{ type: 'video', value: 'video.mp4', audio: 'attached.wav', media_mode, trim_start: 1, trim_end: 2 }], refmods: [] },
      viewUrl: value => value });
    const dialog = doc.body.children[0];
    const audios = dialog.querySelectorAll('audio');
    assert.equal(audios.length, 1);
    assert.equal(audios[0].src, 'attached.wav');
    const videos = dialog.querySelectorAll('video');
    assert.equal(videos.length, media_mode === 'audio' ? 0 : 1);
    if (videos.length) assert.equal(videos[0].muted, true);
    cleanup(); delete global.document;
  }
});

test('rejects malformed guide links rather than guessing a loader topology', async () => {
  const { buildDaSiWaRefModExportPrompt: build } = await load();
  const output = graph(); output[4].inputs.guide = '1';
  assert.throws(() => build(output, '1', { items: [{ type: 'image', value: 'p.png' }] }, 'x'), /exactly one Director Guide/i);
});

test('history network failure identifies the queued job without claiming cancellation', async () => {
  const { queueDaSiWaRefModExport: queue } = await load();
  await assert.rejects(() => queue({ app: { graphToPrompt: async () => ({ output: {}, workflow: {} }) }, node: { id: 1 }, snapshot: { items: [], refmods: [{ name: 'saved' }] }, api: { fetchApi: async url => {
    if (url === '/prompt') return { ok: true, json: async () => ({ prompt_id: 'pending-job' }) };
    throw new Error('connection lost');
  } } }), /pending-job.*may still.*connection lost/i);
});

test('closing during serialization prevents queue submission and removes all handlers', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  const doc = dom(); global.document = doc;
  let serialized, posts = 0;
  const cleanup = open({ api: { fetchApi: async url => { if (url === '/dasiwa/refmods') return { ok: true, json: async () => [] }; posts++; } }, app: { graphToPrompt: () => new Promise(resolve => { serialized = resolve; }) }, node: { id: 1 }, snapshot: { items: [{ type: 'audio', value: 'a.wav', trim_start: 1, trim_end: 3 }] }, viewUrl: value => value });
  const dialog = doc.body.children[0], audio = dialog.querySelectorAll('audio')[0];
  await new Promise(resolve => setImmediate(resolve));
  dialog.querySelectorAll('button').at(-1).dispatchEvent(new Event('click'));
  cleanup(); serialized({ output: {}, workflow: {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(posts, 0); assert.equal(doc.body.children.length, 0);
  audio.currentTime = 8; audio.dispatchEvent(new Event('timeupdate')); assert.equal(audio.currentTime, 8);
  delete global.document;
});

test('overlay locks retry after a history failure while its job may still be queued', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  const doc = dom(); global.document = doc;
  const statuses = []; let posts = 0;
  const cleanup = open({ api: { fetchApi: async url => {
    if (url === '/dasiwa/refmods') return { ok: true, json: async () => [] };
    if (url === '/prompt') { posts++; return { ok: true, json: async () => ({ prompt_id: 'still-queued' }) }; }
    throw new Error('offline');
  } }, app: { graphToPrompt: async () => ({ output: {}, workflow: {} }) }, node: { id: 1 }, snapshot: { items: [], refmods: [{ name: 'saved' }] }, viewUrl: value => value, onStatus: value => statuses.push(value) });
  const dialog = doc.body.children[0], create = dialog.querySelectorAll('button').at(-1);
  await new Promise(resolve => setImmediate(resolve));
  create.dispatchEvent(new Event('click')); await new Promise(resolve => setImmediate(resolve));
  assert.equal(create.disabled, true);
  create.dispatchEvent(new Event('click')); await new Promise(resolve => setImmediate(resolve));
  assert.equal(posts, 1); assert.match(statuses.at(-1), /still-queued.*may still/);
  cleanup(); delete global.document;
});

test('library picker explicitly opens metadata, selects individual members and saves without VAEs', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  global.document = dom();
  const source = { items: [{ type: 'image', value: 'current.png' }], refmods: [{ name: 'aggregate' }] };
  const original = structuredClone(source);
  const detail = { name: 'saved / <bundle>', description: 'Bundle notes', concept_type: 'identity', members: [{ kind: 'image', description: 'first', original_path: 'never-preview.png' }, { kind: 'audio', description: 'second' }, { kind: 'video', description: 'third' }] };
  const before = structuredClone(detail), calls = [];
  let finishDetail, posted, exportId;
  const cleanup = open({ snapshot: source, node: { id: 1 }, viewUrl: value => value,
    app: { graphToPrompt: async () => ({ output: {}, workflow: {} }) }, api: { fetchApi: async (url, options) => {
      calls.push(url);
      if (url === '/dasiwa/refmods') return response([{ name: detail.name }]);
      if (url.startsWith('/dasiwa/refmods/detail?')) return new Promise(resolve => { finishDetail = () => resolve(response(detail)); });
      if (url === '/prompt') {
        posted = JSON.parse(options.body);
        exportId = Object.keys(posted.prompt)[0];
        return response({ prompt_id: 'p' });
      }
      return response({ p: { outputs: { [exportId]: { dasiwa_refmod_export: [{ name: 'new-bundle', members: 2 }] } } } });
    } } });
  try {
    const dialog = document.body.children[0];
    const library = dialog.querySelectorAll('select')[0];
    const create = dialog.querySelectorAll('button').at(-1);
    create.dispatchEvent(new Event('click')); assert.deepEqual(calls, ['/dasiwa/refmods']);
    await tick();
    assert.equal(library.children[1].textContent, detail.name);
    dialog.querySelectorAll('input').find(input => input.type === 'text').value = 'must-clear';
    library.value = detail.name; library.dispatchEvent(new Event('change'));
    assert.equal(create.disabled, true);
    create.dispatchEvent(new Event('click')); library.dispatchEvent(new Event('change'));
    assert.deepEqual(calls, ['/dasiwa/refmods', `/dasiwa/refmods/detail?name=${encodeURIComponent(detail.name)}`]);
    finishDetail(); await tick();
    assert.equal(dialog.querySelectorAll('img,video,audio').length, 0);
    assert.match(dialog.querySelectorAll('p').map(node => node.textContent).join(' '), /stored as encoded latents.*usable for generation/);
    assert.equal(create.textContent, 'Save Changes');
    assert.equal(dialog.querySelectorAll('input').find(input => input.type === 'text').value, '');
    assert.equal(dialog.querySelectorAll('select')[1].value, 'identity');
    const checks = dialog.querySelectorAll('input').filter(input => input.type === 'checkbox');
    assert.equal(checks.length, 3); assert.ok(checks.every(input => input.checked));
    const areas = dialog.querySelectorAll('textarea');
    assert.deepEqual(areas.map(area => area.value), ['Bundle notes', 'first', 'second', 'third']);
    checks.forEach(check => { check.checked = false; });
    create.dispatchEvent(new Event('click')); await tick();
    assert.equal(posted, undefined);
    assert.match(dialog.querySelectorAll('p').map(node => node.textContent).join(' '), /Select at least one/);
    checks[0].checked = checks[2].checked = true;
    areas[1].value = '  edited first  '; areas[3].value = '';
    create.dispatchEvent(new Event('click')); await tick();
    assert.equal(Object.keys(posted.prompt).length, 1);
    const inputs = posted.prompt[exportId].inputs;
    assert.deepEqual(JSON.parse(inputs.timeline_data), { items: [], refmods: [{ name: detail.name, members: [{ index: 0, description: '  edited first  ' }, { index: 2, description: '' }] }] });
    assert.equal(inputs.name, ''); assert.equal(inputs.description, 'Bundle notes'); assert.equal(inputs.concept_type, 'identity');
    assert.deepEqual(source, original); assert.deepEqual(detail, before);
  } finally { cleanup(); delete global.document; }
});

test('library failures are visible and preserve current draft; closing ignores late loads', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  for (const failure of ['list', 'detail', 'malformed', 'close', 'removed']) {
    global.document = dom();
    let resolveDetail, detailSignal, removed = false;
    const statuses = [], calls = [];
    const cleanup = open({ snapshot: { items: [{ type: 'image', value: 'keep.png' }] }, viewUrl: value => value,
      node: { id: 1 }, app: {}, isRemoved: () => removed, onStatus: message => statuses.push(message), api: { fetchApi: async (url, options) => {
        calls.push(url);
        if (url === '/dasiwa/refmods') {
          if (failure === 'list') throw new Error('list offline');
          return response([{ name: 'saved' }]);
        }
        if (failure === 'detail') return { ok: false, status: 404, json: async () => ({ error: 'gone' }) };
        if (failure === 'malformed') return response({ name: 'saved', members: [] });
        detailSignal = options.signal;
        return new Promise(resolve => { resolveDetail = resolve; });
      } } });
    try {
      const dialog = document.body.children[0], create = dialog.querySelectorAll('button').at(-1);
      await tick();
      const areas = dialog.querySelectorAll('textarea'); areas[0].value = 'keep bundle'; areas[1].value = 'keep member';
      if (failure !== 'list') {
        const library = dialog.querySelectorAll('select')[0]; library.value = 'saved'; library.dispatchEvent(new Event('change'));
        await tick();
      }
      if (failure === 'close' || failure === 'removed') {
        const count = statuses.length;
        if (failure === 'close') { cleanup(); assert.equal(detailSignal.aborted, true); } else removed = true;
        resolveDetail(response({ name: 'saved', description: 'late', concept_type: 'style', members: [{ kind: 'audio', description: 'late' }] }));
        await tick();
        assert.equal(statuses.length, count);
        assert.equal(areas[0].value, 'keep bundle');
        assert.equal(dialog.querySelectorAll('input').filter(input => input.type === 'checkbox').length, 1);
      } else {
        assert.match(statuses.at(-1), /Could not.*(offline|404|Invalid)/);
        assert.equal(create.disabled, false);
        assert.equal(dialog.querySelectorAll('img')[0].src, 'keep.png');
        assert.deepEqual(areas.map(area => area.value), ['keep bundle', 'keep member']);
      }
      assert.equal(calls.includes('/prompt'), false);
    } finally { cleanup(); delete global.document; }
  }
});

test('placeholder restores media-line creation view after automatic RefMod loading', async () => {
  const { openDaSiWaRefModExport: open } = await load();
  global.document = dom();
  const source = { items: [{ type: 'image', value: 'line.png', prompt: 'line description' }], refmods: [] };
  const original = structuredClone(source);
  const cleanup = open({ snapshot: source, node: { id: 1 }, app: {}, viewUrl: value => value,
    api: { fetchApi: async url => response(url === '/dasiwa/refmods' ? [{ name: 'saved' }] :
      { name: 'saved', description: 'bundle', concept_type: 'identity', members: [{ kind: 'audio', description: 'voice' }] }) } });
  try {
    await tick();
    const dialog = document.body.children[0], library = dialog.querySelectorAll('select')[0];
    const create = dialog.querySelectorAll('button').at(-1);
    assert.equal(dialog.querySelectorAll('button').some(button => button.textContent === 'Open'), false);
    library.value = 'saved'; library.dispatchEvent(new Event('change')); await tick();
    assert.equal(create.textContent, 'Save Changes');
    assert.equal(dialog.querySelectorAll('img').length, 0);
    library.value = ''; library.dispatchEvent(new Event('change'));
    assert.equal(create.textContent, 'Create RefMod');
    assert.equal(create.disabled, false);
    assert.equal(dialog.querySelectorAll('img')[0].src, 'line.png');
    assert.deepEqual(dialog.querySelectorAll('textarea').map(area => area.value), ['', 'line description']);
    assert.equal(dialog.querySelectorAll('select')[1].value, 'generic');
    assert.deepEqual(source, original);
    library.value = 'saved'; library.dispatchEvent(new Event('change')); await tick();
    assert.equal(create.textContent, 'Save Changes');
  } finally { cleanup(); delete global.document; }
});

test('explicit saved member selections reject empty or invalid indices and descriptions', async () => {
  const { buildDaSiWaRefModExportPrompt: build } = await load();
  for (const members of [[], null, [{ index: -1, description: '' }], [{ index: 0.5, description: '' }], [{ index: 0, description: 4 }]]) {
    assert.throws(() => build({}, 1, { items: [], refmods: [{ name: 'saved', members }] }, 'invalid'), /member/);
  }
});

test('builds unique isolated export with exact snapshot and metadata without mutation', async () => {
  const { buildDaSiWaRefModExportPrompt: build } = await load();
  assert.equal(typeof build, 'function');
  const output = graph();
  const snapshot = { mode: 'T2VA', items: [{ id: 'hidden', type: 'image', value: 'p.png', enabled: false, prompt: 'identity', order: 7 }], refmods: [{ name: 'saved', strength: 0, enabled: false }] };
  const original = structuredClone({ output, snapshot });
  const result = build(output, '1', snapshot, 'request', { name: 'Portrait', description: 'Bundle text', concept_type: 'identity' });
  assert.deepEqual(Object.keys(result.prompt).sort(), ['2', result.exportId].sort());
  const exported = result.prompt[result.exportId];
  assert.equal(exported.class_type, 'DaSiWaH3RefModExport');
  assert.deepEqual(JSON.parse(exported.inputs.timeline_data), snapshot);
  assert.deepEqual(exported.inputs.vae, ['2', 0]);
  assert.equal(exported.inputs.request_id, 'request');
  assert.equal(exported.inputs.name, 'Portrait');
  assert.equal(exported.inputs.description, 'Bundle text');
  assert.equal(exported.inputs.concept_type, 'identity');
  assert.deepEqual({ output, snapshot }, original);
  assert.notEqual(build(output, '1', snapshot, 'other').exportId, result.exportId);
});
