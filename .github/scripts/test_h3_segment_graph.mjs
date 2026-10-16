import assert from 'node:assert/strict';
import { test } from 'node:test';
import { installSegmentGraphValidation } from '../../js/minimax_h3_continuity.js';

function segmentPrompt(prefix = '') {
  const id = name => prefix + name;
  return {
    [id('director')]: { class_type: 'MiniMaxH3Director', inputs: { timeline_data: JSON.stringify({ continuity: { version: 3, output_mode: 'segment', capture: true } }) } },
    [id('guide')]: { class_type: 'MiniMaxH3DirectorGuide', inputs: { guide: [id('director'), 0] } },
    [id('prepare')]: { class_type: 'DaSiWaH3ContinuitySegmentPrepare', inputs: { context: [id('guide'), 2] } },
    [id('decode')]: { class_type: 'DaSiWaH3ContinuitySegmentDecode', inputs: { latent: [id('prepare'), 0], context: [id('prepare'), 1] } },
    [id('export')]: { class_type: 'DaSiWa_EnhancedVideoCombine', inputs: { images: [id('decode'), 0] } },
    [id('publish')]: { class_type: 'DaSiWaH3ContinuityPublish', inputs: { filename: [id('export'), 1], ticket: [id('prepare'), 2] } },
  };
}

for (const prefix of ['', '1512:']) for (const upscale of [false, true]) {
  test(`compiled ${prefix ? 'subgraph' : 'root'} prompt, upscale ${upscale}`, async () => {
    const output = segmentPrompt(prefix);
    if (upscale) {
      output[prefix + 'upscale'] = { class_type: 'DaSiWaH3TiledUpscale', inputs: { latent: [prefix + 'prepare', 0], continuity_context: [prefix + 'prepare', 1] } };
      output[prefix + 'decode'].inputs.latent = [prefix + 'upscale', 0];
    }
    const result = { output, workflow: { nodes: [{ type: 'subgraph-host' }] } };
    const app = { async graphToPrompt(...args) { assert.equal(this, app); assert.deepEqual(args, [42]); return result; } };
    installSegmentGraphValidation(app);
    const installed = app.graphToPrompt;
    installSegmentGraphValidation(app);
    assert.equal(app.graphToPrompt, installed);
    assert.equal(await app.graphToPrompt(42), result);
  });
}

test('rejects missing segment chain without falling back to full timeline', async () => {
  const output = segmentPrompt();
  delete output.prepare;
  const app = { async graphToPrompt() { return { output }; } };
  installSegmentGraphValidation(app);
  await assert.rejects(app.graphToPrompt(), /No full-timeline fallback/);
});

test('requires each Director to have its own connected segment chain', async () => {
  const output = { ...segmentPrompt(), other: segmentPrompt().director };
  const app = { async graphToPrompt() { return { output }; } };
  installSegmentGraphValidation(app);
  await assert.rejects(app.graphToPrompt(), /No full-timeline fallback/);
});

test('full output and ordinary workflow saving need no segment chain', async () => {
  const output = { director: { class_type: 'MiniMaxH3Director', inputs: { timeline_data: '{"continuity":{"output_mode":"full"}}' } } };
  const app = { graph: { serialize: () => ({ nodes: [] }) }, async graphToPrompt() { return { output }; } };
  const serialize = app.graph.serialize;
  installSegmentGraphValidation(app);
  assert.equal(app.graph.serialize, serialize);
  assert.deepEqual(app.graph.serialize(), { nodes: [] });
  assert.equal((await app.graphToPrompt()).output, output);
});

for (const [name, continuity, mode, active] of [
  ['capture disabled', { capture: false }, 'T2VA', false],
  ['cleared checkpoint overrides stale continue', { capture: false, operation: 'continue', source_id: '' }, 'T2VA', false],
  ['cleared video ignores checkpoint source', { capture: false, operation: 'continue', source_kind: 'video', source_id: 'old', source_video_id: '' }, 'T2VA', false],
  ['legacy inactive source', { version: 2, operation: 'new', source_id: 'old' }, 'T2VA', false],
  ['unversioned inactive source', { version: undefined, operation: 'new', source_id: 'old' }, 'T2VA', false],
  ['Image Inpaint without source', { capture: true }, 'Image Inpaint', false],
  ['active capture', { capture: true }, 'T2VA', true],
  ['checkpoint continuation without capture', { capture: false, operation: 'new', source_id: 'take' }, 'T2VA', true],
  ['video continuation without capture', { capture: false, operation: 'new', source_kind: 'video', source_video_id: 'video' }, 'T2VA', true],
  ['legacy continuation without capture', { version: 2, capture: false, operation: 'continue', source_id: 'take' }, 'T2VA', true],
  ['Image Inpaint continuation remains active', { capture: false, source_id: 'take' }, 'Image Inpaint', true],
]) {
  test(`segment validation gating: ${name}`, async () => {
    const output = { director: segmentPrompt().director };
    output.director.inputs.mode = mode;
    output.director.inputs.timeline_data = JSON.stringify({ continuity: { version: 3, output_mode: 'segment', ...continuity } });
    const result = { output };
    const app = { async graphToPrompt() { return result; } };
    installSegmentGraphValidation(app);
    if (active) await assert.rejects(app.graphToPrompt(), /No full-timeline fallback/);
    else assert.equal(await app.graphToPrompt(), result);
  });
}

test('propagates prompt compilation errors', async () => {
  const error = new Error('compile failed');
  const app = { async graphToPrompt() { throw error; } };
  installSegmentGraphValidation(app);
  await assert.rejects(app.graphToPrompt(), e => e === error);
});
