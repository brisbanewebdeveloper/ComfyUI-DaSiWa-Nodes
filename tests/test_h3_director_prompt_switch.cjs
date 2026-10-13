const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.resolve(__dirname, '../js/minimax_h3_director.js'), 'utf8');
function region(start, end) {
  assert.ok(source.includes(start) && source.includes(end));
  return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}
function harness(continuing) {
  const c = { session: 'test', source_kind: 'checkpoint', source_id: continuing ? 'old-take' : '', source_video_id: '', continuation_prompt: 'Older continuation Forge draft' };
  const context = vm.createContext({
    builderState: { simple_prompt: 'Older new-clip Forge draft' },
    continuityState: () => c, ensureContinuitySession: () => c,
    isContinuing: () => !!(c.source_kind === 'video' ? c.source_video_id : c.source_id),
    mode: () => 'REF2VA', sourceRevision: 0, pendingResume: '',
    emit() { context.saved = JSON.parse(JSON.stringify({ builder: context.builderState, continuity: c })); },
    render() {}, setStatus() {},
  });
  vm.runInContext(region('  const activePrompt =', '  let continuityClips =') + '\n' +
    region('  function clearContinuitySource()', '  function continuityGraphReady()') +
    '\nglobalThis.promptAPI = { activePrompt, setActivePrompt, setEditedPrompt, clearContinuitySource, selectContinuitySource };', context);
  return { context, c, api: context.promptAPI };
}
for (const continuing of [false, true]) {
  for (const edited of ['User edit\nwith <Picture 1> and music', '', '  exact whitespace  \n']) {
    test(`manual prompt survives source round trip (continuing=${continuing}, text=${JSON.stringify(edited)})`, () => {
      const { api, context, c } = harness(continuing);
      api.setEditedPrompt(edited);
      if (continuing) api.clearContinuitySource(); else api.selectContinuitySource('another-take');
      assert.equal(api.activePrompt(), edited);
      if (continuing) api.selectContinuitySource('old-take'); else api.clearContinuitySource();
      assert.equal(api.activePrompt(), edited);
      assert.equal(context.saved.builder.simple_prompt, edited);
      assert.equal(context.saved.continuity.continuation_prompt, edited);
      assert.equal(c.session, 'test');
    });
  }
}
test('manual edits in either context also survive imported-video selection', () => {
  const { api, c } = harness(false);
  api.setEditedPrompt('Edited before import');
  c.source_kind = 'video'; c.source_video_id = 'imported-video';
  assert.equal(api.activePrompt(), 'Edited before import');
  api.setEditedPrompt('Edited continuation');
  api.clearContinuitySource();
  assert.equal(api.activePrompt(), 'Edited continuation');
});
test('both textarea edit handlers persist the shared manual prompt', () => {
  assert.match(source, /onChange: setEditedPrompt, fieldKey: "simple_prompt"/);
  assert.match(source, /area\.oninput = \(\) => setEditedPrompt\(area\.value\)/);
});
test('programmatic continuation templates still leave the base prompt untouched', () => {
  const { api, context } = harness(true);
  api.setActivePrompt('Continuation structure');
  assert.equal(api.activePrompt(), 'Continuation structure');
  assert.equal(context.builderState.simple_prompt, 'Older new-clip Forge draft');
});
test('source selection alone retains existing distinct saved drafts', () => {
  const { api } = harness(false);
  api.selectContinuitySource('old-take');
  assert.equal(api.activePrompt(), 'Older continuation Forge draft');
  api.clearContinuitySource();
  assert.equal(api.activePrompt(), 'Older new-clip Forge draft');
});
