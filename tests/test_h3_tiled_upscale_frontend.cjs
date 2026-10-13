const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const TYPE = 'DaSiWaH3TiledUpscale';

function harness() {
  let extension;
  const source = fs.readFileSync(path.join(__dirname, '../js/minimax_h3_tiled_upscale.js'), 'utf8');
  // Load the real extension with only ComfyUI's app import replaced by a stub.
  vm.runInNewContext(source.replace(/^import \{ app \} from "\.\.\/\.\.\/scripts\/app\.js";\n/, ''), {
    app: { registerExtension(value) { extension = value; } },
  });
  const data = { name: TYPE, input: { required: {}, optional: {
    memory_budget_mb: ['INT', { default: 0 }],
    enforce_memory_budget: ['BOOLEAN', { default: true }],
    memory_reserve_pct: ['FLOAT', { default: 5, min: 0, max: 100, step: 5, tooltip: 'Backend tooltip' }],
  } } };
  extension.beforeRegisterNodeDef(function NodeType() {}, data);
  function widget(name, value, type = 'number') {
    return {
      name, value, type, options: { serialize: true },
      draw() { return 'native drawing'; }, computeSize() { return [220, 20]; },
      serializeValue() { return this.value; },
    };
  }
  function node() {
    const result = {
      comfyClass: TYPE, size: [220, 300], dirtyCalls: 0,
      widgets: [widget('seed', 123), widget('control_after_generate', 'fixed', 'combo'),
        widget('spatial_tiling', true, 'toggle'), widget('temporal_chunking', true, 'toggle'),
        widget('memory_budget_mb', 0), widget('enforce_memory_budget', true, 'toggle'),
        widget('memory_reserve_pct', 5, 'slider')],
      setDirtyCanvas() { this.dirtyCalls++; },
      onConfigure(info, extra) { this.configureReceiver = this; this.configureExtra = extra; return 'configured'; },
    };
    const toggle = result.widgets[5];
    toggle.callback = function(value, extra) {
      assert.equal(this, toggle); this.value = value; this.lastExtra = extra; return 'toggle-result';
    };
    const slider = result.widgets[6];
    slider.options = { serialize: true, min: 0, max: 100, step: 50, step2: 5, round: 5 };
    slider.callback = function(value) { this.value = Math.min(this.options.max, Math.max(this.options.min, Math.round(value / this.options.round) * this.options.round)); };
    extension.nodeCreated(result);
    return result;
  }
  // LiteGraph owns indexed restoration. This stub restores only saved slots,
  // then invokes the node hook, rather than letting the extension restore all inputs.
  function configure(node, values, extra = 'extra') {
    const info = { widgets_values: values };
    values.forEach((value, index) => { if (node.widgets[index]) node.widgets[index].value = value; });
    return node.onConfigure(info, extra);
  }
  function serialized(node) { return node.widgets.map(widget => widget.serializeValue()); }
  return { extension, data, node, configure, serialized };
}

test('native schema slider follows the LLM display pattern; no slots added or reordered', () => {
  const h = harness();
  assert.equal(h.data.input.optional.memory_reserve_pct[1].display, 'slider');
  assert.equal(h.data.input.optional.memory_reserve_pct[1].default, 5);
  assert.equal(h.data.input.optional.memory_reserve_pct[1].step, 5);
  assert.equal(h.data.input.optional.memory_reserve_pct[1].tooltip, 'Backend tooltip');
  const n = h.node();
  assert.deepEqual(n.widgets.map(w => w.name), ['seed', 'control_after_generate', 'spatial_tiling', 'temporal_chunking', 'memory_budget_mb', 'enforce_memory_budget', 'memory_reserve_pct']);
  assert.equal(n.widgets[5].label, 'Memory Planning');
  assert.notEqual(n.widgets[5].hidden, true);
  assert.equal(n.widgets[4].hidden, true);
  assert.equal(n.widgets[4].draw(), undefined);
  assert.deepEqual(Array.from(n.widgets[4].computeSize()), [0, -4]);
  assert.notEqual(n.widgets[6].hidden, true);
  assert.deepEqual(h.serialized(n), [123, 'fixed', true, true, 0, true, 5]);
});

for (const cap of [0, 768, 16384]) {
  for (const enabled of [true, false]) {
    test(`old indexed workflow preserves cap=${cap}, enforcement=${enabled} and defaults appended reserve`, () => {
      const h = harness(); const n = h.node();
      n.widgets[6].value = 80; // Also exercise reconfigure on an existing node.
      const saved = [42, 'randomize', false, true, cap, enabled];
      const snapshot = saved.slice();
      assert.equal(h.configure(n, saved), 'configured');
      assert.equal(n.configureReceiver, n);
      assert.equal(n.configureExtra, 'extra');
      h.extension.loadedGraphNode(n);
      assert.deepEqual(saved, snapshot);
      assert.deepEqual(h.serialized(n), [...saved, 5]);
      assert.equal(n.widgets[4].hidden, true);
      assert.equal(n.widgets[6].hidden === true, !enabled);
      assert.notEqual(n.widgets[5].hidden, true);
      assert.ok(n.widgets.every(w => w.options.serialize === true));
    });
  }
}

test('new workflows retain saved reserve, including 0 and 100, through repeated lifecycle hooks', () => {
  for (const reserve of [0, 35, 100]) {
    const h = harness(); const n = h.node(); const widgets = n.widgets.slice();
    const saved = [123, 'fixed', true, false, 4096, false, reserve];
    h.configure(n, saved);
    const callback = n.widgets[5].callback;
    h.extension.nodeCreated(n); h.extension.loadedGraphNode(n);
    assert.equal(n.widgets[5].callback, callback);
    assert.deepEqual(n.widgets, widgets);
    assert.deepEqual(h.serialized(n), saved);
    assert.equal(n.widgets[6].hidden, true);
  }
});

test('toggle hides/restores native slider without altering budget, reserve or tiling, preserving callback context', () => {
  const h = harness(); const n = h.node();
  h.configure(n, [123, 'fixed', true, true, 2048, true, 25]);
  const slider = n.widgets[6], toggle = n.widgets[5];
  const nativeDraw = slider.draw, nativeCompute = slider.computeSize;
  assert.equal(toggle.callback.call(toggle, false, 'argument'), 'toggle-result');
  assert.equal(toggle.lastExtra, 'argument');
  assert.equal(slider.hidden, true);
  assert.deepEqual(Array.from(slider.computeSize()), [0, -4]);
  assert.equal(slider.draw(), undefined);
  assert.deepEqual(h.serialized(n), [123, 'fixed', true, true, 2048, false, 25]);
  toggle.callback.call(toggle, true);
  assert.equal(slider.hidden, false);
  assert.equal(slider.draw, nativeDraw);
  assert.equal(slider.computeSize, nativeCompute);
  assert.equal(n.widgets[4].hidden, true);
  assert.ok(n.dirtyCalls > 0);
  slider.callback.call(slider, 40);
  assert.deepEqual(h.serialized(n), [123, 'fixed', true, true, 2048, true, 40]);
});

test('serialized boolean strings are not interpreted with JavaScript truthiness', () => {
  const h = harness(); const n = h.node();
  h.configure(n, [123, 'fixed', true, true, 1024, 'false', 15]);
  assert.equal(n.widgets[6].hidden, true);
  assert.equal(h.serialized(n)[5], 'false');
  n.widgets[5].callback.call(n.widgets[5], 'true');
  assert.equal(n.widgets[6].hidden, false);
});

test('unrelated node and missing optional widgets are safe; no frontend substitute backend inputs', () => {
  const h = harness();
  const other = { comfyClass: 'OtherNode', widgets: [] };
  h.extension.nodeCreated(other); h.extension.loadedGraphNode(other);
  assert.equal(other.onConfigure, undefined);
  const data = { name: 'OtherNode', input: { optional: { memory_reserve_pct: ['FLOAT', {}] } } };
  h.extension.beforeRegisterNodeDef(function() {}, data);
  assert.deepEqual(data.input.optional.memory_reserve_pct[1], {});
  const empty = { type: TYPE, widgets: [] };
  h.extension.nodeCreated(empty); h.extension.loadedGraphNode(empty);
  assert.equal(empty.widgets.length, 0);
});
