const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const file = path.join(__dirname, '..', 'js', 'enhanced_video_combine_preview.js');
const source = fs.readFileSync(file, 'utf8').replace(/^import .*;\s*$/gm, '');
let extension;
const context = {
    app: { registerExtension: (value) => { extension = value; } },
    api: { apiURL: (value) => value },
    URLSearchParams,
    Set,
};
vm.runInNewContext(source, context);
assert.ok(extension);
assert.equal(context.shouldUseTranscodedPreview({codec: 'H.264', container: 'MP4', bit_depth: 8}), false);
for (const codec of ['H.264', 'H.265', 'AV1', 'VP9']) {
    for (const container of ['MP4', 'MKV', 'WebM']) {
        for (const bit_depth of [8, 10]) {
            const video = {filename: 'clip.mp4', subfolder: 'folder with spaces', type: 'output', codec, container, bit_depth};
            const expected = !(codec === 'H.264' && container === 'MP4' && bit_depth === 8);
            assert.equal(context.shouldUseTranscodedPreview(video), expected);
            assert.match(context.videoUrl(video), /^\/view\?/);
            assert.match(context.transcodedVideoUrl(video), /^\/dasiwa\/enhanced-video-preview\?/);
            assert.equal(new URLSearchParams(context.videoUrl(video).split('?')[1]).get('subfolder'), video.subfolder);
        }
    }
}
assert.equal(context.shouldUseTranscodedPreview({codec:'H.264', container:'MP4'}), false);
assert.equal(context.shouldUseTranscodedPreview({codec:'unknown', container:'unknown'}), true);
for (const name of ['DaSiWa_EnhancedVideoCombine']) {
    class Node {}
    extension.beforeRegisterNodeDef(Node, { name, input: { required: {} } });
    assert.equal(typeof Node.prototype.onNodeCreated, 'function', `${name} missing preview`);
    assert.equal(typeof Node.prototype.onExecuted, 'function', `${name} missing video update`);
}
console.log('Preview registered on regular combine node');
