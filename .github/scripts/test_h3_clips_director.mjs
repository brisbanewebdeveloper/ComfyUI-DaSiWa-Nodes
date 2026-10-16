import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync}from 'node:fs';
const source=readFileSync(new URL('../../js/minimax_h3_director.js',import.meta.url),'utf8');
// Active/inactive execution validation is covered in test_h3_segment_graph.mjs.
test('Director exposes only contextual Join clips and Advanced output policy hooks',()=>{
 assert.match(source,/output_mode === "segment" \? segmentNodes : segmentNodes \|\| types\.has\("DaSiWaH3ContinuityAppend"\)/);
 assert.match(source,/button\("Join clips…"/);assert.match(source,/Output /);assert.match(source,/New segment only/);
 assert.match(source,/closeClipAssembly\?\.\(\)/);assert.match(source,/installSegmentGraphValidation\(app\)/);assert.doesNotMatch(source,/dataWidget\.serializeValue\s*=|checkSegmentTopology/);
 assert.match(source,/Exports only the new section, without repeating the source/);
 assert.match(source,/Exports the source and the new section together/);
});
