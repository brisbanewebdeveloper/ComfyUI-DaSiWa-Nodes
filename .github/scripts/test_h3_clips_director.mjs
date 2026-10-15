import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync}from 'node:fs';
const source=readFileSync(new URL('../../js/minimax_h3_director.js',import.meta.url),'utf8');
test('Director exposes only contextual Join clips and Advanced output policy hooks',()=>{
 assert.match(source,/output_mode === "segment" \? types\.has\("DaSiWaH3ContinuitySegmentPrepare"\)/);
 assert.match(source,/button\("Join clips…"/);assert.match(source,/Output /);assert.match(source,/New segment only/);
 assert.match(source,/closeClipAssembly\?\.\(\)/);assert.match(source,/dataWidget\.serializeValue\s*=/);assert.match(source,/validateSegmentGraph/);
});
