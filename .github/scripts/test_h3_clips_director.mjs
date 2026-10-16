import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync}from 'node:fs';
const source=readFileSync(new URL('../../js/minimax_h3_director.js',import.meta.url),'utf8');
const {normalizeContinuity,sourceId,validateSegmentGraph}=await import('../../js/minimax_h3_continuity.js');
// Execute only trusted local production source, not a copy of its activation condition.
function serializedTimeline(continuity,modeValue='T2VA',original) {
 const dataWidget={value:'saved timeline',serializeValue:original};
 const node={id:'director',graph:{_nodes:[]}};
 const continuityState=()=>normalizeContinuity(continuity);
 const install=source.slice(source.indexOf('  const checkSegmentTopology ='),source.indexOf('  function openRefModExport()'));
 new Function('node','dataWidget','continuityState','isContinuing','mode','validateSegmentGraph',install)(node,dataWidget,continuityState,()=>!!sourceId(continuityState()),()=>modeValue,validateSegmentGraph);
 return dataWidget;
}
test('actual serialization wrapper skips segment topology for inactive new takes',()=>{
 for(const [capture,mode] of [[false,'T2VA'],[false,'Image Inpaint'],[true,'Image Inpaint']]) {
  const widget=serializedTimeline({version:3,output_mode:'segment',capture},mode,function(...args){assert.equal(this,widget);return args;});
  assert.deepEqual(widget.serializeValue('queue',7),['queue',7]);
 }
 assert.equal(serializedTimeline({version:3,output_mode:'segment',capture:false}).serializeValue(),'saved timeline');
});
test('actual serialization wrapper still validates active capture and continuation',()=>{
 for(const settings of [{capture:true},{capture:false,source_id:'parent'},{capture:false,source_kind:'video',source_video_id:'upload'}]) {
  assert.throws(()=>serializedTimeline({version:3,output_mode:'segment',...settings}).serializeValue(),/New segment only requires/);
 }
 assert.equal(serializedTimeline({version:3,output_mode:'full',capture:true}).serializeValue(),'saved timeline');
});
test('Director exposes only contextual Join clips and Advanced output policy hooks',()=>{
 assert.match(source,/output_mode === "segment" \? segmentNodes : segmentNodes \|\| types\.has\("DaSiWaH3ContinuityAppend"\)/);
 assert.match(source,/button\("Join clips…"/);assert.match(source,/Output /);assert.match(source,/New segment only/);
 assert.match(source,/closeClipAssembly\?\.\(\)/);assert.match(source,/dataWidget\.serializeValue\s*=/);assert.match(source,/validateSegmentGraph/);
});
