import test from 'node:test';
import assert from 'node:assert/strict';
const load=()=>import('../../js/minimax_h3_clips.js');
test('queue preflights then polls only its prompt and own output',async()=>{
 const {queueClipAssembly}=await load(); const calls=[];
 const api={clientId:'client',fetchApi:async(url,options)=>{calls.push([url,options]); return {ok:true,json:async()=>url.includes('preflight')?{ok:true,seconds:2,mode:'copy'}:url==='/prompt'?{prompt_id:'own'}:{other:{outputs:{x:{h3_clip_assembly:[{filename:'wrong'}]}}},own:{outputs:{'h3-clip-assemble':{h3_clip_assembly:[{filename:'joined.mp4',seconds:2,mode:'copy'}]}}}}};}};
 const result=await queueClipAssembly({api,state,pollMs:0}); assert.equal(result.filename,'joined.mp4');
 assert.deepEqual(calls.map(c=>c[0]),['/df_h3_continuity/assembly/preflight','/prompt','/history/own']);
 assert.equal(Object.keys(JSON.parse(calls[1][1].body).prompt).length,1);
});
test('queue fails closed on preflight, HTTP errors, interruption and absent result',async()=>{
 const {queueClipAssembly}=await load();
 for(const scenario of ['preflight','http','interrupted','missing']) {
 const api={fetchApi:async url=>({ok:scenario!=='http',status:400,json:async()=>url.includes('preflight')?{ok:scenario!=='preflight',issues:['bad canvas']}:url==='/prompt'?{prompt_id:'p'}:{p:{status:{completed:true,messages:scenario==='interrupted'?[['execution_interrupted',{}]]:[]}}}})};
 await assert.rejects(queueClipAssembly({api,state,pollMs:0}),/bad canvas|400|interrupted|without/);
 }
 const controller=new AbortController();controller.abort();
 await assert.rejects(queueClipAssembly({api:{fetchApi:()=>assert.fail('aborted request')},state,signal:controller.signal}),{name:'AbortError'});
});
const state={tab:'files',rows:[{kind:'media',filename:'a.mp4'}]};
test('assembly prompt is exactly one model-free node with collision-safe identity',async()=>{
 const {buildClipAssemblyPrompt}=await load();
 const output={'h3-clip-assemble':{class_type:'Sampler'}};
 const before=structuredClone(output), {prompt,assembleId}=buildClipAssemblyPrompt(output,state);
 assert.deepEqual(output,before); assert.notEqual(assembleId,'h3-clip-assemble'); assert.equal(Object.keys(prompt).length,1);
 assert.equal(prompt[assembleId].class_type,'DaSiWaH3ClipAssemble'); assert.equal(JSON.parse(prompt[assembleId].inputs.assembly_data).rows[0].filename,'a.mp4');
});
