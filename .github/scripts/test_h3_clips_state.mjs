import test from 'node:test';
import assert from 'node:assert/strict';
test('video snapshots preserve references and project safe trimmed handles', async () => {
 const {snapshotTimelineVideos, assemblyPayload} = await load();
 const items=[{id:'v',type:'video',value:'folder/a.mp4',enabled:true,media_mode:'video_audio',trim_start:1,trim_end:3,slot:2}, {type:'video',value:'audio.mp4',media_mode:'audio'},{type:'image',value:'a.png'},{type:'video',value:'off.mp4',enabled:false}];
 const before=structuredClone(items), rows=snapshotTimelineVideos(items);
 assert.equal(rows.length,1); rows[0].trim_start=2; assert.deepEqual(items,before);
 assert.deepEqual(assemblyPayload({tab:'files',rows}).rows,[{kind:'media',filename:'folder/a.mp4',trim_start:2,trim_end:3}]);
 for(const filename of ['/etc/a.mp4','../a.mp4','https://host/a.mp4','C:\\a.mp4']) assert.throws(()=>assemblyPayload({tab:'files',rows:[{kind:'media',filename}]}), /handle/);
 assert.throws(()=>assemblyPayload({rows:[]}), /Select/);
});
test('chain follows parents, rejects branches and collapses full prefixes', async()=>{
 const {ancestorChain,selectContinuityRows}=await load();
 const clips=[{clip_id:'a',status:'ready',parent_id:'',seconds:5,output_path:'a.mp4'},{clip_id:'b',parent_id:'a',status:'ready',seconds:10,output_path:'b.mp4'},{clip_id:'c',parent_id:'b',status:'ready',seconds:15,export:{kind:'segment',frames:120}},{clip_id:'alt',parent_id:'a',status:'ready',output_path:'alt.mp4'}];
 assert.deepEqual(ancestorChain(clips,'c').map(c=>c.clip_id),['a','b','c']);
 assert.deepEqual(selectContinuityRows(clips,['a','b','c'],'s').map(c=>c.clip_id),['b','c']);
 assert.throws(()=>selectContinuityRows(clips,['b','alt'],'s'),/Alternative takes/);
 assert.throws(()=>ancestorChain([{clip_id:'x',parent_id:'x'}],'x'),/cycle/);
});
test('output mode defaults full for every migration and rejects explicit invalid modes',async()=>{
 const {normalizeContinuity}=await import('../../js/minimax_h3_continuity.js');
 for(const version of [1,2,3]) assert.equal(normalizeContinuity({version}).output_mode,'full');
 assert.equal(normalizeContinuity({version:3,output_mode:'segment'}).output_mode,'segment');
 assert.throws(()=>normalizeContinuity({output_mode:'wrong'}),/output mode/);
});
test('segment topology rejects mere node presence and accepts connected local render chain',async()=>{
 const {validateSegmentGraph}=await import('../../js/minimax_h3_continuity.js');
 const graph={d:{class_type:'MiniMaxH3Director'},g:{class_type:'MiniMaxH3DirectorGuide',inputs:{guide:['d',0]}},p:{class_type:'DaSiWaH3ContinuitySegmentPrepare',inputs:{context:['g',2]}},dec:{class_type:'DaSiWaH3ContinuitySegmentDecode',inputs:{latent:['p',0],context:['p',1]}},exp:{class_type:'EnhancedVideoCombine',inputs:{images:['dec',0]}},pub:{class_type:'DaSiWaH3ContinuityPublish',inputs:{ticket:['p',2],filename:['exp',0]}}};
 assert.doesNotThrow(()=>validateSegmentGraph(graph,'d'));
 graph.up={class_type:'DaSiWaH3TiledUpscale',inputs:{latent:['p',0],continuity_context:['p',1]}};graph.dec.inputs.latent=['up',0];assert.doesNotThrow(()=>validateSegmentGraph(graph,'d'));
 graph.up.inputs.latent=['g',0];assert.throws(()=>validateSegmentGraph(graph,'d'),/Segment Prepare/);graph.up.inputs.latent=['p',0];
 delete graph.dec.inputs.context;assert.throws(()=>validateSegmentGraph(graph,'d'),/Segment Prepare/);
});
test('assembly boundary rejects invalid rows, oversize requests, unsupported policies and invalid names',async()=>{
 const {assemblyPayload,normalizeClipAssembly}=await load();
 for(const raw of [{version:2},{tab:'wrong'},{rows:{}},{repair_transitions:'yes'},{audio_policy:'generate'}])assert.throws(()=>normalizeClipAssembly(raw));
 for(const rows of [[null],Array.from({length:101},()=>({kind:'media',filename:'a.mp4'})),[{kind:'media',filename:'a.mp4',trim_start:true}],[{kind:'media',filename:'a.mp4',trim_end:Infinity}]]) assert.throws(()=>assemblyPayload({tab:'files',rows}));
 assert.throws(()=>assemblyPayload({tab:'files',filename_prefix:'../unsafe',rows:[{kind:'media',filename:'a.mp4'}]}),/name|prefix/);
});
const load = () => import('../../js/minimax_h3_clips_state.js');
test('assembly defaults are independent and inert', async () => {
 const {normalizeClipAssembly} = await load();
 const a=normalizeClipAssembly(), b=normalizeClipAssembly();
 assert.equal(a.version,1); assert.equal(a.tab,'continuity'); assert.equal(a.repair_transitions,false);
 a.rows.push({clip_id:'one'}); assert.deepEqual(b.rows,[]);
});
