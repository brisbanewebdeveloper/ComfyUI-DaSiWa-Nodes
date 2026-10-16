import test from 'node:test';
import assert from 'node:assert/strict';
const load=()=>import('../../js/minimax_h3_clips_state.js');
const graph=()=>({
  model:{class_type:'UNETLoader',inputs:{unet_name:'h3.safetensors'}},
  clip:{class_type:'CLIPLoader',inputs:{clip_name:'qwen.safetensors'}},
  vae:{class_type:'VAELoader',inputs:{vae_name:'video.safetensors'}},
  avae:{class_type:'VAELoader',inputs:{vae_name:'audio.safetensors'}},
  d:{class_type:'MiniMaxH3Director',inputs:{timeline_data:'{}'}},
  g:{class_type:'MiniMaxH3DirectorGuide',inputs:{clip:['clip',0],vae:['vae',0],guide:['d',0],audio_vae:['avae',0]}},
  guider:{class_type:'BasicGuider',inputs:{model:['model',0],conditioning:['g',0]}},
  noise:{class_type:'RandomNoise',inputs:{noise_seed:7}},
  sampler:{class_type:'SamplerCustomAdvanced',inputs:{noise:['noise',0],guider:['guider',0],latent_image:['g',1]}},
  decode:{class_type:'VAEDecode',inputs:{samples:['sampler',0],vae:['vae',0]}},
  save:{class_type:'SaveVideo',inputs:{video:['decode',0]}},
  unrelated:{class_type:'SaveImage',inputs:{images:['other',0]}},
  prepare:{class_type:'DaSiWaH3ContinuitySegmentPrepare',inputs:{sampled:['sampler',0],context:['g',2]}},
});
test('transition prompt swaps only the Director Guide into the sampler closure',async()=>{
 const {buildTransitionPrompt}=await load(); const output=graph(), before=structuredClone(output);
 const spec={version:1,rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'}]};
 const {prompt,renderId}=buildTransitionPrompt(output,'d',spec,'r1');
 assert.deepEqual(output,before);
 assert.deepEqual(Object.keys(prompt).sort(),['avae','clip','d','g','guider','model','noise','sampler','vae',renderId].sort());
 assert.equal(prompt.g.class_type,'DaSiWaH3TransitionGuide');
 assert.deepEqual(JSON.parse(prompt.g.inputs.transition_data),spec);
 assert.deepEqual(prompt.g.inputs.guide,['d',0]);
 assert.deepEqual(prompt[renderId],{class_type:'DaSiWaH3TransitionRender',inputs:{latent:['sampler',0],vae:['vae',0],audio_vae:['avae',0],transition:['g',2]}});
});
test('transition prompt fails clearly on unsupported Director wiring',async()=>{
 const {buildTransitionPrompt}=await load();
 const noGuide=graph();delete noGuide.g;
 assert.throws(()=>buildTransitionPrompt(noGuide,'d',{},'r'),/Director Guide/);
 const noAudio=graph();delete noAudio.g.inputs.audio_vae;
 assert.throws(()=>buildTransitionPrompt(noAudio,'d',{},'r'),/audio VAE/);
 const twoSamplers=graph();twoSamplers.second={class_type:'KSampler',inputs:{latent_image:['g',1]}};
 assert.throws(()=>buildTransitionPrompt(twoSamplers,'d',{},'r'),/exactly one sampler/);
 const noSampler=graph();delete noSampler.sampler;delete noSampler.prepare;delete noSampler.decode;
 assert.throws(()=>buildTransitionPrompt(noSampler,'d',{},'r'),/exactly one sampler/);
});
test('transition requests cover each boundary and bridges interleave in order',async()=>{
 const {transitionRequests,interleaveBridges,normalizeClipAssembly}=await load();
 const state=normalizeClipAssembly({tab:'files',repair_transitions:true,context_frames:39,transition_frames:46,transition_prompt:' turn ',
   rows:[{kind:'media',filename:'a.mp4',trim_end:4},{kind:'media',filename:'skip.mp4',selected:false},{kind:'media',filename:'b.mp4'},{kind:'media',filename:'c.mp4',trim_start:1}]});
 const requests=transitionRequests(state);
 assert.equal(requests.length,2);
 assert.deepEqual(requests[0],{version:1,rows:[{kind:'media',filename:'a.mp4',trim_start:0,trim_end:4},{kind:'media',filename:'b.mp4',trim_start:0,trim_end:null}],context_frames:39,transition_frames:46,prompt:'turn',filename_prefix:'H3_Transition'});
 assert.equal(requests[1].rows[1].filename,'c.mp4');
 assert.deepEqual(interleaveBridges(state,['h3_transitions/t1.mp4','h3_transitions/t2.mp4']).rows.map(r=>r.filename),['a.mp4','h3_transitions/t1.mp4','b.mp4','h3_transitions/t2.mp4','c.mp4']);
});
test('transition settings validate the H3 grid and keep the final join model-free',async()=>{
 const {normalizeClipAssembly,assemblyPayload}=await load();
 for(const raw of [{context_frames:6},{transition_frames:17},{transition_prompt:'x'.repeat(4001)},{transition_prompt:3}]) assert.throws(()=>normalizeClipAssembly(raw));
 const state=normalizeClipAssembly({});assert.equal(state.context_frames,22);assert.equal(state.transition_frames,29);assert.equal(state.transition_prompt,'');
 assert.equal(assemblyPayload({tab:'files',repair_transitions:true,rows:[{kind:'media',filename:'a.mp4'}]}).repair_transitions,false);
});
test('dialog renders transitions before one model-free join and inserts every bridge',async()=>{
 const {queueTransitionJoin}=await import('../../js/minimax_h3_clips.js');
 const posts=[];let n=0;
 const api={clientId:'c',fetchApi:async(url,options)=>{
   if(url==='/prompt'){const body=JSON.parse(options.body);posts.push(body);return {ok:true,json:async()=>({prompt_id:'p'+posts.length})};}
   if(url.includes('preflight'))return {ok:true,json:async()=>({ok:true,seconds:9,frames:216,mode:'reencode'})};
   const id=url.split('/').at(-1),body=posts[Number(id.slice(1))-1];
   const out=Object.entries(body.prompt).find(([,node])=>['DaSiWaH3TransitionRender','DaSiWaH3ClipAssemble'].includes(node.class_type));
   const value=out[1].class_type==='DaSiWaH3TransitionRender'?{h3_transition:[{filename:`h3_transitions/t${++n}.mp4`,frames:29,seconds:29/24}]}:{h3_clip_assembly:[{filename:'joined.mp4',seconds:9,mode:'reencode'}]};
   return {ok:true,json:async()=>({[id]:{outputs:{[out[0]]:value}}})};}};
 const state={tab:'files',repair_transitions:true,rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'},{kind:'media',filename:'c.mp4'}]};
 const status=[];
 const result=await queueTransitionJoin({api,state,graphPrompt:async()=>({output:graph(),workflow:{}}),directorId:'d',pollMs:0,onStatus:s=>status.push(s),confirmPlan:async()=>true});
 assert.equal(result.filename,'joined.mp4');
 assert.deepEqual(posts.map(p=>Object.values(p.prompt).some(node=>node.class_type==='DaSiWaH3TransitionRender')),[true,true,false]);
 assert.deepEqual(JSON.parse(Object.values(posts[2].prompt)[0].inputs.assembly_data).rows.map(r=>r.filename),['a.mp4','h3_transitions/t1.mp4','b.mp4','h3_transitions/t2.mp4','c.mp4']);
 assert.ok(status.some(s=>/transition 1 of 2/i.test(s))&&status.some(s=>/transition 2 of 2/i.test(s)));
});
test('transition join rejects clips shorter than the context before any render',async()=>{
 const {queueTransitionJoin}=await import('../../js/minimax_h3_clips.js');
 const calls=[];const api={fetchApi:async url=>{calls.push(url);return {ok:true,json:async()=>({ok:true,seconds:2,frames:48,mode:'reencode',rows:[{frames:10},{frames:38}]})};}};
 await assert.rejects(queueTransitionJoin({api,state:{tab:'files',repair_transitions:true,rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'}]},graphPrompt:async()=>({output:graph()}),directorId:'d',confirmPlan:async()=>true}),/shorter than the transition context/);
 assert.ok(!calls.includes('/prompt'));
});
test('transition join needs two selected clips and stops before rendering when declined',async()=>{
 const {queueTransitionJoin}=await import('../../js/minimax_h3_clips.js');
 const calls=[];const api={fetchApi:async url=>{calls.push(url);return {ok:true,json:async()=>({ok:true,seconds:2,frames:48,mode:'reencode'})};}};
 await assert.rejects(queueTransitionJoin({api,state:{tab:'files',repair_transitions:true,rows:[{kind:'media',filename:'a.mp4'}]},graphPrompt:async()=>({output:graph()}),directorId:'d'}),/two clips/);
 await assert.rejects(queueTransitionJoin({api,state:{tab:'files',repair_transitions:true,rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'}]},graphPrompt:async()=>({output:graph()}),directorId:'d',confirmPlan:async()=>false}),/cancelled/);
 assert.ok(!calls.includes('/prompt'));
});
test('copy video is an opt-in boolean sent with the join request',async()=>{
 const {normalizeClipAssembly,assemblyPayload}=await load();
 assert.equal(normalizeClipAssembly({}).copy_video,false);
 assert.throws(()=>normalizeClipAssembly({copy_video:'yes'}));
 const rows=[{kind:'media',filename:'a.mp4'}];
 assert.equal(assemblyPayload({tab:'files',rows}).copy_video,false);
 assert.equal(assemblyPayload({tab:'files',rows,copy_video:true}).copy_video,true);
});
