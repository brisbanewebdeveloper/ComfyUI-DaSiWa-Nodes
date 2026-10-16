import test from 'node:test';
import assert from 'node:assert/strict';
class Element extends EventTarget {
 constructor(tag){super();this.tagName=tag.toUpperCase();this.children=[];this.isConnected=true;this.style={};this.value='';this.disabled=false;}
 append(...children){this.children.push(...children);} replaceChildren(...children){this.children=children;}
 setAttribute(key,value){this[key]=value;} removeAttribute(key){delete this[key];}
 remove(){this.isConnected=false;} focus(){document.activeElement=this;}
 showModal(){this.open=true;} close(){this.open=false;} pause(){} load(){}
 querySelectorAll(){return this.children.flatMap(c=>typeof c==='object'?[c,...c.querySelectorAll()]:[]);}
}
function setup(){globalThis.document={activeElement:new Element('button'),createElement:t=>new Element(t),head:new Element('head'),body:new Element('body')};return document;}
const all=e=>[e,...e.children.filter(c=>typeof c==='object').flatMap(all)];
const click=async e=>{e.dispatchEvent(new Event('click'));await new Promise(r=>setTimeout(r,0));};
test('clip fields have visible labels and H3 transitions need the Director graph',async()=>{
 setup();const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async()=>({ok:true,json:async()=>({seconds:3,sessions:[]})})},timeline:{clip_assembly:{tab:'files',rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'}]}},onChange:()=>{},viewUrl:p=>p});
 const dialog=document.body.children.at(-1);
 assert.equal(all(dialog).filter(e=>e.textContent==='Start (s)').length,1);
 assert.equal(all(dialog).filter(e=>e.textContent==='End (s)').length,1);
 const toggle=all(dialog).find(e=>e.type==='checkbox'&&e.title?.includes('bridge'));
 assert.equal(toggle.disabled,true);
 assert.ok(all(dialog).some(e=>e.textContent?.includes('wired to a Director Guide and sampler')));close();
});
test('enabling H3 transitions saves settings and relabels the join action',async()=>{
 setup();let saved;const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async()=>({ok:true,json:async()=>({seconds:3,sessions:[]})})},timeline:{clip_assembly:{tab:'files',rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'}]}},onChange:s=>{saved=s;},viewUrl:p=>p,graphPrompt:async()=>({output:{}}),directorId:'d'});
 const dialog=document.body.children.at(-1);
 let toggle=all(dialog).find(e=>e.type==='checkbox'&&e.title?.includes('bridge'));
 assert.equal(toggle.disabled,false);toggle.checked=true;toggle.dispatchEvent(new Event('change'));
 assert.equal(saved.repair_transitions,true);
 assert.ok(all(dialog).some(e=>e.textContent==='Join with H3 transitions'));
 const selects=all(dialog).filter(e=>e.tagName==='SELECT');assert.equal(selects.length,2);assert.ok(selects.every(e=>!e.disabled));
 selects[0].value='46';selects[0].dispatchEvent(new Event('change'));assert.equal(saved.transition_frames,46);
 selects[1].value='39';selects[1].dispatchEvent(new Event('change'));assert.equal(saved.context_frames,39);
 const text=all(dialog).find(e=>e['aria-label']==='Transition description');text.value='She opens the door.';text.dispatchEvent(new Event('change'));
 assert.equal(saved.transition_prompt,'She opens the door.');close();
});
test('Remove deletes only the chosen montage row, not files or reference media',async()=>{
 setup();const calls=[],timeline={items:[{type:'video',value:'a.mp4'}],continuity:{source_id:'p'},clip_assembly:{tab:'files',rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'a.mp4'}]}};
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async url=>{calls.push(url);return {ok:true,json:async()=>({seconds:3,sessions:[]})};}},timeline,onChange:s=>{timeline.clip_assembly=s;},viewUrl:p=>p});
 const dialog=document.body.children.at(-1);await click(all(dialog).find(e=>e.textContent==='Remove'));
 assert.equal(timeline.clip_assembly.rows.length,1);assert.deepEqual(timeline.items,[{type:'video',value:'a.mp4'}]);assert.equal(timeline.continuity.source_id,'p');
 assert.ok(calls.every(url=>!url.includes('delete')&&!url.includes('/prompt')));close();
});
test('source duration and selected duration are detected without changing blank end; failures are explicit',async()=>{
 setup();const timeline={clip_assembly:{tab:'files',rows:[{kind:'media',filename:'a.mkv',trim_start:1,trim_end:null},{kind:'media',filename:'bad.mp4'}]}};
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async url=>({ok:!url.includes('bad.mp4'),status:url.includes('bad.mp4')?400:200,json:async()=>url.includes('bad.mp4')?{error:'Cannot read metadata'}:{seconds:3,sessions:[]}})},timeline,onChange:s=>{timeline.clip_assembly=s;},viewUrl:p=>p});
 await new Promise(r=>setTimeout(r,0));const dialog=document.body.children.at(-1);
 assert.ok(all(dialog).some(e=>e.textContent?.includes('Source duration: 3.000 s')&&e.textContent?.includes('Selected: 2.000 s')));
 assert.ok(all(dialog).some(e=>e.textContent?.includes('Duration unavailable')));
 const ends=all(dialog).filter(e=>e.type==='number'&&e['aria-label']?.endsWith('End (s)'));
 assert.equal(ends[0].value,'3.000');assert.equal(ends[1].value,'');
 assert.equal(timeline.clip_assembly.rows[0].trim_end,null);close();
});
test('checkpoint rows explain takes and the chain selector',async()=>{
 setup();const clips=[{clip_id:'root1',parent_id:'',status:'ready',frames:124,seconds:124/24,export_available:true,export:{kind:'full',filename:'a.mp4'}},
   {clip_id:'next2',parent_id:'root1',status:'ready',frames:243,seconds:243/24,segment_frames:119,export_available:true,export:{kind:'segment',filename:'b.mp4'}}];
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{apiURL:p=>p,fetchApi:async url=>({ok:true,json:async()=>url.endsWith('sessions')?{sessions:[]}:{clips}})},timeline:{continuity:{session:'s'}},onChange:()=>{},viewUrl:p=>p});
 await new Promise(r=>setTimeout(r,0));const dialog=document.body.children.at(-1);
 assert.ok(all(dialog).some(e=>e.textContent?.includes('next2 · adds 4.96 s · ends at 10.13 s · continues root1 · segment export')));
 assert.ok(all(dialog).some(e=>e.textContent?.includes('root1 · adds 5.17 s · ends at 5.17 s · first take · full-timeline export')));
 const chain=all(dialog).filter(e=>e.textContent==='Select chain to here');
 assert.equal(chain.length,2);assert.match(chain[0].title,/every earlier take it continues/);
 assert.ok(!all(dialog).some(e=>e.textContent==='Chain to here'));close();
});
test('late duration metadata preserves the focused trim edit and removed rows stay removed',async()=>{
 setup();let resolve;const response=new Promise(r=>{resolve=r;});let saved;
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async url=>url.includes('media-info')?response:{ok:true,json:async()=>({sessions:[]})}},timeline:{clip_assembly:{tab:'files',rows:[{kind:'media',filename:'a.mp4'}]}},onChange:s=>{saved=s;},viewUrl:p=>p});
 const dialog=document.body.children.at(-1),input=all(dialog).find(e=>e.type==='number');input.value='1.25';input.focus();
 resolve({ok:true,json:async()=>({seconds:3})});await new Promise(r=>setTimeout(r,0));
 assert.equal(document.activeElement,input);assert.ok(all(dialog).includes(input));assert.equal(input.value,'1.25');
 await click(all(dialog).find(e=>e.textContent==='Remove'));assert.equal(saved.rows.length,0);close();
});
test('a later upload error keeps earlier uploaded selections and reports the error',async()=>{
 setup();let saved,count=0;const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async()=>({ok:true,json:async()=>({sessions:[]})})},timeline:{clip_assembly:{tab:'files'}},onChange:s=>{saved=s;},upload:async()=>{if(++count===2)throw new Error('upload failed');return 'first.mp4';}});
 const dialog=document.body.children.at(-1),input=all(dialog).find(e=>e.type==='file');input.files=[{name:'first.mp4'},{name:'second.mp4'}];input.dispatchEvent(new Event('change'));
 await new Promise(r=>setTimeout(r,0));assert.equal(saved.rows[0].filename,'first.mp4');assert.ok(all(dialog).some(e=>e.textContent==='upload failed'));close();
});
test('reopening restores explicit foreign montage session without changing pinned source',async()=>{
 setup();const calls=[],timeline={continuity:{session:'pinned',source_id:'parent'},clip_assembly:{rows:[{kind:'checkpoint',session:'chosen',clip_id:'c'}]}};
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');const close=openClipAssembly({api:{fetchApi:async url=>{calls.push(url);return {ok:true,json:async()=>({sessions:[],clips:[]})};}},timeline,onChange:()=>{}});
 await new Promise(r=>setTimeout(r,0));assert.ok(calls.includes('/df_h3_continuity/session/chosen'));assert.equal(timeline.continuity.session,'pinned');close();
});
test('continuity list shows added duration, disables unavailable exports and previews existing tail images',async()=>{
 setup();const timeline={continuity:{session:'s'}};
 const clips=[{clip_id:'a',parent_id:'',status:'ready',frames:120,seconds:5,thumbnails:['tail.jpg']},{clip_id:'b',parent_id:'a',status:'ready',frames:240,seconds:10,extension_frames:120,thumbnails:['tail.jpg']},{clip_id:'bad',status:'ready',export_available:false}];
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{apiURL:p=>p,fetchApi:async url=>({ok:true,json:async()=>url.endsWith('sessions')?{sessions:[]}:{clips}})},timeline,onChange:s=>{timeline.clip_assembly=s;},viewUrl:p=>p});
 await new Promise(r=>setTimeout(r,0));const dialog=document.body.children.at(-1);
 assert.ok(all(dialog).some(e=>e.textContent?.startsWith('b · adds 5.00 s')));
 assert.equal(all(dialog).find(e=>e['aria-label']==='Select bad').disabled,true);
 const bRow=all(dialog).find(e=>e.children.some?.(c=>c.textContent?.startsWith('b ·')));
 await click(bRow.children.find(e=>e.textContent==='Preview'));
 assert.ok(all(dialog).some(e=>e.tagName==='IMG'&&e.src==='/df_h3_continuity/tail/s/b/0'));
 await click(bRow.children.find(e=>e.textContent==='Select chain to here'));
 assert.deepEqual(timeline.clip_assembly.rows.map(r=>r.clip_id),['a','b']);close();
});
test('checkpoint preview preserves the verified subfolder and temp type',async()=>{
 setup();const clips=[{clip_id:'a',parent_id:'',status:'ready',frames:22,seconds:22/24,export_available:true,export:{kind:'full',filename:'same.mp4',subfolder:'nested',type:'temp'}}];
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{apiURL:p=>p,fetchApi:async url=>({ok:true,json:async()=>url.endsWith('sessions')?{sessions:[]}:{clips}})},timeline:{continuity:{session:'s'}},onChange:()=>{},viewUrl:p=>p});
 await new Promise(r=>setTimeout(r,0));const dialog=document.body.children.at(-1);
 await click(all(dialog).find(e=>e.textContent==='Preview'));
 const video=all(dialog).find(e=>e.tagName==='VIDEO');const url=new URL(video.src,'http://example.test');
 assert.equal(url.pathname,'/view');assert.equal(url.searchParams.get('filename'),'same.mp4');assert.equal(url.searchParams.get('subfolder'),'nested');assert.equal(url.searchParams.get('type'),'temp');close();
});
test('copying timeline videos keeps separately uploaded files and can refresh the copied snapshot',async()=>{
 setup();let saved;const timeline={items:[{id:'v',type:'video',value:'reference.mp4'}],clip_assembly:{tab:'files',rows:[{kind:'media',filename:'uploaded.mp4'}]}};
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async()=>({ok:true,json:async()=>({sessions:[]})})},timeline,onChange:s=>{saved=s;},viewUrl:p=>p,upload:async()=>''});
 const dialog=document.body.children.at(-1);
 await click(all(dialog).find(e=>e.textContent==='Use enabled timeline videos'));
 assert.deepEqual(saved.rows.map(r=>r.filename),['uploaded.mp4','reference.mp4']);
 await click(all(dialog).find(e=>e.textContent==='Use enabled timeline videos'));
 assert.equal(saved.rows.length,2);close();
});
test('dialog uploads, reorders and deselects independent media and does not queue until confirmation',async()=>{
 setup();const calls=[], timeline={clip_assembly:{tab:'files',rows:[{kind:'media',filename:'a.mp4'},{kind:'media',filename:'b.mp4'}]}};
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const cleanup=openClipAssembly({api:{fetchApi:async(url)=>{calls.push(url);return {ok:true,json:async()=>url.includes('preflight')?{ok:true,seconds:4,frames:96,mode:'reencode',width:64,height:64,fps:24,notes:['Video and audio will be re-encoded.']}:{sessions:[]}};}},timeline,onChange:s=>{timeline.clip_assembly=s;},viewUrl:p=>p,upload:async()=> 'uploaded.mp4'});
 const dialog=document.body.children.at(-1);
 await click(all(dialog).find(e=>e['aria-label']==='Move b.mp4 up'));
 assert.deepEqual(timeline.clip_assembly.rows.map(r=>r.filename),['b.mp4','a.mp4']);
 const check=all(dialog).find(e=>e['aria-label']==='Select a.mp4');check.checked=false;check.dispatchEvent(new Event('change'));
 assert.equal(timeline.clip_assembly.rows.find(r=>r.filename==='a.mp4').selected,false);
 await click(all(dialog).find(e=>e.textContent==='Join only — no AI'));
 assert.ok(all(dialog).some(e=>e.textContent?.includes('64×64 · 24 fps')));
 assert.ok(all(dialog).find(e=>e.textContent==='Re-encode and join'));assert.ok(!calls.includes('/prompt'));
 cleanup();
});
test('native HTMLCollection has no Array.at method',async()=>{
 setup();const create=document.createElement;document.createElement=tag=>{const e=create(tag);e.children.at=undefined;return e;};
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const close=openClipAssembly({api:{fetchApi:async()=>({ok:true,json:async()=>({sessions:[]})})},timeline:{},onChange:()=>{}});close();
});
test('dialog opens without queue/model calls, copies videos separately, closes and restores focus',async()=>{
 setup();const prior=document.activeElement, calls=[], timeline={items:[{id:'v',type:'video',value:'a.mp4',trim_start:1,trim_end:3}],continuity:{session:'s',source_id:'p'},builder_state:{prompt:'untouched'}};
 const before=structuredClone(timeline);
 const {openClipAssembly}=await import('../../js/minimax_h3_clips.js');
 const cleanup=openClipAssembly({api:{fetchApi:async url=>{calls.push(url);return {ok:true,json:async()=>url.endsWith('sessions')?{sessions:[]}:{clips:[]}};}},timeline,onChange:s=>{timeline.clip_assembly=s;},viewUrl:p=>p,upload:async()=> 'uploaded.mp4'});
 const dialog=document.body.children.at(-1);
 assert.equal(dialog['aria-label'],'Join clips');assert.equal(dialog.open,true);
 await click(all(dialog).find(e=>e.textContent==='Video files'));
 await click(all(dialog).find(e=>e.textContent==='Use enabled timeline videos'));
 assert.equal(timeline.clip_assembly.rows[0].filename,'a.mp4');
 const {clip_assembly,...rest}=timeline;assert.deepEqual(rest,before);
 assert.ok(calls.every(url=>!url.includes('prompt')&&!url.includes('preflight')));
 assert.ok(all(dialog).some(e=>e.tagName==='LEGEND'&&e.textContent==='H3 transitions'));
 cleanup();assert.equal(dialog.isConnected,false);assert.equal(document.activeElement,prior);
});
