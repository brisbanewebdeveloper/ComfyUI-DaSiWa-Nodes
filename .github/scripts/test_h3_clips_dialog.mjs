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
 assert.ok(all(dialog).some(e=>e.textContent?.startsWith('b · +5.00s')));
 assert.equal(all(dialog).find(e=>e['aria-label']==='Select bad').disabled,true);
 const bRow=all(dialog).find(e=>e.children.some?.(c=>c.textContent?.startsWith('b ·')));
 await click(bRow.children.find(e=>e.textContent==='Preview'));
 assert.ok(all(dialog).some(e=>e.tagName==='IMG'&&e.src==='/df_h3_continuity/tail/s/b/0'));
 await click(bRow.children.find(e=>e.textContent==='Chain to here'));
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
 assert.ok(all(dialog).find(e=>e.type==='checkbox'&&e.disabled));
 cleanup();assert.equal(dialog.isConnected,false);assert.equal(document.activeElement,prior);
});
