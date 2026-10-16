import {assemblyPayload, normalizeClipAssembly, snapshotTimelineVideos, ancestorChain, selectContinuityRows, clipUnavailable, transitionRequests, interleaveBridges, buildTransitionPrompt} from './minimax_h3_clips_state.js';

async function requestJSON(api,url,options,signal) {
  signal?.throwIfAborted();
  const response=await api.fetchApi(url,{...options,signal});
  let body;
  try { body=await response.json(); } catch { throw new Error(`Unreadable response (${response.status}) from ${url}.`); }
  if(!response.ok) throw new Error(`Request failed (${response.status}): ${JSON.stringify(body)}`);
  signal?.throwIfAborted(); return body;
}
export async function preflightClipAssembly(api,state,signal) {
  const body=await requestJSON(api,'/df_h3_continuity/assembly/preflight',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(assemblyPayload(state))},signal);
  if(body.ok !== true) throw new Error(`Cannot join: ${JSON.stringify(body.issues ?? body)}`);
  return body;
}
export async function queueClipAssembly({api,state,signal,onStatus=()=>{},confirmPlan=async()=>true,pollMs=1000}) {
  onStatus('validating');
  const plan=await preflightClipAssembly(api,state,signal);
  if(!await confirmPlan(plan)) throw new Error('Join cancelled before queueing.');
  signal?.throwIfAborted();
  const {prompt,assembleId}=buildClipAssemblyPrompt({},state);
  const result=await queueAndWait({api,prompt,outputId:assembleId,key:'h3_clip_assembly',signal,onStatus,pollMs});
  onStatus('verifying');
  if(typeof result.filename !== 'string' || !result.filename || !Number.isFinite(result.seconds ?? result.duration) || typeof result.mode !== 'string') throw new Error('Invalid assembly result.');
  onStatus('complete'); return result;
}
/** One H3 render per boundary through this Director's sampler, then one model-free join with the bridges inserted. */
export async function queueTransitionJoin({api,state,graphPrompt,directorId,signal,onStatus=()=>{},confirmPlan=async()=>true,pollMs=1000}) {
  const requests=transitionRequests(state);
  if(!requests.length) throw new Error('Select at least two clips to generate H3 transitions.');
  onStatus('validating');
  const plan=await preflightClipAssembly(api,{...state,tab:'files'},signal);
  if(plan.rows?.some(row=>row.frames<requests[0].context_frames)) throw new Error('A selected clip is shorter than the transition context; trim less or reduce context.');
  if(!await confirmPlan({...plan,transitions:requests.length})) throw new Error('Join cancelled before queueing.');
  const bridges=[];
  for(const [index,spec] of requests.entries()) {
    signal?.throwIfAborted();
    onStatus(`Rendering H3 transition ${index+1} of ${requests.length}…`);
    const {output,workflow}=await graphPrompt();
    const {prompt,renderId}=buildTransitionPrompt(output,directorId,spec,crypto.randomUUID());
    const result=await queueAndWait({api,prompt,outputId:renderId,key:'h3_transition',signal,onStatus:()=>{},pollMs,workflow});
    if(typeof result.filename !== 'string' || !result.filename.startsWith('h3_transitions/')) throw new Error('Invalid transition result.');
    bridges.push(result.filename);
  }
  return queueClipAssembly({api,state:interleaveBridges(state,bridges),signal,onStatus,pollMs});
}
async function queueAndWait({api,prompt,outputId,key,signal,onStatus,pollMs,workflow}) {
  const queued=await requestJSON(api,'/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_id:api.clientId,prompt,...(workflow?{extra_data:{extra_pnginfo:{workflow}}}:{})})},signal);
  if(!queued.prompt_id) throw new Error('Queue returned no prompt ID.');
  onStatus('queued');
  while(true) {
    let history;
    try { history=await requestJSON(api,`/history/${encodeURIComponent(queued.prompt_id)}`,{},signal); }
    catch(error) { if(error.name==='AbortError') throw error; const failure=new Error(`Monitoring stopped; job ${queued.prompt_id} may still run. Check ComfyUI history before retrying. ${error.message}`); failure.queuedPromptId=queued.prompt_id; throw failure; }
    const entry=history[queued.prompt_id];
    const failure=entry?.status?.messages?.find(([kind])=>['execution_error','execution_interrupted'].includes(kind));
    if(failure || entry?.status?.status_str==='error') throw new Error(`Job failed/interrupted: ${JSON.stringify(failure ?? entry.status)}`);
    const result=entry?.outputs?.[outputId]?.[key]?.[0];
    if(result) return result;
    if(entry?.status?.completed) throw new Error('Job completed without a result. Check server logs.');
    onStatus('processing');
    await new Promise((resolve,reject)=>{
      const abort=()=>{clearTimeout(timer);reject(signal.reason);};
      const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve();},pollMs);
      signal?.addEventListener('abort',abort,{once:true});
    });
    signal?.throwIfAborted();
  }
}
export function openClipAssembly({api,timeline,onChange,viewUrl,upload,graphPrompt=null,directorId=null,allowNativeEditing=()=>{},onStatus=()=>{},isRemoved=()=>false}) {
  const state=normalizeClipAssembly(timeline.clip_assembly), prior=document.activeElement, controller=new AbortController();
  const listeners=[], media=[], durations=new Map();
  let closed=false,busy=false,locked=false,clips=[],session=state.rows.find(row=>row.kind==='checkpoint')?.session ?? timeline.continuity?.session ?? '',loadRevision=0;
  const el=(tag,text)=>{const item=document.createElement(tag);if(text!=null)item.textContent=text;return item;};
  const listen=(target,type,action)=>{target.addEventListener(type,action);listeners.push(()=>target.removeEventListener(type,action));};
  const dialog=el('dialog');dialog.className='ds-h3-clips';dialog.setAttribute('aria-label','Join clips');dialog.setAttribute('aria-modal','true');
  const style=el('style',`.ds-h3-clips{box-sizing:border-box;width:min(960px,94vw);max-height:90vh;overflow:auto;padding:20px;border:1px solid #775032;border-radius:12px;background:var(--comfy-menu-bg,#17120f);color:var(--input-text,#e7c4a2);font:13px/1.5 system-ui}.ds-h3-clips::backdrop{background:#05090dc9}.ds-h3-clips header,.ds-h3-clips nav,.ds-h3-clips footer,.ds-h3-clips .clip-row{display:flex;align-items:center;gap:9px;margin:8px 0}.ds-h3-clips header{justify-content:space-between}.ds-h3-clips h2{margin:0}.ds-h3-clips button{padding:6px 10px;border:1px solid #775032;border-radius:7px;background:#251a12;color:inherit;cursor:pointer}.ds-h3-clips button:hover:not(:disabled){background:#493020;border-color:#ac6834}.ds-h3-clips button:disabled{opacity:.5;cursor:default}.ds-h3-clips :focus-visible{outline:2px solid #e9a262;outline-offset:2px}.ds-h3-clips input,.ds-h3-clips select{background:var(--comfy-input-bg,#211913);color:inherit;border:1px solid #775032;border-radius:4px;padding:5px}.ds-h3-clips .clip-row{border-bottom:1px solid #624027;padding:7px 0}.ds-h3-clips .clip-name{flex:1;overflow-wrap:anywhere;min-width:100px}.ds-h3-clips .clip-row input[type=number]{width:75px}.ds-h3-clips video{display:block;max-width:100%;max-height:260px;margin:10px auto}.ds-h3-clips .clip-status{white-space:pre-wrap;overflow-wrap:anywhere}.ds-h3-clips label{display:inline-flex;align-items:center;gap:7px;margin:6px 0}.ds-h3-clips>label{display:flex;width:100%;box-sizing:border-box}.ds-h3-clips nav button[aria-selected=true]{background:#493020}.ds-h3-clips .clip-list{overflow-x:auto}.ds-h3-clips .clip-trim{flex-direction:column;align-items:flex-start;gap:2px;margin:0}.ds-h3-clips .clip-row .clip-trim input{width:110px;box-sizing:border-box}.ds-h3-clips .clip-time{font-size:11px;opacity:.8;margin-top:4px}.ds-h3-clips .clip-file{display:grid;align-items:center;grid-template-columns:24px minmax(140px,1fr) 78px 110px 110px 78px 34px 34px;gap:8px}.ds-h3-clips .clip-row.clip-file input[type=number]{width:100%;box-sizing:border-box}.ds-h3-clips .clip-head{margin:8px 0 -4px;font-size:12px;opacity:.85}.ds-h3-clips .clip-settings{display:flex;align-items:flex-end;gap:12px;margin:8px 0}.ds-h3-clips fieldset{border:1px solid #624027;border-radius:8px;margin:12px 0;padding:6px 12px}.ds-h3-clips textarea{box-sizing:border-box;background:var(--comfy-input-bg,#211913);color:inherit;border:1px solid #775032;border-radius:4px;padding:5px;font:inherit}.ds-h3-clips h3{margin-bottom:4px}`);
  const button=(text,action,parent)=>{const item=el('button',text);item.type='button';listen(item,'click',action);parent.append(item);return item;};
  const header=el('header');header.append(el('h2','Join clips'));button('Close',()=>cleanup(),header);dialog.append(header);
  const nav=el('nav');nav.setAttribute('role','tablist');dialog.append(nav);
  const content=el('section');content.setAttribute('role','tabpanel');dialog.append(content);
  const status=el('p','Source files and the pinned Continuity source remain unchanged.');status.className='clip-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  const previewPanel=el('div');dialog.append(previewPanel);
  const name=el('input');name.value=state.filename_prefix;name.setAttribute('aria-label','Output filename prefix');allowNativeEditing(name);
  listen(name,'change',()=>{state.filename_prefix=name.value;save();});
  const nameLabel=el('label');nameLabel.append(el('span','Output filename prefix'),name);name.title='Prefix for the new joined file. A unique suffix is added automatically; source files are never overwritten.';dialog.append(nameLabel,status);
  const actions=el('footer');dialog.append(actions);
  const report=message=>{if(!closed&&!isRemoved()){status.textContent=message;onStatus(message);}};
  const save=()=>{onChange(normalizeClipAssembly(state));};
  const stopMedia=()=>media.splice(0).forEach(item=>{item.pause();item.removeAttribute('src');item.load();});
  const exportUrl=descriptor=>{
    const path=`/view?${new URLSearchParams({filename:descriptor.filename,subfolder:descriptor.subfolder ?? '',type:descriptor.type ?? 'output'})}`;
    return api.apiURL?.(path) ?? path;
  };
  const preview=(path,start=0,end=null)=>{
    stopMedia();previewPanel.replaceChildren();if(!path)return report('No export preview available. Missing export files and missing latent checkpoints are checked separately by preflight.');
    const video=el('video');video.controls=true;video.preload='metadata';video.src=path;media.push(video);
    listen(video,'loadedmetadata',()=>{video.currentTime=start;});
    listen(video,'timeupdate',()=>{if(end!=null&&video.currentTime>=end)video.pause();});previewPanel.append(video);
  };
  const cleanup=()=>{
    if(closed)return;closed=true;controller.abort();listeners.splice(0).forEach(remove=>remove());stopMedia();dialog.close();dialog.remove();style.remove();
    if(prior?.isConnected)prior.focus();
    if(busy&&!isRemoved())onStatus('Monitoring stopped. Submitted jobs continue; use ComfyUI queue controls to cancel only that job.');
  };
  listen(dialog,'cancel',event=>{event.preventDefault();cleanup();});listen(dialog,'close',cleanup);
  listen(dialog,'keydown',event=>{
    event.stopPropagation();
    if(event.key==='Escape'){event.preventDefault();cleanup();}
    if(event.key==='Tab') {
      const controls=[...dialog.querySelectorAll('button:not(:disabled),input:not(:disabled):not([hidden]),select:not(:disabled),video[controls]')].filter(item=>!item.disabled&&!item.hidden);
      const first=controls[0],last=controls.at(-1);
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
    }
  });
  const payloadState=()=>{
    const next=normalizeClipAssembly(state);
    if(state.tab==='continuity') {
      const rows=state.rows.filter(row=>row.kind==='checkpoint'&&row.selected!==false);
      if(rows.some(row=>row.session!==session))throw new Error('Choose one session explicitly; cross-session chains are not supported.');
      next.rows=selectContinuityRows(clips,rows.map(row=>row.clip_id),session);
    }
    return next;
  };
  const run=async()=>{
    if(busy||locked)return;
    try {
      const snapshot=payloadState();assemblyPayload(snapshot);busy=true;render();
      const transitions=snapshot.tab==='files'&&snapshot.repair_transitions;
      const confirmPlan=async plan=>{
        if(closed)return false;
        const total=Number(plan.seconds)+(plan.transitions??0)*snapshot.transition_frames/24;
        report(`Duration: ${total.toFixed(3)}s · ${plan.transitions ? (plan.frames ?? 0)+plan.transitions*snapshot.transition_frames : plan.frames ?? '?'} frames · ${plan.mode}${plan.width && plan.height ? ` · ${plan.width}×${plan.height} · ${plan.fps} fps` : ''}. ${plan.transitions ? `Then ${plan.transitions} H3 transition render${plan.transitions>1?'s':''} through this Director's sampler (${(snapshot.transition_frames/24).toFixed(2)}s each, added between clips) before a model-free join.` : 'Audio may be decoded/encoded once for accurate joining. No AI rendering.'}${Array.isArray(plan.notes) ? '\n'+plan.notes.join('\n') : ''}`);
        return await new Promise(resolve=>{
          const confirm=button(plan.transitions?'Render transitions and join':/normal/i.test(plan.mode)?'Normalize and join':/encod/i.test(plan.mode)?'Re-encode and join':'Confirm join',()=>{confirm.remove();cancel.remove();resolve(true);},actions);
          const cancel=button('Cancel join',()=>{confirm.remove();cancel.remove();resolve(false);},actions);
          controller.signal.addEventListener('abort',()=>resolve(false),{once:true});confirm.focus();
        });
      };
      const options={api,state:snapshot,signal:controller.signal,onStatus:report,confirmPlan};
      const result=await (transitions?queueTransitionJoin({...options,graphPrompt,directorId}):queueClipAssembly(options));
      if(closed)return;
      report(`${result.filename} · ${Number(result.seconds ?? result.duration).toFixed(3)}s · ${result.mode}`);
      const descriptor=Array.isArray(result.preview)?result.preview[0]:result.preview;
      const url=descriptor&&typeof descriptor==='object'?exportUrl(descriptor):null;
      preview(url ?? viewUrl(result.filename,'output'));
    } catch(error){if(error.queuedPromptId)locked=true;if(error.name!=='AbortError')report(error.message);}
    finally{busy=false;if(!closed)render();}
  };
  const join=button('Join only — no AI',run,actions);
  const loadSession=async()=>{
    const revision=++loadRevision;clips=[];render();
    if(!session)return;
    try{
      const data=await requestJSON(api,`/df_h3_continuity/session/${encodeURIComponent(session)}`,{},controller.signal);
      if(closed||revision!==loadRevision)return;
      clips=Array.isArray(data.clips)?data.clips:[];render();
    }catch(error){if(error.name!=='AbortError')report(error.message);}
  };
  const durationFor=filename=>{
    if(!durations.has(filename)){
      const info={loading:true,updates:new Set()};durations.set(filename,info);
      void requestJSON(api,`/df_h3_continuity/assembly/media-info?${new URLSearchParams({filename})}`,{},controller.signal).then(data=>{
        if(!Number.isFinite(data.seconds)||data.seconds<=0)throw new Error('No finite source duration was returned.');
        info.seconds=data.seconds;
      }).catch(error=>{if(error.name!=='AbortError')info.error=error.message;}).finally(()=>{
        info.loading=false;if(!closed&&!isRemoved())for(const update of info.updates)update();
      });
    }
    return durations.get(filename);
  };
  const render=()=>{
    for(const info of durations.values())info.updates.clear();
    content.replaceChildren();nav.replaceChildren();
    for(const [tab,label]of [['continuity','Continuity segments'],['files','Video files']]){
      const item=button(label,()=>{if(busy)return;state.tab=tab;save();render();},nav);item.setAttribute('role','tab');item.setAttribute('aria-selected',String(state.tab===tab));item.disabled=busy;
    }
    join.disabled=busy||locked;name.disabled=busy;
    const tools=el('div');content.append(tools);
    if(state.tab==='files') {
      const input=el('input');input.type='file';input.accept='video/*,.mkv,.avi,.mov,.webm,.mp4,.m4v';input.multiple=true;input.hidden=true;tools.append(input);
      button('Add videos…',()=>input.click(),tools).disabled=busy;
      listen(input,'change',async()=>{
        if(busy)return;busy=true;render();
        try {for(const file of input.files ?? []){const filename=await upload(file);if(closed)return;state.rows.push({kind:'media',filename,trim_start:0,trim_end:null});save();}}
        catch(error){report(error.message);}finally{busy=false;if(!closed)render();}
      });
      button('Use enabled timeline videos',()=>{state.rows=state.rows.filter(row=>row.kind!=='media'||!row.timeline_snapshot).concat(snapshotTimelineVideos(timeline.items ?? []).map(row=>({...row,timeline_snapshot:true})));save();render();},tools).disabled=busy;
    } else {
      const sessions=el('select');sessions.setAttribute('aria-label','Assembly session');sessions.title='Saved Continuity session to browse for montage. This does not change the Director’s pinned source.';const sessionLabel=el('label');sessionLabel.append(el('span','Continuity session'),sessions);tools.append(sessionLabel);
      const option=el('option',session||'Choose a saved session');option.value=session;sessions.append(option);sessions.value=session;sessions.disabled=busy;
      listen(sessions,'change',()=>{session=sessions.value;state.rows=state.rows.filter(row=>row.kind!=='checkpoint');save();void loadSession();});
      void requestJSON(api,'/df_h3_continuity/sessions',{},controller.signal).then(data=>{
        if(closed)return;for(const entry of data.sessions ?? []){if(entry.session===session)continue;const opt=el('option',entry.session);opt.value=entry.session;sessions.append(opt);}
      }).catch(error=>{if(error.name!=='AbortError')report(error.message);});
      button('Refresh',()=>void loadSession(),tools).disabled=busy;
    }
    content.append(el('h3','Clips in playback order'));
    content.append(el('p',state.tab==='files'?'Start and End are positions in each source file, in seconds; End starts at the full length. Uncheck to skip a clip; Remove only clears it from this list.':'Each row is one saved take. Tick the ones you want; they play from top to bottom. “Select chain to here” ticks a take plus all the takes that led up to it. Several takes of the same step are alternatives, so pick just one. Nothing here deletes a take.'));
    const list=el('div');list.className='clip-list';content.append(list);
    if(state.tab==='files'){const head=el('div');head.className='clip-file clip-head';head.setAttribute('aria-hidden','true');for(const text of ['','','','Start (s)','End (s)','','',''])head.append(el('span',text));list.append(head);}
    const selected=state.rows.filter(row=>row.kind===(state.tab==='files'?'media':'checkpoint'));
    const candidates=state.tab==='files'?selected:[...selected.map(row=>({...clips.find(clip=>clip.clip_id===row.clip_id),...row})),...clips.filter(clip=>!selected.some(row=>row.clip_id===clip.clip_id)).map(clip=>({...clip,kind:'checkpoint',session,selected:false}))];
    for(const [index,row] of candidates.entries()) {
      const line=el('div');line.className=state.tab==='files'?'clip-row clip-file':'clip-row';list.append(line);
      const reason=state.tab==='continuity'?clipUnavailable(row):'';
      const check=el('input');check.type='checkbox';check.checked=row.selected!==false;check.disabled=busy||!!reason;check.setAttribute('aria-label',`Select ${row.clip_id ?? row.filename}`);line.append(check);
      listen(check,'change',()=>{let saved=state.rows.find(item=>item===row || row.kind==='checkpoint'&&item.kind==='checkpoint'&&item.clip_id===row.clip_id&&item.session===session);if(!saved){saved={kind:'checkpoint',session,clip_id:row.clip_id};state.rows.push(saved);}saved.selected=check.checked;save();render();});
      const label=el('span',state.tab==='files'?row.filename:`${row.clip_id} · adds ${Number((row.segment_frames ?? row.extension_frames ?? (row.parent_id ? Math.max(0,(row.frames ?? 0)-(clips.find(clip=>clip.clip_id===row.parent_id)?.frames ?? row.frames ?? 0)) : row.frames ?? 0))/(row.fps || 24)).toFixed(2)} s · ends at ${Number(row.seconds ?? 0).toFixed(2)} s · ${row.parent_id ? `continues ${row.parent_id}` : 'first take'} · ${reason || (row.export?.source_prefix?'imported start video':row.export?.kind==='segment'?'segment export':'full-timeline export')}`);label.className='clip-name';line.append(label);
      let paintDuration=()=>{},endInput=null;
      if(state.tab==='files'){
        const timing=el('div');timing.className='clip-time';label.append(timing);
        const info=durationFor(row.filename);
        paintDuration=()=>{
          if(Number.isFinite(info.seconds)){
            if(endInput&&row.trim_end==null&&endInput.value===''&&document.activeElement!==endInput)endInput.value=info.seconds.toFixed(3);
            const start=row.trim_start??0,end=row.trim_end??info.seconds;
            const selected=Number.isFinite(start)&&Number.isFinite(end)&&start>=0&&end>start&&end<=info.seconds?`${(end-start).toFixed(3)} s`:'Invalid trim';
            timing.textContent=`Source duration: ${info.seconds.toFixed(3)} s · Selected: ${selected}`;
          }else timing.textContent=info.loading?'Reading source duration…':'Duration unavailable — Join preflight will check this file.';
          timing.title=info.error ?? 'Source duration is read from the file; selected duration uses Start and End. Joining snaps cuts to the output frame grid.';
        };
        info.updates.add(paintDuration);
      }
      button('Preview',()=>{
        if(state.tab==='continuity' && row.thumbnails?.length && !row.export?.filename){
          stopMedia();previewPanel.replaceChildren();const image=el('img');image.alt=`Tail preview of ${row.clip_id}`;
          const path=`/df_h3_continuity/tail/${encodeURIComponent(session)}/${encodeURIComponent(row.clip_id)}/0`;
          image.src=api.apiURL?.(path) ?? path;image.style.maxWidth='100%';image.style.maxHeight='260px';previewPanel.append(image);
        }else preview(state.tab==='files'?viewUrl(row.filename):row.export?.filename?exportUrl(row.export):null,row.trim_start??0,row.trim_end);
      },line).disabled=busy||!!reason;
      if(state.tab==='continuity'){
        const chain=button('Select chain to here',()=>{
        try{const chain=ancestorChain(clips,row.clip_id);if(chain.some(clip=>clipUnavailable(clip)))throw new Error('Ancestor chain contains an unavailable checkpoint.');state.rows=state.rows.filter(item=>item.kind!=='checkpoint').concat(chain.map(clip=>({kind:'checkpoint',session,clip_id:clip.clip_id})));save();render();}catch(error){report(error.message);}
        },line);
        chain.disabled=busy||!!reason;chain.title='Check this take and every earlier take it continues from, in playback order. Replaces the current selection.';
      }
      else for(const [field,title]of [['trim_start','Start (s)'],['trim_end','End (s)']]){
        const trim=el('input');trim.type='number';trim.min='0';trim.step='0.001';trim.value=row[field]??'';trim.disabled=busy;trim.setAttribute('aria-label',`${row.filename} ${title}`);
        trim.title=field==='trim_start'?'Source position where this clip starts, in seconds.':'Source position where this clip ends, in seconds. Leave blank to use the file end.';
        if(field==='trim_end'){trim.placeholder='File end';endInput=trim;}
        allowNativeEditing(trim);line.append(trim);
        listen(trim,'change',()=>{row[field]=trim.value===''&&field==='trim_end'?null:Number(trim.value);save();paintDuration();});
      }
      if(index<selected.length){
        const remove=button(state.tab==='files'?'Remove':'Remove selection',()=>{
          const saved=row.kind==='checkpoint'?state.rows.find(item=>item.kind==='checkpoint'&&item.clip_id===row.clip_id&&item.session===session):row;
          state.rows=state.rows.filter(item=>item!==saved);stopMedia();previewPanel.replaceChildren();save();render();
        },line);
        remove.disabled=busy;remove.title='Remove from this montage selection only. The source file, checkpoint and Director references are not deleted.';
        remove.setAttribute('aria-label',`Remove ${row.clip_id ?? row.filename} from montage`);
      }
      paintDuration();
      for(const [delta,text]of [[-1,'↑'],[1,'↓']]) {
        const move=button(text,()=>{
          const current=state.rows.indexOf(row.kind==='checkpoint'?state.rows.find(item=>item.kind==='checkpoint'&&item.clip_id===row.clip_id&&item.session===session):row);
          const siblings=state.rows.filter(item=>item.kind===row.kind),position=siblings.indexOf(state.rows[current]),other=state.rows.indexOf(siblings[position+delta]);
          if(current<0||other<0)return;[state.rows[current],state.rows[other]]=[state.rows[other],state.rows[current]];save();render();
        },line);move.setAttribute('aria-label',`Move ${row.clip_id ?? row.filename} ${delta<0?'up':'down'}`);move.disabled=busy||index>=selected.length||index+delta<0||index+delta>=selected.length;
      }
    }
    if(!candidates.length)list.append(el('p',state.tab==='files'?'Add videos or copy enabled timeline videos.':'No saved checkpoints in this session.'));
    join.textContent=state.tab==='files'&&state.repair_transitions?'Join with H3 transitions':'Join only — no AI';
    if(state.tab==='files'){
      const section=el('fieldset');section.className='clip-transitions';section.append(el('legend','H3 transitions'));content.append(section);
      const toggle=el('input');toggle.type='checkbox';toggle.checked=state.repair_transitions;toggle.disabled=busy||!graphPrompt;
      toggle.title='Renders a short bridge for every cut with this Director’s Guide, model and sampler: the end of the earlier clip and the start of the next clip are H3 guides, only the generated middle is inserted.';
      listen(toggle,'change',()=>{state.repair_transitions=toggle.checked;save();render();});
      const toggleLabel=el('label');toggleLabel.append(toggle,el('span','Generate an H3 transition between each pair of clips'));section.append(toggleLabel);
      if(!graphPrompt)section.append(el('p','Open Join clips from a Director that is wired to a Director Guide and sampler to render transitions.'));
      const select=(field,title,tip,values,format)=>{
        const input=el('select');input.disabled=busy||!state.repair_transitions;input.title=tip;
        for(const value of values){const option=el('option',format(value));option.value=String(value);input.append(option);}
        input.value=String(state[field]);listen(input,'change',()=>{state[field]=Number(input.value);save();});
        const label=el('label');label.className='clip-trim';label.append(el('span',title),input);return label;
      };
      const row=el('div');row.className='clip-settings';section.append(row);
      row.append(select('transition_frames','Transition length','Generated bridge inserted between the clips; it adds this much time per cut.',[12,29,46,63,80],v=>`${(v/24).toFixed(2)} s (${v} frames)`),
        select('context_frames','Context per side','How much of each clip edge H3 sees as a guide. These frames stay original; longer context helps match motion and sound.',[5,22,39],v=>`${(v/24).toFixed(2)} s (${v} frames)`));
      const prompt=el('textarea');prompt.rows=3;prompt.value=state.transition_prompt;prompt.disabled=busy||!state.repair_transitions;prompt.maxLength=4000;
      prompt.placeholder='Optional: what happens in the bridge (action, camera, sound). Leave empty for a neutral continuous bridge.';
      prompt.setAttribute('aria-label','Transition description');allowNativeEditing(prompt);listen(prompt,'change',()=>{state.transition_prompt=prompt.value;save();});
      const promptLabel=el('label');promptLabel.className='clip-trim';promptLabel.style.width='100%';prompt.style.width='100%';promptLabel.append(el('span','Transition description'),prompt);section.append(promptLabel);
    }
  };
  document.head.append(style);document.body.append(dialog);render();dialog.showModal();header.children[header.children.length-1].focus();void loadSession();
  return cleanup;
}

export function buildClipAssemblyPrompt(output, state) {
  const payload=assemblyPayload(state);
  let assembleId='h3-clip-assemble';
  while(Object.hasOwn(output,assembleId)) assembleId+='-join';
  return {assembleId,prompt:{[assembleId]:{class_type:'DaSiWaH3ClipAssemble',inputs:{assembly_data:JSON.stringify(payload),filename_prefix:payload.filename_prefix}}}};
}
