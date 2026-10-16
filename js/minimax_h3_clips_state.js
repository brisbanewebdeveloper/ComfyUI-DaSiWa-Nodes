export function ancestorChain(clips, leaf) {
  const chain=[], seen=new Set();
  while(leaf) {
    if(seen.has(leaf)) throw new Error('Continuity parent cycle.');
    seen.add(leaf);
    const row=clips.find(clip=>clip.clip_id===leaf);
    if(!row) throw new Error('Missing parent checkpoint.');
    chain.unshift(row); leaf=row.parent_id;
  }
  return chain;
}
export function clipUnavailable(clip) {
  if(clip.status !== 'ready') return 'Not ready';
  if(clip.latent_available === false || clip.checkpoint_available === false) return 'Missing latent checkpoint';
  if(clip.export_available === false || clip.output_available === false) return 'Missing export file';
  return '';
}
export function selectContinuityRows(clips, ids, session) {
  const selected=ids.map(id=>clips.find(clip=>clip.clip_id===id));
  if(selected.some(clip=>!clip || clipUnavailable(clip))) throw new Error('Selected checkpoint is unavailable.');
  for(let i=1;i<selected.length;i++) {
    if(!ancestorChain(clips,selected[i].clip_id).some(clip=>clip.clip_id===selected[i-1].clip_id)) throw new Error('Alternative takes; not a continuation chain. Use separate video files for a free montage.');
  }
  let full=-1;
  selected.forEach((clip,index)=>{if(clip.export?.kind !== 'segment') full=index;});
  return selected.slice(Math.max(0,full)).map(clip=>({kind:'checkpoint',session,clip_id:clip.clip_id}));
}
export function snapshotTimelineVideos(items) {
  if (!Array.isArray(items)) throw new Error('Timeline items must be an array.');
  return items.map((item,index)=>({item,index})).filter(({item}) => item?.type === 'video' && item.enabled !== false && item.media_mode !== 'audio' && typeof item.value === 'string' && item.value.trim())
    .sort((a,b)=>(a.item.order ?? a.index)-(b.item.order ?? b.index))
    .map(({item})=>({id:item.id,kind:'media',filename:item.value,trim_start:item.trim_start ?? 0,trim_end:item.trim_end ?? null}));
}
export function assemblyPayload(raw) {
  const state=normalizeClipAssembly(raw);
  const rows=state.rows.filter(row => row.selected !== false && (state.tab === 'files' ? row.kind === 'media' : row.kind === 'checkpoint')).map(row=>{
    if(row.kind === 'checkpoint') {
      if (![row.session,row.clip_id].every(value=>typeof value === 'string' && /^[A-Za-z0-9_-]+$/.test(value))) throw new Error('Invalid checkpoint identity.');
      return {kind:'checkpoint',session:row.session,clip_id:row.clip_id};
    }
    const filename=row.filename;
    if(typeof filename !== 'string' || !filename.trim() || /[\\\\:\x00-\x1f?#]/.test(filename) || filename.startsWith('/') || filename.split('/').some(part=>!part || part==='..' || part==='.')) throw new Error('Invalid media handle.');
    const start=row.trim_start ?? 0, end=row.trim_end ?? null;
    if(typeof start !== 'number' || !Number.isFinite(start) || start < 0 || end !== null && (typeof end !== 'number' || !Number.isFinite(end) || end<=start)) throw new Error('Invalid video trims.');
    return {kind:'media',filename,trim_start:start,trim_end:end};
  });
  if(!rows.length) throw new Error('Select at least one clip.');
  if(rows.length>100) throw new Error('Select at most 100 clips.');
  return {version:1,rows,repair_transitions:false,audio_policy:'preserve',filename_prefix:state.filename_prefix};
}
export function normalizeClipAssembly(raw = {}) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid clip assembly settings.');
  if (raw.version != null && raw.version !== 1) throw new Error('Unsupported clip assembly version.');
  if (raw.tab != null && !['continuity', 'files'].includes(raw.tab)) throw new Error('Invalid clip assembly tab.');
  if (raw.rows != null && !Array.isArray(raw.rows)) throw new Error('Clip rows must be an array.');
  if (raw.repair_transitions != null && typeof raw.repair_transitions !== 'boolean') throw new Error('Invalid transition setting.');
  if (raw.audio_policy != null && raw.audio_policy !== 'preserve') throw new Error('Only preserve audio is currently available.');
  if (raw.filename_prefix != null && (typeof raw.filename_prefix !== 'string' || !/^[A-Za-z0-9_-]{1,80}$/.test(raw.filename_prefix))) throw new Error('Invalid output name.');
  if (raw.context_frames != null && (!Number.isInteger(raw.context_frames) || raw.context_frames < 5 || raw.context_frames % 17 !== 5)) throw new Error('Context per side must be 5, 22, 39 … frames.');
  if (raw.transition_frames != null && (!Number.isInteger(raw.transition_frames) || raw.transition_frames < 12 || raw.transition_frames % 17 !== 12)) throw new Error('Transition length must be 12, 29, 46 … frames.');
  if (raw.transition_prompt != null && (typeof raw.transition_prompt !== 'string' || raw.transition_prompt.length > 4000)) throw new Error('Transition description must be text of at most 4000 characters.');
  return {version: 1, tab: raw.tab ?? 'continuity', rows: structuredClone(raw.rows ?? []), repair_transitions: raw.repair_transitions ?? false, audio_policy: 'preserve', filename_prefix: raw.filename_prefix ?? 'H3_Joined',
    context_frames: raw.context_frames ?? 22, transition_frames: raw.transition_frames ?? 29, transition_prompt: raw.transition_prompt ?? ''};
}
export function transitionRequests(raw) {
  const state=normalizeClipAssembly(raw), rows=assemblyPayload({...state,tab:'files'}).rows;
  return rows.slice(1).map((row,index)=>({version:1,rows:[rows[index],row],context_frames:state.context_frames,transition_frames:state.transition_frames,prompt:state.transition_prompt.trim(),filename_prefix:'H3_Transition'}));
}
export function interleaveBridges(raw, bridges) {
  const rows=assemblyPayload({...raw,tab:'files'}).rows;
  return {...normalizeClipAssembly(raw),tab:'files',rows:rows.flatMap((row,index)=>index<bridges.length?[row,{kind:'media',filename:bridges[index],trim_start:0,trim_end:null}]:[row])};
}
/** Reuse this Director's own model chain: swap its Guide for the Transition Guide inside the sampler's upstream closure. */
export function buildTransitionPrompt(output, directorId, spec, requestId) {
  const guides=Object.entries(output).filter(([,node])=>node.class_type==='MiniMaxH3DirectorGuide'&&Array.isArray(node.inputs?.guide)&&String(node.inputs.guide[0])===String(directorId)&&node.inputs.guide[1]===0);
  if(guides.length!==1) throw new Error('Connect exactly one Director Guide to this Director to render H3 transitions.');
  const [guideId,guide]=guides[0];
  if(!Array.isArray(guide.inputs.audio_vae)) throw new Error('Connect the H3 audio VAE to the Director Guide to render transitions.');
  const link=value=>Array.isArray(value)&&value.length===2&&Object.hasOwn(output,String(value[0]))&&Number.isInteger(value[1]);
  const samplers=Object.keys(output).filter(id=>Object.values(output[id].inputs??{}).some(value=>link(value)&&String(value[0])===guideId&&value[1]===1));
  if(samplers.length!==1) throw new Error('The Director Guide latent must feed exactly one sampler; transitions reuse that sampler.');
  const prompt={}, pending=[samplers[0]];
  while(pending.length) {
    const id=String(pending.pop());
    if(Object.hasOwn(prompt,id)) continue;
    prompt[id]=structuredClone(output[id]);
    for(const value of Object.values(output[id].inputs??{})) if(link(value)) pending.push(value[0]);
  }
  prompt[guideId]={class_type:'DaSiWaH3TransitionGuide',inputs:{...prompt[guideId].inputs,transition_data:JSON.stringify(spec)}};
  let renderId=`ds-h3-transition-${requestId}`;
  while(Object.hasOwn(output,renderId)) renderId+='-render';
  prompt[renderId]={class_type:'DaSiWaH3TransitionRender',inputs:{latent:[samplers[0],0],vae:guide.inputs.vae,audio_vae:guide.inputs.audio_vae,transition:[guideId,2]}};
  return {prompt,renderId};
}
