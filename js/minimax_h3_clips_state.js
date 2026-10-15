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
  if (state.repair_transitions) throw new Error('H3 repair is unavailable: hardware quality gate has not passed.');
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
  return {version: 1, tab: raw.tab ?? 'continuity', rows: structuredClone(raw.rows ?? []), repair_transitions: raw.repair_transitions ?? false, audio_policy: 'preserve', filename_prefix: raw.filename_prefix ?? 'H3_Joined'};
}
