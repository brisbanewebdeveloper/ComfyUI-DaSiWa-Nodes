const CONCEPTS = ['generic', 'identity', 'style', 'motion', 'audio'];
const isMedia = item => item && ['image', 'video', 'audio'].includes(item.type) && typeof item.value === 'string' && item.value.trim();

export function selectDaSiWaRefModSnapshot(snapshot, rows) {
  const selected = structuredClone(snapshot);
  for (const kind of ['items', 'refmods']) {
    selected[kind] = (selected[kind] || []).flatMap((item, index) => {
      if (kind === 'refmods' && Array.isArray(item.members)) {
        item.members = item.members.flatMap(member => {
          const row = rows.find(row => row.kind === kind && row.index === index && row.memberIndex === member.index);
          return row?.selected ? [{ index: member.index, description: row.description ?? '' }] : [];
        });
        return item.members.length ? [item] : [];
      }
      const row = rows.find(row => row.kind === kind && row.index === index);
      if (!row?.selected) return [];
      item.description = row.description ?? '';
      return [item];
    });
  }
  return selected;
}

// Export is a separate queue job, never a mutation of the Director workflow.
export function buildDaSiWaRefModExportPrompt(output, directorId, snapshot, requestId, settings = {}) {
  if (!snapshot || !Array.isArray(snapshot.items) || (snapshot.refmods != null && !Array.isArray(snapshot.refmods))) throw new Error('Invalid timeline: items and refmods must be arrays.');
  for (const row of snapshot.refmods ?? []) {
    if (row.members !== undefined && (!Array.isArray(row.members) || !row.members.length || row.members.some(member => !Number.isInteger(member?.index) || member.index < 0 || typeof member.description !== 'string'))) throw new Error('Select at least one saved member with a valid index and description.');
  }
  if (!snapshot.items.some(isMedia) && !snapshot.refmods?.some(row => row?.name)) throw new Error('Select at least one media reference or saved RefMod.');
  if (!CONCEPTS.includes(settings.concept_type ?? 'generic')) throw new Error('Unsupported concept type.');
  const prompt = {};
  const inputs = {
    timeline_data: JSON.stringify(snapshot), request_id: requestId,
    name: settings.name ?? '', description: settings.description ?? '', concept_type: settings.concept_type ?? 'generic',
  };
  const visual = snapshot.items.some(item => item.value && (item.type === 'image' || item.type === 'video' && item.media_mode !== 'audio'));
  const audio = snapshot.items.some(item => item.value && (item.type === 'audio' || item.type === 'video' && (['audio', 'video_audio'].includes(item.media_mode) || item.audio)));
  if (visual || audio) {
    const guides = Object.values(output).filter(node => node.class_type === 'MiniMaxH3DirectorGuide' && Array.isArray(node.inputs?.guide) && node.inputs.guide.length === 2 && node.inputs.guide[1] === 0 && String(node.inputs.guide[0]) === String(directorId));
    if (guides.length !== 1) throw new Error('Connect exactly one Director Guide to this Director before exporting raw media.');
    for (const key of [ ...(visual ? ['vae'] : []), ...(audio ? ['audio_vae'] : []) ]) {
      const link = guides[0].inputs[key];
      const loader = output[link?.[0]];
      if (!Array.isArray(link) || link[1] !== 0 || loader?.class_type !== 'VAELoader' || typeof loader.inputs?.vae_name !== 'string' || Object.values(loader.inputs).some(Array.isArray)) {
        throw new Error(`Connect a direct native VAELoader to Director Guide ${key}; custom loader topology is not supported by this button.`);
      }
      prompt[link[0]] = structuredClone(loader);
      inputs[key] = structuredClone(link);
    }
  }
  let exportId = `ds-h3-refmod-export-${requestId}`;
  while (Object.hasOwn(output, exportId)) exportId += '-export';
  prompt[exportId] = { class_type: 'DaSiWaH3RefModExport', inputs };
  return { prompt, exportId };
}

export async function queueDaSiWaRefModExport({ api, app, node, snapshot, settings, signal, onStatus = () => {}, isRemoved = () => false, pollMs = 1000 }) {
  const check = () => {
    signal?.throwIfAborted();
    if (isRemoved()) throw new DOMException('Export monitoring stopped.', 'AbortError');
  };
  const request = async (url, options = {}) => {
    check();
    const response = await api.fetchApi(url, { ...options, signal });
    let body;
    try { body = await response.json(); } catch { throw new Error(`Export request ${url} returned unreadable JSON (${response.status}).`); }
    if (!response.ok) throw new Error(`Export request failed (${response.status}): ${JSON.stringify(body)}`);
    check();
    return body;
  };
  check();
  const { output, workflow } = await app.graphToPrompt();
  check();
  const { prompt, exportId } = buildDaSiWaRefModExportPrompt(output, node.id, snapshot, crypto.randomUUID(), settings);
  onStatus('Submitting RefMod export…');
  const queued = await request('/prompt', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ client_id: api.clientId, prompt, extra_data: { extra_pnginfo: { workflow } } }),
  });
  if (!queued.prompt_id) throw new Error(`Export queue returned no prompt ID: ${JSON.stringify(queued)}`);
  onStatus('RefMod export queued. Closing stops monitoring, not the queued job.');
  while (true) {
    let history;
    try { history = await request(`/history/${encodeURIComponent(queued.prompt_id)}`); }
    catch (error) {
      if (error.name === 'AbortError') throw error;
      const failure = new Error(`Monitoring job ${queued.prompt_id} stopped; it may still be queued or running. Check ComfyUI history before retrying. ${error.message}`);
      failure.queuedPromptId = queued.prompt_id;
      throw failure;
    }
    const entry = history[queued.prompt_id];
    if (entry) {
      const failure = entry.status?.messages?.find(([kind]) => ['execution_error', 'execution_interrupted'].includes(kind));
      if (failure || entry.status?.status_str === 'error') throw new Error(`RefMod export failed: ${JSON.stringify(failure?.[1] ?? entry.status)}`);
      const result = entry.outputs?.[exportId]?.dasiwa_refmod_export?.[0];
      if (result) {
        if (typeof result.name !== 'string' || !result.name) throw new Error('Export returned an invalid saved name.');
        return result;
      }
      if (entry.status?.completed) throw new Error('RefMod export completed without a saved result. Check server logs.');
    }
    await new Promise((resolve, reject) => {
      const abort = () => { clearTimeout(timer); reject(signal.reason); };
      const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, pollMs);
      signal?.addEventListener('abort', abort, { once: true });
    });
    check();
  }
}

/** Call after emit() with a cloned timeline; onSaved refreshes the parent's library only. */
export function openDaSiWaRefModExport({ api, app, node, snapshot, viewUrl, onSaved = () => {}, onStatus = () => {}, isRemoved = () => false }) {
  const timelineSnapshot = structuredClone(snapshot);
  snapshot = structuredClone(timelineSnapshot);
  const priorFocus = document.activeElement;
  const controller = new AbortController();
  const listeners = [], mediaElements = [], rows = [];
  let closed = false, busy = false, locked = false, loading = false;
  const el = (tag, className, text) => {
    const element = document.createElement(tag);
    element.className = `ds-h3-refmod-export-${className}`;
    if (text != null) element.textContent = text;
    return element;
  };
  const listen = (target, type, handler) => { target.addEventListener(type, handler); listeners.push(() => target.removeEventListener(type, handler)); };
  const dialog = el('dialog', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-label', 'Create RefMod');
  const style = el('style', 'style', `
.ds-h3-refmod-export-dialog { box-sizing:border-box; width:min(960px,94vw); max-height:90vh; overflow:auto; padding:24px; border:1px solid #52616d; border-radius:14px; background:var(--comfy-menu-bg,#20252b); color:var(--input-text,#e6ebef); font:14px/1.5 sans-serif; }
.ds-h3-refmod-export-dialog::backdrop { background:#000a; }
.ds-h3-refmod-export-header,.ds-h3-refmod-export-actions { display:flex; justify-content:space-between; align-items:center; gap:16px; }
.ds-h3-refmod-export-title { margin:0; font:600 24px/1.2 Georgia,serif; }
.ds-h3-refmod-export-settings { display:grid; grid-template-columns:2fr 1fr; gap:12px; margin:20px 0; }
.ds-h3-refmod-export-field { display:flex; flex-direction:column; gap:6px; min-width:0; }
.ds-h3-refmod-export-wide { grid-column:1/-1; }
.ds-h3-refmod-export-input,.ds-h3-refmod-export-description,.ds-h3-refmod-export-concept { box-sizing:border-box; width:100%; padding:9px; background:var(--comfy-input-bg,#151a20); color:inherit; border:1px solid #52616d; border-radius:6px; font:inherit; }
.ds-h3-refmod-export-description { min-height:70px; resize:vertical; }
.ds-h3-refmod-export-row { display:grid; grid-template-columns:minmax(150px,240px) 1fr; gap:16px; padding:16px 0; border-top:1px solid #52616d; }
.ds-h3-refmod-export-label { overflow-wrap:anywhere; }
.ds-h3-refmod-export-preview { display:block; width:100%; max-height:180px; object-fit:contain; margin-top:8px; }
.ds-h3-refmod-export-button { padding:8px 16px; color:inherit; background:#344955; border:1px solid #708d9a; border-radius:7px; cursor:pointer; font:inherit; }
.ds-h3-refmod-export-button:hover:not(:disabled) { background:#446676; }
.ds-h3-refmod-export-dialog :focus-visible { outline:2px solid #a5d7e8; outline-offset:3px; }
.ds-h3-refmod-export-dialog :disabled { opacity:.55; cursor:default; }
.ds-h3-refmod-export-note { color:#b8c8d0; }
.ds-h3-refmod-export-status { white-space:pre-wrap; overflow-wrap:anywhere; }
@media(max-width:600px) { .ds-h3-refmod-export-settings,.ds-h3-refmod-export-row { grid-template-columns:1fr; } .ds-h3-refmod-export-dialog { padding:16px; } }
`);
  const header = el('header', 'header');
  const close = el('button', 'button', 'Close'); close.type = 'button';
  header.append(el('h2', 'title', 'Create RefMod'), close);
  dialog.append(header, el('p', 'note', 'Full Reference · independent, unweighted VAE references. No compression or LoRA training. Disabled and mode-hidden media are included when checked.'));
  const settingsPanel = el('div', 'settings');
  const field = (label, control, parent = settingsPanel, wide = false) => {
    const wrapper = el('label', wide ? 'field ds-h3-refmod-export-wide' : 'field');
    wrapper.append(el('span', 'label', label), control); parent.append(wrapper);
  };
  const library = el('select', 'concept'); library.setAttribute('aria-label', 'Saved RefMod library');
  const placeholder = el('option', 'option', 'Select a saved RefMod…'); placeholder.value = ''; library.append(placeholder);
  library.value = '';
  const libraryPanel = el('div', 'actions');
  field('Saved RefMod', library, libraryPanel); dialog.append(libraryPanel);
  const name = el('input', 'input'); name.type = 'text'; name.value = ''; name.placeholder = 'Automatic unique name';
  name.title = 'File/bundle name metadata, not a LoRA trigger. Leave empty for an automatic unique name.';
  field('Bundle name', name);
  const concept = el('select', 'concept');
  for (const value of CONCEPTS) { const option = el('option', 'option', value); option.value = value; concept.append(option); }
  concept.value = 'generic'; concept.title = 'Metadata describing the reference concept; does not train a model.';
  field('Concept type', concept);
  const description = el('textarea', 'description'); description.title = 'Description stored in this bundle; the Director prompt is unchanged.';
  field('Bundle description', description, settingsPanel, true);
  dialog.append(settingsPanel);
  const preview = (item, path = item.value, type = item.type) => {
    const media = el(type === 'image' ? 'img' : type === 'audio' || item.media_mode === 'audio' ? 'audio' : 'video', 'preview');
    media.src = viewUrl(path);
    if (type === 'image') { media.alt = `Reference: ${path}`; return media; }
    media.controls = true; media.preload = 'metadata';
    if (media.tagName.toLowerCase() === 'video') { media.muted = (item.media_mode ?? 'video') === 'video' || Boolean(item.audio); if (item.thumbnail) media.poster = item.thumbnail; }
    mediaElements.push(media);
    const start = Math.max(0, Number(item.trim_start) || 0);
    const end = () => item.trim_end == null ? media.duration : Number(item.trim_end);
    listen(media, 'loadedmetadata', () => { media.currentTime = start; });
    listen(media, 'play', () => { if (media.currentTime < start || media.currentTime >= end()) media.currentTime = start; });
    listen(media, 'seeking', () => { if (media.currentTime < start) media.currentTime = start; if (media.currentTime > end()) media.currentTime = end(); });
    listen(media, 'timeupdate', () => { if (media.currentTime >= end()) { media.pause(); if (media.currentTime > end()) media.currentTime = end(); } else if (media.currentTime < start) media.currentTime = start; });
    return media;
  };
  const references = el('div', 'references'); dialog.append(references);
  const renderRows = (memberDetails = []) => {
    for (const row of rows) row.element.remove();
    rows.length = 0;
    mediaElements.splice(0).forEach(media => { media.pause(); media.removeAttribute('src'); media.load(); });
    for (const kind of ['items', 'refmods']) {
      (snapshot[kind] || []).forEach((item, index) => {
        if (kind === 'items' ? !isMedia(item) : !item?.name) return;
        const members = kind === 'refmods' && Array.isArray(item.members) ? item.members : [null];
        for (const member of members) {
          const row = el('section', 'row'), source = el('div', 'source');
          const checked = el('input', 'checkbox'); checked.type = 'checkbox'; checked.checked = true;
          const label = el('label', 'label'); label.append(checked, el('span', 'filename', kind === 'items' ? `${item.type}: ${item.value}` : `Saved RefMod: ${item.name}${member ? ` · ${memberDetails[member.index]?.kind ?? 'latent'} member ${member.index + 1}` : ''}`)); source.append(label);
          if (kind === 'items') {
            const attachedAudio = item.type === 'video' && typeof item.audio === 'string' && item.audio;
            source.append(preview(item, attachedAudio && item.media_mode === 'audio' ? item.audio : item.value));
            if (item.type === 'video') {
              source.append(el('p', 'note', `Streams: ${item.media_mode ?? 'video'}${attachedAudio ? ' + attached audio' : ''}`));
              if (attachedAudio && item.media_mode !== 'audio') source.append(preview(item, item.audio, 'audio'));
            }
            if (item.type !== 'image') source.append(el('p', 'note', `Selected crop: ${item.trim_start ?? 0}s – ${item.trim_end ?? item.source_duration ?? 'file end'}s`));
          }
          if (kind === 'refmods') source.append(el('p', 'note', 'Reference data is stored as encoded latents and remains usable for generation. Original media files are not embedded, so this editor cannot display a preview.'));
          const text = el('textarea', 'description'); text.value = member?.description ?? item.description ?? (kind === 'items' ? item.prompt ?? '' : '');
          text.title = 'Description used for this export only; the source timeline is unchanged.';
          row.append(source); field('Reference description', text, row);
          references.append(row); rows.push({ kind, index, memberIndex: member?.index, checked, text, element: row });
        }
      });
    }
  };
  renderRows();
  const status = el('p', 'status', rows.length ? 'Select references and create a reusable bundle. Crops must be at most 15 seconds. Storage capacity does not increase generation limits.' : 'No references inserted. Add media or select a saved RefMod first.');
  status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
  const actions = el('footer', 'actions');
  const create = el('button', 'button', 'Create RefMod'); create.type = 'button'; create.disabled = !rows.length;
  actions.append(el('span', 'note', 'Sources and timeline remain unchanged.'), create); dialog.append(status, actions);
  const report = (message, error = false) => { if (!closed && !isRemoved()) { status.textContent = message; onStatus(message, error); } };
  const cleanup = () => {
    if (closed) return;
    closed = true; controller.abort(); listeners.splice(0).forEach(remove => remove());
    mediaElements.forEach(media => { media.pause(); media.removeAttribute('src'); media.load(); });
    dialog.close(); dialog.remove(); style.remove();
    if (priorFocus?.isConnected) priorFocus.focus();
    if (busy && !isRemoved()) onStatus('Export monitoring stopped. Any submitted job continues in the ComfyUI queue; use its queue controls to cancel that job.', false);
  };
  listen(close, 'click', cleanup);
  listen(dialog, 'close', cleanup);
  listen(dialog, 'cancel', event => { event.preventDefault(); cleanup(); });
  listen(dialog, 'keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); cleanup(); }
    if (event.key !== 'Tab') return;
    const focusable = [...dialog.querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),video[controls],audio[controls]')];
    const first = focusable[0], last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  const updateLibraryControls = () => {
    library.disabled = loading || busy || locked;
    create.disabled = loading || busy || locked || !rows.length;
  };
  const fetchLibrary = async url => {
    const response = await api.fetchApi(url, { signal: controller.signal });
    const body = await response.json();
    if (!response.ok) throw new Error(`Library request failed (${response.status}): ${JSON.stringify(body)}`);
    return body;
  };
  listen(library, 'change', async () => {
    if (loading || busy || locked || closed || isRemoved()) return;
    const selectedName = library.value;
    if (!selectedName) {
      snapshot = structuredClone(timelineSnapshot);
      name.value = ''; description.value = ''; concept.value = 'generic';
      create.textContent = 'Create RefMod'; create.title = '';
      renderRows(); updateLibraryControls();
      report('Create a new RefMod from the Director media lines. Sources and timeline remain unchanged.');
      return;
    }
    loading = true; updateLibraryControls(); report('Loading saved RefMod…');
    try {
      const detail = await fetchLibrary(`/dasiwa/refmods/detail?name=${encodeURIComponent(selectedName)}`);
      if (closed || isRemoved()) return;
      if (detail?.name !== selectedName || typeof detail.description !== 'string' || !CONCEPTS.includes(detail.concept_type) || !Array.isArray(detail.members) || !detail.members.length || detail.members.some(member => !member || typeof member.kind !== 'string' || typeof member.description !== 'string')) throw new Error('Invalid saved RefMod details.');
      snapshot = { items: [], refmods: [{ name: detail.name, members: detail.members.map((member, index) => ({ index, description: member.description })) }] };
      name.value = ''; description.value = detail.description; concept.value = detail.concept_type;
      renderRows(detail.members);
      create.textContent = 'Save Changes';
      create.title = 'Save edited references as a new bundle; the original file is never overwritten.';
      report('Stored reference data loaded and ready for generation. Edit members and save a new copy; the original stays unchanged.');
    } catch (error) {
      if (error.name !== 'AbortError') report(`Could not open saved RefMod: ${error.message}`, true);
    } finally {
      loading = false;
      if (!closed && !isRemoved()) updateLibraryControls();
    }
  });
  listen(create, 'click', async () => {
    if (loading || busy || locked || closed || isRemoved()) return;
    busy = true;
    const controls = [...dialog.querySelectorAll('input,select,textarea,button')].filter(control => control !== close);
    controls.forEach(control => { control.disabled = true; }); close.focus();
    try {
      const selected = selectDaSiWaRefModSnapshot(snapshot, rows.map(row => ({ kind: row.kind, index: row.index, memberIndex: row.memberIndex, selected: row.checked.checked, description: row.text.value })));
      const result = await queueDaSiWaRefModExport({ api, app, node, snapshot: selected, settings: { name: name.value, description: description.value, concept_type: concept.value }, signal: controller.signal, isRemoved, onStatus: report });
      if (closed || isRemoved()) return;
      locked = true;
      busy = false;
      report(`Saved ${result.name} (${result.members} members).\n${JSON.stringify(result.counts ?? {})}${result.warnings?.length ? '\nWarnings: ' + result.warnings.join('\n') : ''}${result.message ? '\n' + result.message : ''}`);
      try { await onSaved(result); } catch (error) { report(`Saved ${result.name}, but library refresh failed: ${error.message}`, true); }
    } catch (error) {
      if (error.queuedPromptId) locked = true;
      if (error.name !== 'AbortError') report(error.message, true);
    } finally {
      busy = false;
      if (!closed && !locked) {
        controls.forEach(control => { control.disabled = false; });
        updateLibraryControls();
      }
    }
  });
  document.head.append(style); document.body.append(dialog); dialog.showModal(); name.focus();
  loading = true; updateLibraryControls();
  (async () => {
    try {
      const entries = await fetchLibrary('/dasiwa/refmods');
      if (closed || isRemoved()) return;
      if (!Array.isArray(entries) || entries.some(entry => typeof entry?.name !== 'string' || !entry.name)) throw new Error('Invalid saved RefMod library.');
      for (const entry of entries) {
        const option = el('option', 'option', entry.name); option.value = entry.name; library.append(option);
      }
    } catch (error) {
      if (error.name !== 'AbortError') report(`Could not load saved RefMod library: ${error.message}`, true);
    } finally {
      loading = false;
      if (!closed && !isRemoved()) updateLibraryControls();
    }
  })();
  return cleanup;
}
