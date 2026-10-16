"""CPU-only native latent/store contracts; no checkpoint or GPU allocation."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import hashlib
import os
import pytest
sys.path.insert(0, os.environ.get('H3_COMFY',str(Path(__file__).resolve().parents[3] / 'ComfyUI')))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
from comfy.nested_tensor import NestedTensor
pkg = ModuleType('h3_segment_test')
pkg.__path__ = [str(Path(__file__).resolve().parents[2])]
sys.modules[pkg.__name__] = pkg
core = importlib.import_module(pkg.__name__ + '.nodes.h3_continuity.core')

def latent(frames, value=0.):
    return {'samples': NestedTensor((torch.full((1,24,(frames-5)//17*5+2,4,4),value), torch.full((1,32,2,round(frames*40/24)),value)))}

def root(store):
    t = store.stage(latent(124), dict(session='s', run_id='root', mode='REF2VA', resolved_prompt='root'))
    store.publish(t, 'root.mp4')
    return store.inspect('s','root')

def context():
    return dict(session='s', source_id='root', mode='REF2VA', resolved_prompt='new', extension_frames=119, overlap_frames=22, layout={'overlap_video_tokens':7, 'overlap_audio_tokens':37})

def test_range_reads_only_requested_parent_slices(tmp_path, monkeypatch):
    store = core.ClipStore(tmp_path)
    m = root(store)
    t = store.stage_delta(latent(141,9.),context(),m)
    store.publish(t,'delta.mp4')
    c = {**context(), 'source_id':t['clip_id'], 'layout':{'overlap_video_tokens':7,'overlap_audio_tokens':37}}
    t2 = store.stage_delta(latent(141,12.),c,store.inspect('s',t['clip_id']))
    store.publish(t2,'delta2.mp4')
    monkeypatch.setattr(core,'load_file',lambda *a,**k: pytest.fail('Full load forbidden'))
    tail = store.load_range('s',t2['clip_id'],70,107,400,603)
    v,a = tail['samples'].tensors
    assert torch.equal(v,torch.cat([torch.full((1,24,2,4,4),9.),torch.full((1,24,35,4,4),12.)],2))
    assert torch.equal(a,torch.cat([torch.full((1,32,2,5),9.),torch.full((1,32,2,198),12.)],-1))
    assert v.untyped_storage().nbytes() == v.numel()*v.element_size()
    with pytest.raises(ValueError): store.load_range('s',t2['clip_id'],-1,107,0,603)

@pytest.mark.parametrize('damage', ['missing','cycle','gap','canvas','shape','dtype','replace','escape'])
def test_delta_rejects_corrupt_ancestry(tmp_path,damage):
    store = core.ClipStore(tmp_path)
    m = root(store)
    t = store.stage_delta(latent(141,9.),context(),m)
    store.publish(t,'delta.mp4')
    path = store.member('s',t['clip_id'],'clip.json')
    import json
    d = json.loads(path.read_text())
    if damage == 'missing': store.member('s','root','latent.safetensors').unlink()
    elif damage == 'cycle': d['parent_id'] = t['clip_id']
    elif damage == 'gap': d['segment_start_frame'] += 17
    elif damage == 'canvas': d['width'] = 128
    elif damage in ('shape','dtype'):
        from safetensors.torch import save_file
        v,a = latent(124)['samples'].tensors
        save_file({'video':torch.zeros(1,24,35,4,4), 'audio':torch.zeros(1,32,2,197) if damage == 'shape' else torch.zeros(1,32,2,198,dtype=torch.float64)},str(store.member('s',t['clip_id'],'delta.safetensors')))
    elif damage == 'replace':
        from safetensors.torch import save_file
        v,a = latent(124,2.)['samples'].tensors
        save_file({'video':v,'audio':a},str(store.member('s','root','latent.safetensors')))
    elif damage == 'escape':
        pth = store.member('s',t['clip_id'],'delta.safetensors')
        outside = tmp_path.parent / (tmp_path.name + '-outside')
        pth.rename(outside)
        pth.symlink_to(outside)
    core.atomic_json(path,d)
    with pytest.raises((ValueError,OSError)): store.inspect('s',t['clip_id'])

def test_schema2_explicit_full_adapter_checks_budget(tmp_path,monkeypatch):
    store = core.ClipStore(tmp_path)
    m = root(store)
    t = store.stage_delta(latent(141,9.),context(),m)
    store.publish(t,'delta.mp4')
    full,meta = store.load('s',t['clip_id'])
    assert core.validate_h3_av_latent(full,name='full')[2] == 243
    import psutil
    monkeypatch.setattr(psutil,'virtual_memory',lambda: type('Mem',(),{'available':1})())
    with pytest.raises(ValueError,match='RAM'): store.load('s',t['clip_id'])

def test_output_mode_default_and_strict_values():
    assert core.parse_settings({})['output_mode'] == 'full'
    assert core.parse_settings({'output_mode':'segment'})['output_mode'] == 'segment'
    for value in (None,False,1,'wrong',[]):
        with pytest.raises(ValueError): core.parse_settings({'output_mode':value})

@pytest.mark.parametrize('source_frames',[124,141,158,243,515])
def test_guide_segment_reads_bounded_tail(tmp_path,monkeypatch,source_frames):
    store = core.ClipStore(tmp_path)
    t = store.stage(latent(source_frames,3.),dict(session='s',run_id='root',mode='REF2VA',resolved_prompt='root'))
    store.publish(t,'root.mp4')
    meta = store.inspect('s','root')
    settings = core.parse_settings({'version':3,'source_id':'root','session':'s','output_mode':'segment'},5)
    guide = dict(mode='REF2VA',width=64,height=64)
    updated,expected,layout = core.prepare_continuation(latent(source_frames,3.),meta,guide,settings)
    monkeypatch.setattr(store,'load',lambda *a: pytest.fail('Full source load forbidden'))
    updated2,target,layout2,tail = core.prepare_segment_continuation(store,meta,guide,settings)
    assert updated2 == updated and layout2 == layout
    assert all(torch.equal(a,b) for a,b in zip(expected['samples'].tensors,target['samples'].tensors))
    assert tail['samples'].tensors[0].shape[2] <= 47
    cond = core.add_tail([[torch.zeros(1),{}]],tail,target,layout2)
    key = cond[0][1]['minimax_keyframes'][0]
    assert key['latent'].shape[2] == layout['overlap_video_tokens']
    assert key['audio_latent'].shape[-1] == layout['overlap_audio_tokens']
    assert torch.all(key['latent'] == 3.)

def test_director_guide_segment_never_full_loads(tmp_path,monkeypatch):
    mod = importlib.import_module('h3_segment_test.nodes.nodes_minimax_h3_director_guide')
    store = core.ClipStore(tmp_path)
    root(store)
    monkeypatch.setattr(core,'ClipStore',lambda: store)
    monkeypatch.setattr(store,'load',lambda *a: pytest.fail('Full source load forbidden'))
    monkeypatch.setattr(mod.MiniMaxH3DirectorGuide,'_apply_native',lambda self,*a: ([[torch.zeros(1),{}]],None))
    guide = dict(mode='REF2VA',width=64,height=64,continuity={'version':3,'source_id':'root','session':'s','output_mode':'segment'})
    positive,target,ctx = mod.MiniMaxH3DirectorGuide().apply(None,None,guide)
    assert ctx['source_frames'] == 124 and ctx['output_mode'] == 'segment'
    assert ctx['source_id'] == 'root'
    assert target['samples'].tensors[0].shape[2] == 42
    assert 'source_file_stamp' in ctx
    from safetensors.torch import save_file
    v,a = latent(124,2.)['samples'].tensors
    save_file({'video':v,'audio':a},str(store.member('s','root','latent.safetensors')))
    segment = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    monkeypatch.setattr(segment,'ClipStore',lambda:store)
    with pytest.raises(ValueError,match='changed'): segment.DaSiWaH3ContinuitySegmentPrepare().prepare(target,ctx)

@pytest.mark.parametrize('overlap,extension',[(5,357),(22,119),(73,17)])
def test_segment_prepare_keeps_original_tail_not_sampled_overlap(tmp_path,monkeypatch,overlap,extension):
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    store = core.ClipStore(tmp_path)
    root(store)
    monkeypatch.setattr(mod,'ClipStore',lambda:store)
    ctx = {**context(),'output_mode':'segment','operation':'continue','overlap_frames':overlap,'extension_frames':extension}
    _,sampled,_ = core.prepare_continuation(latent(124),store.inspect('s','root'),dict(mode='REF2VA',width=64,height=64),core.parse_settings({**ctx,'version':2,'continuation_prompt':'new'}))
    for tensor in sampled['samples'].tensors: tensor.fill_(9.)
    render,segment,ticket = mod.DaSiWaH3ContinuitySegmentPrepare().prepare(sampled,ctx)
    v,a = render['samples'].tensors
    assert torch.all(v[:,:,:37] == 0.) and torch.all(v[:,:,37:] == 9.)
    assert torch.all(a[...,:207] == 0.)
    assert segment['render_kind'] == 'segment' and segment['source_frames'] == 124
    assert segment['layout']['overlap_video_tokens'] == (overlap-5)//17*5+2
    assert segment['layout']['overlap_audio_tokens'] == round(124*40/24)-round((124-overlap)*40/24)
    assert store.inspect('s',ticket['clip_id'],ready=False)['schema'] == 2
    assert ctx['source_id'] == 'root' and 'run_id' not in ctx
    _,_,ticket2 = mod.DaSiWaH3ContinuitySegmentPrepare().prepare(sampled,ctx)
    assert ticket2 != ticket

def test_segment_graph_requires_prepare_decode_context():
    from test_h3_continuity_validation import graph
    validation = importlib.import_module('h3_segment_test.nodes.h3_continuity.validation')
    p = graph()
    p['append']['class_type'] = 'DaSiWaH3ContinuitySegmentPrepare'
    p['publish']['inputs']['ticket'] = ['append',2]
    p['decode'] = {'class_type':'DaSiWaH3ContinuitySegmentDecode','inputs':{'latent':['append',0],'context':['append',1]}}
    p['export']['inputs']['audio'] = ['decode',1]
    validation.validate_capture_graph(p,'guide','segment')
    for change in ('context','latent','audio'):
        import copy
        bad = copy.deepcopy(p)
        if change == 'audio': bad['export']['inputs']['audio'] = ['other',0]
        else: bad['decode']['inputs'][change] = ['guide',2 if change == 'context' else 1]
        with pytest.raises(ValueError): validation.validate_capture_graph(bad,'guide','segment')
    with pytest.raises(ValueError,match='Segment Prepare'): validation.validate_capture_graph(graph(),'guide','segment')
    post = copy.deepcopy(p)  # spatial post-processing between decode and exporter keeps 24 fps segments
    post['wm'] = {'class_type':'DaSiWa_Watermark','inputs':{'images':['decode',0]}}
    post['export']['inputs']['images'] = ['wm',0]
    validation.validate_capture_graph(post,'guide','segment')
    post['wm']['inputs']['images'] = ['other',0]
    with pytest.raises(ValueError): validation.validate_capture_graph(post,'guide','segment')
    full = copy.deepcopy(p)
    validation.validate_capture_graph(full,'guide','full')  # one wiring serves both output modes
    full['publish']['inputs']['ticket'] = ['append',1]
    with pytest.raises(ValueError): validation.validate_capture_graph(full,'guide','full')

def test_segment_upscale_plan_local_prefix_and_audio(tmp_path,monkeypatch):
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    up = importlib.import_module('h3_segment_test.nodes.h3_upscale_continuity')
    store = core.ClipStore(tmp_path)
    store.publish(store.stage(latent(515,3.),dict(session='s',run_id='root',mode='REF2VA',resolved_prompt='root')),'root.mp4')
    monkeypatch.setattr(mod,'ClipStore',lambda:store)
    monkeypatch.setattr(up,'ClipStore',lambda:store)
    _,sampled,layout = core.prepare_continuation(latent(515,3.),store.inspect('s','root'),dict(mode='REF2VA',width=64,height=64),core.parse_settings({'version':3,'source_id':'root'},5))
    for tensor in sampled['samples'].tensors: tensor.fill_(9.)
    render,ctx,ticket = mod.DaSiWaH3ContinuitySegmentPrepare().prepare(sampled,{**context(),'layout':layout,'operation':'continue','output_mode':'segment'})
    v,a = render['samples'].tensors
    plan = up.continuity_plan(v,a,ctx)
    assert plan['source_tokens'] == (515-5)//17*5+2-ctx['render_plan']['video_start']
    assert plan['refine_start_token'] == plan['source_tokens']-7
    assert plan['frame_offset'] == 515-22-ctx['render_plan']['frame_offset']
    cond = [[torch.zeros(1),{'minimax_keyframes':[{'resolved_frame_index':0,'latent':torch.zeros(1,24,7,4,4),'audio_latent':torch.zeros(1,32,2,37)}]}]]
    aligned = up.align_continuity_conditioning(cond,v,a,plan)
    key = aligned[0][1]['minimax_keyframes'][0]
    assert key['resolved_frame_index'] == plan['frame_offset']
    assert torch.all(key['latent'] == 3.) and torch.all(key['audio_latent'] == 3.)
    assert cond[0][1]['minimax_keyframes'][0]['resolved_frame_index'] == 0
    with pytest.raises(ValueError,match='cumulative'): up.continuity_plan(v,a,{**ctx,'render_kind':'full'})

def test_legacy_append_rejects_segment_mode_before_full_load(monkeypatch):
    nodes = importlib.import_module('h3_segment_test.nodes.h3_continuity.nodes')
    monkeypatch.setattr(nodes,'ClipStore',lambda: pytest.fail('No heavy load before mode rejection'))
    with pytest.raises(ValueError,match='Segment Prepare'):
        nodes.DaSiWaH3ContinuityAppend().commit(latent(141),{**context(),'operation':'continue','output_mode':'segment'})

def test_delta_chain_survives_metadata_preserving_backup(tmp_path):
    import shutil
    store = core.ClipStore(tmp_path/'original')
    m = root(store)
    t = store.stage_delta(latent(141,9.),context(),m)
    store.publish(t,'delta.mp4')
    shutil.copytree(tmp_path/'original',tmp_path/'backup')
    backup = core.ClipStore(tmp_path/'backup')
    assert backup.inspect('s',t['clip_id'])['frames'] == 243

@pytest.mark.parametrize('overlap',[0,4,23,141])
def test_stage_delta_rejects_invalid_global_context(tmp_path,overlap):
    store = core.ClipStore(tmp_path)
    meta = root(store)
    with pytest.raises(ValueError): store.stage_delta(latent(260 if overlap == 141 else 141),{**context(),'overlap_frames':overlap},meta)
    assert len(list(store.session_dir('s').iterdir())) == 2

def test_stage_delta_rejects_non_floating_sampler_streams(tmp_path):
    store = core.ClipStore(tmp_path)
    meta = root(store)
    v,a = latent(141)['samples'].tensors
    with pytest.raises(ValueError,match='floating'):
        store.stage_delta({'samples':NestedTensor((v.to(torch.int64),a))},context(),meta)

def test_segment_guide_reuses_import_without_full_latent_load(tmp_path,monkeypatch):
    mod = importlib.import_module('h3_segment_test.nodes.nodes_minimax_h3_director_guide')
    video = importlib.import_module('h3_segment_test.nodes.h3_continuity.video_source')
    store = core.ClipStore(tmp_path)
    root(store)
    monkeypatch.setattr(core,'ClipStore',lambda:store)
    monkeypatch.setattr(video,'read_manifest',lambda *a:{'filename':'source.mp4','sha256':'abc'})
    monkeypatch.setattr(video,'input_video',lambda *a:tmp_path/'source.mp4')
    monkeypatch.setattr(video,'file_digest',lambda *a,**k:'abc')
    monkeypatch.setattr(video,'import_cache_id',lambda *a:'root')
    monkeypatch.setattr(video,'import_checkpoint',lambda *a:pytest.fail('Existing segment import must not full-load'))
    monkeypatch.setattr(mod,'_validate_h3_vaes',lambda *a,**k:None)
    monkeypatch.setattr(mod.MiniMaxH3DirectorGuide,'_apply_native',lambda self,*a:([[torch.zeros(1),{}]],None))
    guide = dict(mode='REF2VA',width=64,height=64,continuity={'version':3,'source_kind':'video','source_video_id':'upload','session':'s','output_mode':'segment'})
    _,target,ctx = mod.MiniMaxH3DirectorGuide().apply(None,None,guide)
    assert ctx['source_id'] == 'root' and ctx['source_frames'] == 124

def test_stage_delta_rejects_source_changed_after_guide(tmp_path):
    store = core.ClipStore(tmp_path)
    meta = root(store)
    seg = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_store')
    ctx = {**context(),'source_file_stamp':seg.file_stamp(store.member('s','root','latent.safetensors'))}
    from safetensors.torch import save_file
    v,a = latent(124,3.)['samples'].tensors
    save_file({'video':v,'audio':a},str(store.member('s','root','latent.safetensors')))
    with pytest.raises(ValueError,match='changed'): store.stage_delta(latent(141),ctx,meta)

@pytest.mark.parametrize('failure',['cancel','source_change','write'])
def test_stage_delta_failure_cleans_only_new_directory(tmp_path,monkeypatch,failure):
    store = core.ClipStore(tmp_path)
    meta = root(store)
    seg = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_store')
    import comfy.model_management as mm
    original = seg.save_file
    saved = []
    def save(tensors,path):
        if failure == 'write': raise OSError('disk full')
        original(tensors,path)
        saved.append(True)
        if failure == 'source_change':
            v,a = latent(124,3.)['samples'].tensors
            original({'video':v,'audio':a},str(store.member('s','root','latent.safetensors')))
    def check():
        if failure == 'cancel' and saved: raise RuntimeError('interrupted')
    monkeypatch.setattr(seg,'save_file',save)
    monkeypatch.setattr(mm,'throw_exception_if_processing_interrupted',check)
    with pytest.raises((ValueError,OSError,RuntimeError)): store.stage_delta(latent(141),context(),meta)
    assert len(list(store.session_dir('s').iterdir())) == 2
    assert store.inspect('s','root')['status'] == 'ready'

def test_stage_delta_writes_only_new_tokens(tmp_path):
    store = core.ClipStore(tmp_path)
    metadata = root(store)
    source = store.member('s','root','latent.safetensors')
    before = hashlib.sha256(source.read_bytes()).digest()
    sampled = latent(141, 9.)
    t = store.stage_delta(sampled, context(), metadata)
    m = store.inspect('s',t['clip_id'],ready=False)
    assert m['schema'] == 2 and m['storage'] == 'delta' and m['frames'] == 243
    from safetensors.torch import load_file
    data = load_file(str(store.member('s',t['clip_id'],'delta.safetensors')))
    assert data['video'].shape == (1,24,35,4,4)
    assert data['audio'].shape == (1,32,2,198)
    assert torch.all(data['video'] == 9) and torch.all(data['audio'] == 9)
    assert hashlib.sha256(source.read_bytes()).digest() == before
