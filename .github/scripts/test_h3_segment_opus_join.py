"""Real Opus exports: Publish a chain, then Join Clips in both output modes."""
from test_h3_segment_opus import audio_value, encode, combine, core, root, context, latent, importlib
from pathlib import Path
import av
import pytest


@pytest.mark.parametrize('container', ['MKV', 'WebM'])
@pytest.mark.parametrize('copy_video', [False, True])
def test_published_opus_chain_joins_sample_exactly(tmp_path, monkeypatch, combine, container, copy_video):
    nodes = importlib.import_module('h3_segment_test.nodes.h3_continuity.nodes')
    assembly = importlib.import_module('h3_segment_test.nodes.h3_continuity.assembly')
    store = core.ClipStore(tmp_path/'store')
    metadata = root(store)
    suffix = '.webm' if container == 'WebM' else '.mkv'
    prefix = tmp_path/('root'+suffix)
    audio, _ = combine._audio_file(audio_value(0,124))
    encode(prefix, container, 124, audio)
    store.publish(dict(session='s',clip_id='root'),str(prefix))
    monkeypatch.setattr(nodes, 'ClipStore', lambda:store)
    import folder_paths
    monkeypatch.setattr(folder_paths,'get_output_directory',lambda:str(tmp_path))
    selected = [dict(kind='checkpoint',session='s',clip_id='root')]
    for index, frames in enumerate((187,238)):
        metadata = store.inspect('s', selected[-1]['clip_id'])
        ctx = {**context(), 'source_id':metadata['clip_id'], 'extension_frames':frames}
        ticket = store.stage_delta(latent(frames+22,9.),ctx,metadata)
        audio, _ = combine._audio_file(audio_value(metadata['frames'],frames))
        path = tmp_path/(f'segment{index}'+suffix)
        encode(path,container,frames,audio)
        nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
        assert store.inspect('s',ticket['clip_id'])['export']['audio']['samples'] == frames*2000
        selected.append(dict(kind='checkpoint',**ticket))
    data = dict(version=1,rows=selected,repair_transitions=False,audio_policy='preserve',copy_video=copy_video)
    preflight = assembly.preflight(data,store,input_root=tmp_path,output_root=tmp_path)
    assert preflight['ok'], preflight
    assert preflight['frames'] == 549
    if copy_video:
        assert preflight['mode'] == 'copy', preflight
    result = assembly.assemble(data,'joined',store,input_root=tmp_path,output_root=tmp_path,check=lambda:None)
    if copy_video and result['mode'] != 'copy':
        assert result['mode'] == 'reencode' and any('copy failed' in note for note in result['notes']), result
    else:
        assert result['mode'] == ('copy' if copy_video else 'reencode'), result
    output = Path(result['filename'])
    with av.open(str(output)) as media:
        assert sum(1 for _ in media.decode(video=0)) == 549
    with av.open(str(output)) as media:
        stream = media.streams.audio[0]
        assert stream.codec_context.sample_rate == 48000
        assert stream.duration*stream.time_base*48000 == 549*2000
        frames = list(media.decode(audio=0))
        assert frames[0].pts == 0
        assert 549*2000 <= sum(f.samples for f in frames) <= 549*2000+1024
    assert not list(tmp_path.glob('*.partial*'))
