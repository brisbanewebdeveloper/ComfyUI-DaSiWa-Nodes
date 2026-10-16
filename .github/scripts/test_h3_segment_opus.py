"""Real shared-writer regression: global H3 boundaries survive Opus resampling."""
from test_h3_segment_backend import core, latent, context, root, importlib, torch
from fractions import Fraction
from types import SimpleNamespace
import sys
import av
import numpy as np
import pytest

helper = importlib.import_module('h3_segment_test.nodes.helper_pyav_video')


@pytest.fixture
def combine(monkeypatch):
    monkeypatch.setitem(sys.modules, 'server', SimpleNamespace(PromptServer=None))
    return importlib.import_module('h3_segment_test.nodes.nodes_enhanced_video_combine')


def audio_value(source, frames):
    count = round(Fraction((source+frames)*32000,24))-round(Fraction(source*32000,24))
    wave = torch.arange(count).float().mul_(0.01).sin_().mul_(0.1).expand(1,2,-1).clone()
    return dict(waveform=wave, sample_rate=32000,
                h3_segment_interval=dict(start=(source,24),end=(source+frames,24)))


def encode(path, container, frames, audio, audio_encoder='libopus'):
    helper.encode_attempt(str(path),container,'libvpx-vp9' if container == 'WebM' else 'libx264',
                          16,16,24,8,35,
                          lambda: (np.full((16,16,3),i%255,np.uint8) for i in range(frames)),
                          audio=audio,audio_encoder=audio_encoder)


def decoded_audio(path):
    with av.open(str(path)) as media:
        rate = media.streams.audio[0].codec_context.sample_rate
        count = sum(f.samples for f in media.decode(audio=0))
    return rate,count


@pytest.mark.parametrize('container',['MKV','WebM'])
@pytest.mark.parametrize('source',[5,124,243])
@pytest.mark.parametrize('frames',[17,119,187,238])
def test_real_opus_exact_global_interval(tmp_path,combine,container,source,frames):
    value = audio_value(source,frames)
    audio,_ = combine._audio_file(value)
    assert audio[2] == value['h3_segment_interval']
    path = tmp_path / ('segment.webm' if container == 'WebM' else 'segment.mkv')
    encode(path,container,frames,audio)
    assert decoded_audio(path) == (48000,frames*2000)


@pytest.mark.parametrize('container,frames',[('MKV',187),('WebM',238),('MKV',119)])
def test_publish_accepts_real_shared_writer(tmp_path,monkeypatch,combine,container,frames):
    nodes = importlib.import_module('h3_segment_test.nodes.h3_continuity.nodes')
    store = core.ClipStore(tmp_path/'store')
    metadata = root(store)
    ctx = {**context(),'extension_frames':frames}
    ticket = store.stage_delta(latent(frames+22,9.),ctx,metadata)
    monkeypatch.setattr(nodes,'ClipStore',lambda:store)
    import folder_paths
    monkeypatch.setattr(folder_paths,'get_output_directory',lambda:str(tmp_path))
    path = tmp_path / ('segment.webm' if container == 'WebM' else 'segment.mkv')
    audio,_ = combine._audio_file(audio_value(124,frames))
    encode(path,container,frames,audio)
    nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
    assert store.inspect('s',ticket['clip_id'])['export']['audio']['samples'] == frames*2000


@pytest.mark.parametrize('interval',[None,{}, {'start':(0,0),'end':(17,24)},
    {'start':(True,24),'end':(17,24)}, {'start':(17,24),'end':(0,24)},
    {'start':(0.0,24),'end':(17,24)}, {'start':(-1,24),'end':(17,24)}])
def test_bad_contract_rejected(combine,interval):
    value = audio_value(5,17)
    value['h3_segment_interval'] = interval
    with pytest.raises(ValueError,match='H3 segment'): combine._audio_file(value)


@pytest.mark.parametrize('difference',[-1000,1,1000])
def test_changed_waveform_rejected(tmp_path,combine,difference):
    value = audio_value(5,17)
    count = value['waveform'].shape[-1]+difference
    value['waveform'] = torch.zeros(1,2,count)
    with pytest.raises(ValueError,match='source sample interval'): combine._audio_file(value)
    with pytest.raises(ValueError,match='source sample interval'):
        encode(tmp_path/'bad.mkv','MKV',17,(np.zeros((2,count),np.float32),32000,value['h3_segment_interval']))
    assert not (tmp_path/'bad.mkv').exists()
    assert not list(tmp_path.glob('*.attempt-*'))


def test_large_resampler_mismatch_rejected(tmp_path,combine,monkeypatch):
    original = helper.av.AudioResampler
    class BrokenResampler:
        def __init__(self,**kwargs): self.real = original(**kwargs)
        def resample(self,frame): return [] if frame is None else self.real.resample(frame)
    monkeypatch.setattr(helper.av,'AudioResampler',BrokenResampler)
    audio,_ = combine._audio_file(audio_value(5,187))
    with pytest.raises(ValueError,match='meaningfully'):
        encode(tmp_path/'bad.mkv','MKV',187,audio)
    assert not (tmp_path/'bad.mkv').exists()


def test_normal_audio_is_not_padded_to_video(tmp_path,combine):
    value = dict(waveform=torch.zeros(1,2,16000),sample_rate=32000)
    audio,_ = combine._audio_file(value)
    assert len(audio) == 2
    encode(tmp_path/'normal.mkv','MKV',119,audio)
    assert decoded_audio(tmp_path/'normal.mkv') == (48000,24000)
    encode(tmp_path/'pcm.mkv','MKV',119,audio,'pcm_s16le')
    assert decoded_audio(tmp_path/'pcm.mkv') == (32000,16000)
    assert helper._audio_rate('aac',32000) == 32000
    assert helper._audio_rate('libopus',24000) == 24000


@pytest.mark.parametrize('audio_encoder',['aac','pcm_s16le'])
def test_contract_does_not_change_native_rate_codecs(tmp_path,combine,audio_encoder):
    audio,_ = combine._audio_file(audio_value(124,119))
    encode(tmp_path/'contract.mkv','MKV',119,audio,audio_encoder)
    encode(tmp_path/'plain.mkv','MKV',119,audio[:2],audio_encoder)
    assert decoded_audio(tmp_path/'contract.mkv') == decoded_audio(tmp_path/'plain.mkv')
    assert decoded_audio(tmp_path/'contract.mkv')[0] == 32000


def test_contract_rejects_multiple_batches(combine):
    value = audio_value(5,17)
    value['waveform'] = value['waveform'].expand(2,-1,-1)
    with pytest.raises(ValueError,match='source sample interval'): combine._audio_file(value)


@pytest.mark.parametrize('frames',[22,39])
def test_first_take_interval_and_serialized_rationals(tmp_path,combine,frames):
    value = audio_value(0,frames)
    value['h3_segment_interval'] = {'start':[0,24],'end':[frames,24]}
    audio,_ = combine._audio_file(value)
    encode(tmp_path/'first.mkv','MKV',frames,audio)
    assert decoded_audio(tmp_path/'first.mkv') == (48000,frames*2000)


def test_combine_forwards_contract_to_shared_writer(tmp_path,combine,monkeypatch):
    audio,_ = combine._audio_file(audio_value(5,17))
    monkeypatch.setattr(combine,'_available_encoders',lambda:{'libx264','libopus'})
    path = tmp_path/'forwarded.mkv'
    combine._encode_with_available_encoder(None,'H.264',8,16,16,24,
        lambda: (np.zeros((16,16,3),np.uint8) for _ in range(17)),
        str(path),'MKV',35,35,None,audio=audio,audio_codec='Opus')
    assert decoded_audio(path) == (48000,34000)
