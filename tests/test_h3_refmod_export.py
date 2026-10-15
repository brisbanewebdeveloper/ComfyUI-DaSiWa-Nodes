"""CPU contracts for queued Director archive creation; no model weights."""
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'ComfyUI'))
import comfy.cli_args
comfy.cli_args.args.cpu = True
package = types.ModuleType('dasiwa_export_test_nodes')
package.__path__ = [str(ROOT / 'nodes')]
sys.modules[package.__name__] = package


def exporter():
    return importlib.import_module(package.__name__ + '.refmod_export')


def test_bundle_roundtrip(tmp_path, monkeypatch):
    m = exporter()
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    image = torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2)
    audio = torch.ones(1, 32, 2, 7)
    name = m.dasiwa_save_bundle([(image, {'kind': 'image'}), (audio, {'kind': 'audio'})], [],
                                name='hero', description='test', concept_type='identity')
    assert name == 'hero'
    with safe_open(tmp_path / 'hero.safetensors', framework='pt') as f:
        meta = json.loads(f.metadata()['refmod_meta'])
        assert meta['_format_version'] == 5 and meta['kind'] == 'bundle'
        assert meta['description'] == 'test' and meta['concept_type'] == 'identity'
        assert [v['kind'] for v in meta['members']] == ['image', 'audio']
        assert set(f.keys()) == {'ref_0', 'ref_1'}
    fmt = importlib.import_module(package.__name__ + '.helper_refmod_format')
    monkeypatch.setattr(fmt, 'refmods_roots', lambda: [str(tmp_path)])
    loaded = fmt.load_refmods(name)
    assert torch.equal(loaded[0][0], image) and torch.equal(loaded[1][0], audio)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/b', 'a\\b', '%2fescape',
                                   '%252fescape', 'CON', 'CON .txt', 'nul.txt', 'COM1', 'x:', 'x\n', 'x\x85y', '.', '..', 'x.', 'x '])
def test_writer_rejects_unsafe_names(tmp_path, monkeypatch, name):
    m = exporter()
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    monkeypatch.setattr(m.os, 'link', lambda *args: None)
    with pytest.raises(ValueError, match='name'):
        m.dasiwa_save_bundle([(torch.zeros(1, 24, 1, 2, 2), {'kind': 'image'})], [], name=name)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('refs', [[], [(torch.zeros(1, 4, 1, 2, 2), {'kind': 'image'})],
    [(torch.full((1, 24, 1, 2, 2), float('nan')), {'kind': 'image'})],
    [(torch.zeros(1, 24, 1, 2, 2), {'kind': 'invalid'})]])
def test_writer_rejects_invalid_members(tmp_path, monkeypatch, refs):
    m = exporter()
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    with pytest.raises(ValueError):
        m.dasiwa_save_bundle(refs, [])
    assert not list(tmp_path.iterdir())


def test_collision_and_disk_failure_leave_original(tmp_path, monkeypatch):
    m = exporter()
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    refs = [(torch.zeros(1, 24, 1, 2, 2), {'kind': 'image'})]
    m.dasiwa_save_bundle(refs, [], name='keep')
    original = (tmp_path / 'keep.safetensors').read_bytes()
    with pytest.raises(FileExistsError):
        m.dasiwa_save_bundle(refs, [], name='keep')
    assert (tmp_path / 'keep.safetensors').read_bytes() == original
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(m, 'save_file', fail)
    with pytest.raises(OSError, match='disk full'):
        m.dasiwa_save_bundle(refs, [])
    assert [p.name for p in tmp_path.iterdir()] == ['keep.safetensors']


def test_cancellation_after_serialization_does_not_publish(tmp_path, monkeypatch):
    m = exporter()
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    save = m.save_file
    def serialize(*args, **kwargs):
        save(*args, **kwargs)
        def cancel():
            raise RuntimeError('interrupted')
        monkeypatch.setattr(m, 'dasiwa_interrupted', cancel)
    monkeypatch.setattr(m, 'save_file', serialize)
    with pytest.raises(RuntimeError, match='interrupted'):
        m.dasiwa_save_bundle([(torch.zeros(1, 24, 1, 2, 2), {'kind': 'image'})], [])
    assert not list(tmp_path.iterdir())


def test_output_node_flattens_disabled_saved_bundle(tmp_path, monkeypatch):
    m = exporter()
    fmt = importlib.import_module(package.__name__ + '.helper_refmod_format')
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    monkeypatch.setattr(fmt, 'refmods_roots', lambda: [str(tmp_path)])
    refs = [(torch.full((1, 24, 1, 2, 2), 3.), {'kind': 'image', 'description': 'original'}),
            (torch.ones(1, 32, 2, 7), {'kind': 'audio'})]
    m.dasiwa_save_bundle(refs, [], name='source')
    node = m.DaSiWaH3RefModExport()
    schema = node.INPUT_TYPES()
    assert list(schema['required']) == ['timeline_data', 'request_id', 'name', 'description', 'concept_type']
    assert schema['required']['concept_type'][0] == ['generic', 'identity', 'style', 'motion', 'audio']
    assert node.FUNCTION == 'create_refmod' and node.OUTPUT_NODE
    assert node.RETURN_NAMES == ('refmod_name',)
    result = node.create_refmod(json.dumps({'refmods': [{'name': 'source', 'enabled': False, 'strength': 0}]}),
                                'job', name='output', description='bundle', concept_type='style')
    ui = result['ui']['dasiwa_refmod_export'][0]
    assert result['result'] == ('output',)
    assert ui['counts'] == {'image': 1, 'video': 0, 'audio': 1} and ui['members'] == 2
    loaded = fmt.load_refmods('output')
    assert torch.equal(loaded[0][0], refs[0][0])
    assert loaded[0][1]['description'] == 'original'
    assert loaded[0][1]['source_refmod'] == 'source'


@pytest.mark.parametrize('description', ['Edited export note', ''])
def test_saved_reference_description_override_is_export_only(tmp_path, monkeypatch, description):
    m = exporter()
    fmt = importlib.import_module(package.__name__ + '.helper_refmod_format')
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    monkeypatch.setattr(fmt, 'refmods_roots', lambda: [str(tmp_path)])
    latent = torch.full((1, 24, 1, 2, 2), 3.)
    m.dasiwa_save_bundle([(latent, {'kind': 'image', 'description': 'Original member note'})], [], name='source')
    m.DaSiWaH3RefModExport().create_refmod(
        json.dumps({'refmods': [{'name': 'source', 'description': description, 'strength': 0}]}),
        'job', name='edited')
    assert fmt.load_refmods('edited')[0][1]['description'] == description
    assert fmt.load_refmods('source')[0][1]['description'] == 'Original member note'
    assert torch.equal(fmt.load_refmods('edited')[0][0], latent)


@pytest.mark.parametrize('state', [[], {'items': 'bad'}, {'refmods': 'bad'}, {},
                                  {'items': [{'type': 'image', 'value': ''}]},
                                  {'items': [{'type': 'url', 'value': 'x'}]}])
def test_invalid_or_empty_snapshot(state):
    with pytest.raises(ValueError):
        exporter().dasiwa_export_refs(json.dumps(state))


class VisualVAE:
    first_stage_model = type('MiniMaxH3VideoVAE', (), {})()

    def __init__(self):
        self.calls = []

    def encode(self, pixels):
        self.calls.append(pixels.shape)
        t = 1 if pixels.shape[0] == 1 else 2 + (pixels.shape[0] - 5) // 17 * 5
        return torch.full((1, 24, t, pixels.shape[1] // 16, pixels.shape[2] // 16), .25)


def test_disabled_image_uses_real_core_and_roundtrips(tmp_path, monkeypatch):
    from PIL import Image
    m = exporter()
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(tmp_path))
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    Image.new('RGB', (64, 32)).save(tmp_path / 'a.png')
    vae = VisualVAE()
    snapshot = {'items': [{'id': 'hidden', 'type': 'image', 'value': 'a.png', 'enabled': False}]}
    before = json.dumps(snapshot)
    result = m.DaSiWaH3RefModExport().create_refmod(before, 'job', vae=vae)
    assert len(vae.calls) == 1 and vae.calls[0] == (1, 32, 64, 3)
    name = result['result'][0]
    assert name.startswith('director-')
    fmt = importlib.import_module(package.__name__ + '.helper_refmod_format')
    monkeypatch.setattr(fmt, 'refmods_roots', lambda: [str(tmp_path)])
    latent, meta = fmt.load_refmods(name)[0]
    assert latent.shape == (1, 24, 1, 2, 4) and torch.all(latent == .25)
    assert meta['source_item_id'] == 'hidden' and meta['source_enabled'] is False
    assert json.dumps(snapshot) == before


def test_symlink_escape_is_rejected_before_vae(tmp_path, monkeypatch):
    m = exporter()
    inside = tmp_path / 'input'
    inside.mkdir()
    (tmp_path / 'outside.png').write_bytes(b'not decoded')
    (inside / 'link.png').symlink_to(tmp_path / 'outside.png')
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(inside))
    with pytest.raises(ValueError, match='escapes'):
        m.dasiwa_export_refs(json.dumps({'items': [{'type': 'image', 'value': 'link.png'}]}))


class AudioVAE:
    first_stage_model = type('MiniMaxH3AudioVAE', (), {})()
    audio_sample_rate = 32000

    def __init__(self):
        self.calls = []

    def encode(self, samples):
        self.calls.append(samples.clone())
        return torch.ones(1, 32, 2, max(1, samples.shape[1] // 800))


@pytest.fixture
def media_files(tmp_path):
    import av
    import numpy as np
    import wave
    samples = (np.sin(np.arange(64000) * .05) * 12000).astype(np.int16)
    with wave.open(str(tmp_path / 'sound.wav'), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(32000)
        f.writeframes(samples.tobytes())
    with av.open(str(tmp_path / 'clip.mkv'), 'w') as c:
        video = c.add_stream('ffv1', rate=24)
        video.width = video.height = 32
        video.pix_fmt = 'yuv420p'
        audio = c.add_stream('pcm_s16le', rate=32000)
        audio.layout = 'mono'
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i, dtype=np.uint8), format='rgb24')
            for packet in video.encode(frame):
                c.mux(packet)
        for packet in video.encode():
            c.mux(packet)
        frame = av.AudioFrame.from_ndarray(samples[None], format='s16', layout='mono')
        frame.sample_rate = 32000
        for packet in audio.encode(frame):
            c.mux(packet)
        for packet in audio.encode():
            c.mux(packet)
    return tmp_path


@pytest.mark.parametrize('mode,attached,expected', [('video', False, ['video']),
    ('audio', False, ['audio']), ('video_audio', False, ['video', 'audio']),
    ('video', True, ['video', 'audio']), ('video_audio', True, ['video', 'audio'])])
def test_video_streams_real_decode_core_crop(media_files, monkeypatch, mode, attached, expected):
    m = exporter()
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    visual, audio = VisualVAE(), AudioVAE()
    item = {'id': 'pair', 'type': 'video', 'value': 'clip.mkv', 'media_mode': mode,
            'trim_start': .5, 'trim_end': 1.5, 'enabled': False}
    if attached:
        item['audio'] = 'sound.wav'
    refs, warnings = m.dasiwa_export_refs(json.dumps({'items': [item]}), visual, audio)
    assert [meta['kind'] for _, meta in refs] == expected
    assert len(visual.calls) == ('video' in expected) and len(audio.calls) == ('audio' in expected)
    assert all(meta['source_item_id'] == 'pair' and meta['trim_start'] == .5 and meta['trim_end'] == 1.5 for _, meta in refs)
    assert bool(warnings) == ('video' in expected)
    if audio.calls:
        assert audio.calls[0].shape == (1, 32000, 1)


def test_standalone_audio_real_decode_core(media_files, monkeypatch):
    m = exporter()
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    vae = AudioVAE()
    refs, warnings = m.dasiwa_export_refs(json.dumps({'items': [
        {'type': 'audio', 'value': 'sound.wav', 'trim_start': .5, 'trim_end': 1.5}]}), audio_vae=vae)
    assert [meta['kind'] for _, meta in refs] == ['audio'] and not warnings
    assert vae.calls[0].shape == (1, 32000, 1)


@pytest.mark.parametrize('stream_kind', ['video', 'audio'])
def test_offset_container_crops_use_relative_presentation_time(tmp_path, stream_kind):
    import av
    import numpy as np
    from fractions import Fraction
    path = tmp_path / 'offset.mkv'
    with av.open(str(path), 'w') as container:
        video = container.add_stream('ffv1', rate=24)
        video.width = video.height = 32
        video.pix_fmt = 'yuv420p'
        audio = container.add_stream('pcm_s16le', rate=32000)
        audio.layout = 'mono'
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i + 20, dtype=np.uint8), format='rgb24')
            frame.pts = 240 + i
            frame.time_base = Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = np.full((1, 64000), 2000, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format='s16', layout='mono')
        frame.sample_rate = 32000
        frame.pts = 320000
        frame.time_base = Fraction(1, 32000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    with pytest.raises(ValueError, match='crop'):
        exporter().dasiwa_media_crop({'trim_start': 0, 'trim_end': 3}, str(path), stream_kind)
    result = exporter().dasiwa_decode_media(str(path), stream_kind, 0, 1)
    if stream_kind == 'video':
        assert result.shape[0] == 24
        assert abs(float(result[0].mean()) - 20 / 255) < .02
    else:
        assert result['waveform'].shape[-1] == 32000


def test_decoder_seeks_and_checks_deadline_even_for_discarded_frames(monkeypatch):
    from contextlib import contextmanager
    from fractions import Fraction
    m = exporter()
    media = importlib.import_module(package.__name__ + '.h3_continuity.pyav_media')
    stream = types.SimpleNamespace(time_base=Fraction(1, 1000), codec_context=types.SimpleNamespace(thread_count=0))
    seeks, checks = [], []
    container = types.SimpleNamespace(start_time=10000000, streams=types.SimpleNamespace(video=[stream]),
                                      seek=lambda *args, **kwargs: seeks.append((args, kwargs)))
    @contextmanager
    def opened(path):
        yield container
    def guard(timeout, interrupt=False):
        assert timeout <= 60
        def check():
            checks.append(True)
            if len(checks) > 1:
                raise ValueError('Media processing timed out')
        return check
    def decoded(container, stream, check):
        check()
        assert checks, 'Deadline guard was not used for discarded frames'
        yield None
    monkeypatch.setattr(media, 'open_media', opened)
    monkeypatch.setattr(media, 'guard', guard)
    monkeypatch.setattr(media, 'decoded', decoded)
    with pytest.raises(ValueError, match='timed out'):
        m.dasiwa_decode_media('not-opened', 'video', 500, 501)
    assert seeks == [((510000,), {'stream': stream, 'backward': True})]
    assert checks


@pytest.mark.parametrize('start,end', [(0, float('nan')), (3, 2), (-1, 2), (0, 16), (0, 3)])
def test_invalid_crop_before_decode(media_files, monkeypatch, start, end):
    m = exporter()
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    vae = VisualVAE()
    with pytest.raises(ValueError, match='crop'):
        m.dasiwa_export_refs(json.dumps({'items': [{'type': 'video', 'value': 'clip.mkv',
            'trim_start': start, 'trim_end': end}]}), vae)
    assert not vae.calls


def test_preflight_rejects_large_source_without_decode(monkeypatch):
    from contextlib import contextmanager
    m = exporter()
    media = importlib.import_module(package.__name__ + '.h3_continuity.pyav_media')
    stream = types.SimpleNamespace(duration=20, time_base=1, width=3840, height=2160,
                                   average_rate=120, guessed_rate=120)
    @contextmanager
    def opened(path):
        yield types.SimpleNamespace(streams=types.SimpleNamespace(video=[stream]), duration=20000000)
    monkeypatch.setattr(media, 'open_media', opened)
    with pytest.raises(ValueError, match='decode estimate'):
        m.dasiwa_media_crop({'trim_start': 0, 'trim_end': 2}, 'header only', 'video')


def test_partial_failure_does_not_publish(media_files, monkeypatch):
    from PIL import Image
    m = exporter()
    storage = media_files / 'refmods'
    storage.mkdir()
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(storage)])
    Image.new('RGB', (32, 32)).save(media_files / 'first.png')
    vae = VisualVAE()
    with pytest.raises(ValueError, match='does not exist'):
        m.DaSiWaH3RefModExport().create_refmod(json.dumps({'items': [
            {'type': 'image', 'value': 'first.png'}, {'type': 'image', 'value': 'missing.png'}]}), 'job', vae=vae)
    assert len(vae.calls) == 1 and not list(storage.iterdir())


def test_cancellation_inside_decode(media_files, monkeypatch):
    m = exporter()
    def cancel():
        raise RuntimeError('interrupted')
    monkeypatch.setattr(m, 'dasiwa_interrupted', cancel)
    with pytest.raises(RuntimeError, match='interrupted'):
        m.dasiwa_decode_media(str(media_files / 'clip.mkv'), 'video', 0, 1)


def test_grid_metadata_and_attached_source(media_files, monkeypatch):
    m = exporter()
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    refs, _ = m.dasiwa_export_refs(json.dumps({'items': [{
        'type': 'video', 'value': 'clip.mkv', 'audio': 'sound.wav', 'media_mode': 'video_audio',
        'trim_start': .5, 'trim_end': 1.5, 'forge_role': 'identity'}]}), VisualVAE(), AudioVAE())
    meta = refs[0][1]
    assert meta['source_frame_count'] == 24 and meta['encoded_frame_count'] == 22
    assert meta['source_frame_rate'] == 24 and meta['source_role'] == 'identity'
    assert refs[1][1]['source_audio'] == 'sound.wav'


def test_decode_limits_bad_metadata_frame_count(media_files, monkeypatch):
    m = exporter()
    media = importlib.import_module(package.__name__ + '.h3_continuity.pyav_media')
    import av
    import numpy as np
    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')
    from fractions import Fraction
    frame.pts = 0
    frame.time_base = Fraction(1, 24)
    monkeypatch.setattr(media, 'decoded', lambda *args: (frame for _ in range(4000)))
    with pytest.raises(ValueError, match='decode'):
        m.dasiwa_decode_media(str(media_files / 'clip.mkv'), 'video', 0, 1)


def test_audio_decode_limits_repeated_timestamps(media_files, monkeypatch):
    m = exporter()
    media = importlib.import_module(package.__name__ + '.h3_continuity.pyav_media')
    import av
    import numpy as np
    from fractions import Fraction
    frame = av.AudioFrame.from_ndarray(np.ones((1, 32000), dtype=np.int16), format='s16', layout='mono')
    frame.sample_rate = 32000
    frame.pts = 0
    frame.time_base = Fraction(1, 32000)
    monkeypatch.setattr(media, 'decoded', lambda *args: (frame for _ in range(20)))
    with pytest.raises(ValueError, match='decode'):
        m.dasiwa_decode_media(str(media_files / 'sound.wav'), 'audio', 0, 1)


@pytest.mark.parametrize('kind,kwargs,message', [('image', {}, 'video VAE'), ('audio', {}, 'audio VAE'),
                                                ('video', {'media_mode': 'invalid'}, 'stream choice')])
def test_missing_vae_or_invalid_stream(media_files, monkeypatch, kind, kwargs, message):
    from PIL import Image
    m = exporter()
    Image.new('RGB', (32, 32)).save(media_files / 'image.png')
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    name = {'image': 'image.png', 'audio': 'sound.wav', 'video': 'clip.mkv'}[kind]
    with pytest.raises(ValueError, match=message):
        m.dasiwa_export_refs(json.dumps({'items': [{'type': kind, 'value': name, **kwargs}]}))


def test_silent_audio_rejected_before_encode(media_files, monkeypatch):
    import wave
    m = exporter()
    with wave.open(str(media_files / 'silence.wav'), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(32000)
        f.writeframes(bytes(64000))
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    vae = AudioVAE()
    with pytest.raises(ValueError, match='silent'):
        m.dasiwa_export_refs(json.dumps({'items': [{'type': 'audio', 'value': 'silence.wav'}]}), audio_vae=vae)
    assert not vae.calls


def test_member_limit_and_first_storage_root(tmp_path, monkeypatch):
    m = exporter()
    first, second = tmp_path / 'first', tmp_path / 'second'
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(first), str(second)])
    refs = [(torch.zeros(1, 24, 1, 2, 2), {'kind': 'image'})]
    name = m.dasiwa_save_bundle(refs, [])
    assert (first / (name + '.safetensors')).is_file() and not second.exists()
    with pytest.raises(ValueError, match='256'):
        m.dasiwa_save_bundle(refs * 257, [])


def test_invalid_request_and_concept_do_not_publish(tmp_path, monkeypatch):
    m = exporter()
    monkeypatch.setattr(m, 'refmods_roots', lambda: [str(tmp_path)])
    with pytest.raises(ValueError, match='request_id'):
        m.DaSiWaH3RefModExport().create_refmod('{}', '')
    with pytest.raises(ValueError, match='concept_type'):
        m.dasiwa_save_bundle([(torch.zeros(1, 24, 1, 2, 2), {'kind': 'image'})], [], concept_type='train')
    assert not list(tmp_path.iterdir())


def test_audio_only_attached_does_not_require_embedded_stream(media_files, monkeypatch):
    import av
    import numpy as np
    m = exporter()
    with av.open(str(media_files / 'silent-video.mkv'), 'w') as container:
        stream = container.add_stream('ffv1', rate=24)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    monkeypatch.setattr(m, 'dasiwa_input_root', lambda: str(media_files))
    refs, _ = m.dasiwa_export_refs(json.dumps({'items': [{
        'type': 'video', 'value': 'silent-video.mkv', 'media_mode': 'audio', 'audio': 'sound.wav',
        'trim_start': .5, 'trim_end': 1.5}]}), audio_vae=AudioVAE())
    assert [meta['kind'] for _, meta in refs] == ['audio']
