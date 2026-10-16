"""Read-only duration lookup with real tiny PyAV media, without Core/models."""
import ast
import asyncio
import hashlib
import importlib
import json
import sys
import threading
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace

import av
import numpy as np
import pytest
from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = '_h3_media_info_tests'
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'nodes/h3_continuity')]
sys.modules[PACKAGE] = package
assembly = importlib.import_module(PACKAGE + '.assembly')


def make_video(path, fps=24, audio=False, codec='ffv1'):
    with av.open(str(path), 'w') as container:
        video = container.add_stream(codec, rate=fps)
        video.width, video.height, video.pix_fmt = 64, 32, 'yuv420p'
        sound = container.add_stream('pcm_s16le', rate=48000) if audio else None
        if sound:
            sound.layout = 'stereo'
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), i * 10, dtype=np.uint8), format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if sound:
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 48000), dtype=np.int16), format='s16', layout='stereo')
            frame.sample_rate = 48000
            for packet in sound.encode(frame):
                container.mux(packet)
            for packet in sound.encode(None):
                container.mux(packet)


def test_real_h264_sei_duration(tmp_path):
    path=tmp_path / 'clip.mp4'
    make_video(path, codec='libx264')
    with av.open(str(path)) as container:
        first=next(container.decode(video=0))
        assert len(first.side_data)>0
    result=assembly.media_info(path.name, input_root=tmp_path)
    assert result['ok'] is True
    assert result['seconds']==pytest.approx(.5, abs=.001)


def test_real_mkv_duration_is_read_only(tmp_path):
    path = tmp_path / 'clip.mkv'
    make_video(path, audio=True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    entries = set(tmp_path.iterdir())
    result = assembly.media_info(path.name, input_root=tmp_path)
    assert result == dict(ok=True, seconds=pytest.approx(.5, abs=.001), width=64, height=32, fps=24., has_audio=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert set(tmp_path.iterdir()) == entries


def call_route(query, monkeypatch, input_root):
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(get_input_directory=lambda: str(input_root)))
    tree = ast.parse((ROOT / 'nodes/h3_continuity/routes.py').read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'assembly_media_info']
    assert functions, 'GET assembly media-info route is not implemented'
    registered = []
    def get(path):
        registered.append(path)
        return lambda fn: fn
    namespace = dict(__package__=PACKAGE, web=web, asyncio=asyncio, routes=SimpleNamespace(get=get))
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<media-info route>', 'exec'), namespace)
    response = asyncio.run(namespace['assembly_media_info'](SimpleNamespace(query=query)))
    assert registered == ['/df_h3_continuity/assembly/media-info']
    return response.status, json.loads(response.body)


def test_get_route_probes_real_media_off_event_loop(tmp_path, monkeypatch):
    make_video(tmp_path / 'clip.mkv')
    original = assembly.media_info
    threads = []
    def observed(filename):
        threads.append(threading.get_ident())
        return original(filename)
    monkeypatch.setattr(assembly, 'media_info', observed)
    status, result = call_route({'filename': 'clip.mkv'}, monkeypatch, tmp_path)
    assert status == 200 and result['ok'] is True
    assert result['seconds'] == pytest.approx(.5, abs=.001)
    assert threads and threads[0] != threading.get_ident()


def test_symlink_inside_input_is_rejected(tmp_path):
    make_video(tmp_path / 'clip.mkv')
    (tmp_path / 'alias.mkv').symlink_to(tmp_path / 'clip.mkv')
    result = assembly.media_info('alias.mkv', input_root=tmp_path)
    assert result['ok'] is False and result['issues']


def test_symlink_cycle_is_explicit_error(tmp_path):
    (tmp_path / 'loop.mkv').symlink_to('loop.mkv')
    result = assembly.media_info('loop.mkv', input_root=tmp_path)
    assert result['ok'] is False and result['issues']


def test_unsupported_join_fps_still_has_duration(tmp_path):
    make_video(tmp_path / 'clip.mkv', fps=30)
    result = assembly.media_info('clip.mkv', input_root=tmp_path)
    assert result['ok'] is True and result['fps'] == 30
    assert result['seconds'] == pytest.approx(.4, abs=.001)
    assert result['has_audio'] is False
    plan = assembly.preflight(dict(version=1, rows=[dict(kind='media', filename='clip.mkv')]),
                              input_root=tmp_path, output_root=tmp_path)
    assert plan['ok'] is False and '24-fps' in plan['issues'][0]


@pytest.mark.parametrize('filename', [None, '', '../clip.mkv', 'sub/../../clip.mkv',
    '..\\clip.mkv', '/etc/passwd', 'https://host/clip.mkv', '//host/share/clip.mkv',
    'C:\\clip.mkv', 'missing.mkv', 'bad\x00.mkv'])
def test_untrusted_handles_never_probe(filename, tmp_path, monkeypatch):
    media = importlib.import_module(PACKAGE + '.pyav_media')
    def forbidden(*args, **kwargs):
        pytest.fail('Unsafe filename reached the media probe')
    monkeypatch.setattr(media, 'probe', forbidden)
    result = assembly.media_info(filename, input_root=tmp_path)
    assert result['ok'] is False and result['issues']
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize('query', [{}, {'filename': ''}, {'filename': '../private.mkv'}, {'filename': 'broken.mkv'}])
def test_route_rejects_missing_or_invalid_media_without_duration(query, tmp_path, monkeypatch):
    (tmp_path / 'broken.mkv').write_bytes(b'not video')
    status, result = call_route(query, monkeypatch, tmp_path)
    assert status == 400 and result['ok'] is False and result['issues']
    assert 'seconds' not in result
    assert str(tmp_path) not in json.dumps(result)


def test_missing_duration_is_explicit_and_decoder_diagnostics_are_private(tmp_path, monkeypatch):
    path = tmp_path / 'clip.mkv'
    make_video(path)
    media = importlib.import_module(PACKAGE + '.pyav_media')
    def unavailable(path):
        raise ValueError(f'Could not determine source duration: {path}')
    monkeypatch.setattr(media, 'probe', unavailable)
    result = assembly.media_info(path.name, input_root=tmp_path)
    assert result['ok'] is False and result['issues']
    assert 'seconds' not in result and str(tmp_path) not in json.dumps(result)


def test_lookup_uses_existing_45_second_guard_without_hashing(tmp_path, monkeypatch):
    make_video(tmp_path / 'clip.mkv')
    media = importlib.import_module(PACKAGE + '.pyav_media')
    original = media.guard
    timeouts = []
    def observed(timeout):
        timeouts.append(timeout)
        return original(timeout)
    def forbidden(*args, **kwargs):
        pytest.fail('Read-only metadata lookup must not hash/import/store media')
    monkeypatch.setattr(media, 'guard', observed)
    monkeypatch.setattr(hashlib, 'sha256', forbidden)
    assert assembly.media_info('clip.mkv', input_root=tmp_path)['ok'] is True
    assert timeouts == [45]
