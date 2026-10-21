"""Explicit AV1/Auto selection, unchanged overrides and actual MP4 output."""
import importlib.util
from pathlib import Path
import sys
import types

import av
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def video(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'nodes'))
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=None))
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path),
        get_temp_directory=lambda: str(tmp_path),
        get_save_image_path=lambda *args: (str(tmp_path), 'test', 1, '', 'test'),
    ))
    spec = importlib.util.spec_from_file_location('video_container_probe', ROOT / 'nodes/nodes_enhanced_video_combine.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_av1_auto_prefers_mp4_and_retains_fallbacks(video):
    assert video._container_candidates('AV1', 'Auto') == ('MP4', 'WebM', 'MKV')
    assert video._container_candidates('VP9', 'Auto') == ('WebM', 'MKV', 'MP4')
    assert video._auto_container_candidates('AV1', 'Auto') == ('MP4',)


@pytest.mark.parametrize('codec', ['AV1', 'VP9', 'H.264', 'H.265 (HEVC)'])
@pytest.mark.parametrize('container', ['MP4', 'MKV', 'WebM'])
def test_explicit_container_is_unchanged(video, codec, container):
    assert video._container_candidates(codec, container) == (container,)


def test_explicit_av1_auto_attempts_container_fallbacks_then_h264(video, monkeypatch):
    attempts = []
    def encode(*args, **kwargs):
        attempts.append((args[1], args[8]))
        if args[1] == 'AV1':
            raise RuntimeError('injected encode failure')
        return 'fixture'
    monkeypatch.setattr(video, '_encode_with_available_encoder', encode)
    video.DaSiWa_EnhancedVideoCombine().combine(
        torch.zeros((2, 32, 32, 3)), 24, 'AV1', 'Auto', '8-bit', 20,
        False, False, 'test', True, False,
    )
    assert attempts == [('AV1', 'MP4'), ('AV1', 'WebM'), ('AV1', 'MKV'), ('H.264', 'MP4')]


@pytest.mark.parametrize('with_audio', [False, True])
def test_real_explicit_av1_auto_writes_decodable_faststart_mp4(video, monkeypatch, with_audio):
    # Bound the probe to CPU encoding; hardware candidate ordering is not changed.
    monkeypatch.setitem(video._ENCODER_NAMES, 'AV1', ('libsvtav1',))
    images = torch.linspace(0, 1, 6)[:, None, None, None].expand(6, 64, 64, 3)
    audio = {'waveform': torch.zeros((1, 1, 12000)), 'sample_rate': 48000} if with_audio else None
    result = video.DaSiWa_EnhancedVideoCombine().combine(
        images, 24, 'AV1', 'Auto', '8-bit', 35, False, False, 'test', True, False, audio=audio,
    )
    path = Path(result['result'][1])
    assert path.suffix == '.mp4'
    assert result['ui']['gifs'][0]['format'] == 'video/mp4'
    assert result['ui']['gifs'][0]['codec'] == 'AV1'
    data = path.read_bytes()
    assert 0 < data.index(b'moov') < data.index(b'mdat')
    with av.open(str(path)) as media:
        assert media.streams.video[0].codec_context.name in ('av1', 'libdav1d')
        assert len(list(media.decode(video=0))) == 6
    if with_audio:
        with av.open(str(path)) as media:
            assert media.streams.audio[0].codec_context.name == 'aac'
            assert sum(frame.samples for frame in media.decode(audio=0)) >= 12000
