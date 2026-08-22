import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from comfy.nested_tensor import NestedTensor


ROOT = Path(__file__).parents[1]
PACKAGE_NAME = "dasiwa_sequence_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)
sequence = importlib.import_module(f"{PACKAGE_NAME}.nodes.helper_minimax_h3_sequence")
cache = importlib.import_module(f"{PACKAGE_NAME}.nodes.helper_minimax_h3_sequence_cache")
nodes = importlib.import_module(f"{PACKAGE_NAME}.nodes.nodes_minimax_h3_sequence")


def guide(mode="FL2VA", length=5, width=32, height=32):
    return {
        "version": 2,
        "mode": mode,
        "prompt": mode,
        "resolved_prompt": mode,
        "width": width,
        "height": height,
        "length": length,
    }


def av_latent(video_t, audio_t, value=0.0, device="cpu"):
    video = torch.full((1, 24, video_t, 2, 2), value, device=device)
    audio = torch.full((1, 32, 2, audio_t), value, device=device)
    return {"samples": NestedTensor((video, audio))}


def audio_for_frames(frames, sample_rate=32000, value=0.0):
    samples = round(frames / 24 * sample_rate)
    return {
        "waveform": torch.full((1, 2, samples), value),
        "sample_rate": sample_rate,
    }


def test_sequence_plan_requires_one_canvas_and_serializes_without_tensors():
    first = sequence.sequence_segment(guide(), True, "first", 0)
    second = sequence.sequence_segment(guide("REF2VA"), False, "second", 2, torch.zeros(5, 32, 32, 3))

    plan = sequence.sequence_plan([first, second])
    summary = sequence.sequence_summary(plan)

    assert plan["width"] == 32
    assert '"mode": "REF2VA"' in summary
    assert '"source": true' in summary
    try:
        sequence.sequence_plan([first, sequence.sequence_segment(guide(width=64), True, "", 0)])
    except ValueError as exc:
        assert "share one canvas" in str(exc)
    else:
        raise AssertionError("mixed sequence canvases must be rejected")


def test_prepare_continuation_pins_synchronized_cpu_tail_and_masks_it():
    source = av_latent(37, 207)
    source_video, source_audio = source["samples"].unbind()
    source_video.copy_(torch.arange(37).reshape(1, 1, 37, 1, 1))
    source_audio.copy_(torch.arange(207).reshape(1, 1, 1, 207))
    target = av_latent(44, 243)

    output = sequence.prepare_continuation(target, sequence.latent_to_cpu(source), 22)
    video, audio = output["samples"].unbind()
    video_mask, audio_mask = output["noise_mask"].unbind()

    assert torch.equal(video[:, :, :7], source_video[:, :, -7:])
    assert torch.equal(audio[..., :37], source_audio[..., -37:])
    assert torch.count_nonzero(video_mask[:, :, :7]) == 0
    assert torch.count_nonzero(audio_mask[..., :37]) == 0
    assert torch.all(video_mask[:, :, 7:] == 1)
    assert torch.all(audio_mask[..., 37:] == 1)


def test_segment_cache_round_trip_rejects_a_changed_fingerprint(tmp_path):
    latent = av_latent(2, 8, value=3.0)
    images = torch.full((5, 32, 32, 3), 0.25)
    audio = audio_for_frames(5, value=0.5)

    with mock.patch.object(cache.folder_paths, "get_output_directory", return_value=str(tmp_path)):
        assert cache.save_segment("node", "shot-a", 0, "fp-a", images, audio, latent)
        loaded = cache.load_segment("node", "shot-a", 0, "fp-a")
        stale = cache.load_segment("node", "shot-a", 0, "fp-b")

    assert stale is None
    assert torch.equal(loaded["images"], images)
    assert torch.equal(loaded["audio"]["waveform"], audio["waveform"])
    loaded_video, loaded_audio = loaded["latent"]["samples"].unbind()
    assert loaded_video.device.type == "cpu"
    assert loaded_audio.device.type == "cpu"


def test_sequence_executor_routes_models_and_uses_cpu_latent_continuity():
    first = sequence.sequence_segment(guide("FL2VA"), True, "", 0)
    second = sequence.sequence_segment(guide("REF2VA"), True, "", 0)
    plan = sequence.sequence_plan([first, second])
    calls = []

    def fake_apply(_clip, _vae, active_guide, _audio_vae):
        frames = active_guide["length"]
        video_t = sequence.video_latent_t(frames)
        audio_t = round(frames / 24 * 40)
        return "positive", av_latent(video_t, audio_t)

    def fake_sample(model, _clip, _vae, active_guide, seed, *_args, **kwargs):
        frames = active_guide["length"]
        calls.append((model, seed, frames, kwargs.get("latent")))
        video_t = sequence.video_latent_t(frames)
        audio_t = round(frames / 24 * 40)
        value = float(len(calls))
        return (
            torch.full((frames, 32, 32, 3), value),
            audio_for_frames(frames, value=value),
            24.0,
            frames,
            av_latent(video_t, audio_t, value=value),
            "sampled",
        )

    with (
        mock.patch.object(nodes.MiniMaxH3DirectorGuide, "apply", side_effect=fake_apply),
        mock.patch.object(nodes, "_sample_and_decode", side_effect=fake_sample),
        mock.patch.object(nodes, "_cleanup_segment_vram"),
        mock.patch.object(nodes, "load_segment", return_value=None),
    ):
        result = nodes.MiniMaxH3SequenceExecutor.execute(
            plan,
            "clip",
            "video-vae",
            10,
            25,
            1.0,
            "res_multistep",
            "simple",
            12.0,
            3.0,
            True,
            5,
            "off",
            True,
            "fl-model",
            "ref-model",
            "audio-vae",
        )

    assert result[0].shape == (10, 32, 32, 3)
    assert result[2:4] == (24.0, 10)
    assert [call[:3] for call in calls] == [
        ("fl-model", 10, 5),
        ("ref-model", 11, 22),
    ]
    continuation = calls[1][3]
    continuation_video, continuation_audio = continuation["samples"].unbind()
    assert torch.all(continuation_video[:, :, :2] == 1)
    assert torch.all(continuation_audio[..., :8] == 1)
    assert '"origin": "generated"' in result[4]


def test_skipped_segment_uses_source_when_cache_is_missing():
    source_images = torch.full((5, 32, 32, 3), 0.75)
    source_audio = audio_for_frames(5, value=0.25)
    segment = sequence.sequence_segment(
        guide(), False, "missing", 0, source_images, source_audio
    )
    plan = sequence.sequence_plan([segment])

    with mock.patch.object(nodes, "load_segment", return_value=None):
        result = nodes.MiniMaxH3SequenceExecutor.execute(
            plan,
            "clip",
            "video-vae",
            0,
            25,
            1.0,
            "res_multistep",
            "simple",
            12.0,
            3.0,
            False,
            22,
            "read_only",
            True,
        )

    assert torch.equal(result[0], source_images)
    assert torch.equal(result[1]["waveform"], source_audio["waveform"])
    assert '"origin": "source"' in result[4]


class MiniMaxH3SequenceTests(unittest.TestCase):
    def test_plan_contract(self):
        test_sequence_plan_requires_one_canvas_and_serializes_without_tensors()

    def test_latent_continuity_contract(self):
        test_prepare_continuation_pins_synchronized_cpu_tail_and_masks_it()

    def test_cache_contract(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            test_segment_cache_round_trip_rejects_a_changed_fingerprint(Path(directory))

    def test_executor_contract(self):
        test_sequence_executor_routes_models_and_uses_cpu_latent_continuity()

    def test_source_passthrough_contract(self):
        test_skipped_segment_uses_source_when_cache_is_missing()


if __name__ == "__main__":
    unittest.main()
