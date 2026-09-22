import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import torch
import pytest
from PIL import Image


class _FolderPaths:
    @staticmethod
    def get_output_directory():
        return "/tmp"

    @staticmethod
    def get_temp_directory():
        return "/tmp"

    @staticmethod
    def get_save_image_path(prefix, output_dir, width, height):
        return output_dir, prefix.replace("/", "_"), 1, "", prefix


sys.modules.setdefault("folder_paths", _FolderPaths())
HELPER_PATH = Path(__file__).parents[1] / "nodes" / "helper_logging.py"
helper_spec = importlib.util.spec_from_file_location("helper_logging", HELPER_PATH)
assert helper_spec is not None and helper_spec.loader is not None
helper_logging = importlib.util.module_from_spec(helper_spec)
sys.modules["helper_logging"] = helper_logging
helper_spec.loader.exec_module(helper_logging)
PYAV_HELPER_PATH = Path(__file__).parents[1] / "nodes" / "helper_pyav_video.py"
pyav_helper_spec = importlib.util.spec_from_file_location("helper_pyav_video", PYAV_HELPER_PATH)
assert pyav_helper_spec is not None and pyav_helper_spec.loader is not None
helper_pyav_video = importlib.util.module_from_spec(pyav_helper_spec)
sys.modules["helper_pyav_video"] = helper_pyav_video
pyav_helper_spec.loader.exec_module(helper_pyav_video)
MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_enhanced_video_combine.py"
spec = importlib.util.spec_from_file_location("nodes_enhanced_video_combine", MODULE_PATH)
assert spec is not None and spec.loader is not None
enhanced_video_combine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_video_combine)


def test_encoding_uses_pyav_without_external_processes():
    source = (Path(__file__).parents[1] / "nodes" / "nodes_enhanced_video_combine.py").read_text(encoding="utf-8")
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert "import subprocess" not in source
    assert "create_subprocess_exec" not in source
    assert "subprocess." not in source
    assert '"av>=18.0"' in pyproject
    assert '"imageio-ffmpeg' not in pyproject


def test_node_schema_and_registration():
    controls = enhanced_video_combine.DaSiWa_EnhancedVideoCombine.INPUT_TYPES()["required"]
    package_source = (Path(__file__).parents[1] / "__init__.py").read_text(encoding="utf-8")
    preview_source = (Path(__file__).parents[1] / "js" / "enhanced_video_combine_preview.js").read_text(encoding="utf-8")

    assert {"images", "bit_depth", "pass_frames", "save_first_frame", "save_last_frame", "crop_to_audio", "audio_codec", "audio_bitrate", "filename_prefix", "quality", "pingpong", "save_metadata", "log_level"} <= controls.keys()
    assert list(controls).index("log_level") == 6
    assert list(controls).index("crop_to_audio") == 12
    assert list(controls).index("audio_codec") == 13
    assert list(controls).index("audio_bitrate") == 14
    assert list(controls).index("save_first_frame") > list(controls).index("audio_bitrate")
    optional = enhanced_video_combine.DaSiWa_EnhancedVideoCombine.INPUT_TYPES()["optional"]
    assert optional["audio"][0] == "AUDIO"
    assert optional["seed"][0] == "INT"
    assert optional["seed"][1]["forceInput"] is True
    assert controls["codec"][0] == ["Auto", "AV1", "VP9", "H.265 (HEVC)", "H.264"]
    assert controls["container"][0] == ["Auto", "WebM", "MKV", "MP4", "Animated WebP", "Animated AVIF"]
    assert controls["quality"][1]["default"] == 20
    assert controls["pingpong"][1]["default"] is False
    assert controls["pass_frames"][1]["default"] is False
    assert controls["filename_prefix"][1]["default"] == "video_%date:hhmmss%"
    assert enhanced_video_combine._output_filename("video_130405", 1, ".mp4", False) == "video_130405_00001.mp4"
    assert enhanced_video_combine._output_filename("video_130405", 1, ".mp4", True) == "video_130405_00001_audio.mp4"

    assert controls["audio_codec"][0] == ["Auto", "AAC", "Opus", "MP3"]
    assert controls["audio_bitrate"][1]["default"] == "192k"
    assert "hideLegacyLogLevelWidget" in preview_source
    assert 'widget.name === "log_level"' in preview_source
    assert "hideFrameExportWidgets(this);" in preview_source
    assert 'for (const name of ["save_first_frame", "save_last_frame"])' in preview_source
    assert "DaSiWa_EnhancedVideoCombine" in package_source
    assert 'name: "DaSiWa.EnhancedVideoCombinePreview"' in preview_source
    assert "this.addDOMWidget" in preview_source
    assert "message?.gifs?.[0] ?? message?.videos?.[0]" in preview_source
    assert "function saveFrame" not in preview_source
    assert "link.download" not in preview_source
    assert "download.download" in preview_source
    assert "previewWidget.aspectRatio = 16 / 9;" in preview_source
    assert "transcodedVideoUrl" in preview_source
    assert "function shouldUseTranscodedPreview(video)" in preview_source
    assert 'const NATIVE_BROWSER_VIDEO = new Set(["H.264|MP4|8"]);' in preview_source
    assert '"AV1|WebM|8"' not in preview_source
    assert '"VP9|WebM|8"' not in preview_source
    assert "getHeight: () => previewHeight()," in preview_source
    assert "node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);" in preview_source
    assert "video.fps" in preview_source
    assert '"Video preview"' not in preview_source
    assert "preview.controls = true;" in preview_source
    assert "preview.controls = false;" not in preview_source
    assert 'preview.controlsList = "nodownload nofullscreen noremoteplayback";' in preview_source
    assert "preview.disablePictureInPicture = true;" in preview_source
    assert 'preview.style.cssText = "display:block;width:100%;background:#111;cursor:pointer";' in preview_source
    assert "const videoFrame = document.createElement" not in preview_source
    assert "previewWidget.aspectRatio = preview.videoWidth / preview.videoHeight" in preview_source
    assert "fitPreviewHeight(previewNode)" in preview_source
    assert "const previewWidth" not in preview_source
    assert "const previewHeight" in preview_source
    assert "aspect-ratio:16/9" not in preview_source
    assert "this.onResize = function (size)" not in preview_source
    assert "this.setSize([size[0], Math.max(1, size[1] + heightDelta)])" not in preview_source
    assert "syncBooleanWidget" in preview_source
    assert 'syncBooleanWidget(this, "save_first_frame", saveFirstFrame.checked)' in preview_source
    assert 'syncBooleanWidget(this, "save_last_frame", saveLastFrame.checked)' in preview_source
    assert "actions.append(saveFirstFrameLabel, saveLastFrameLabel, muteLabel, autoPlayLabel, download)" in preview_source
    assert 'autoPlay.type = "checkbox"' in preview_source
    assert "this.properties ??= {};" in preview_source
    assert "autoPlay.checked = this.properties.autoplay ?? true;" in preview_source
    assert "this.properties.autoplay = autoPlay.checked;" in preview_source
    assert 'autoPlayLabel.append(autoPlay, " Autoplay")' in preview_source
    assert 'margin-left:auto' in preview_source
    assert "if (autoPlay.checked) preview.play().catch(() => {});" in preview_source
    assert 'preview.addEventListener("mouseenter"' in preview_source
    assert 'preview.addEventListener("mouseleave"' in preview_source
    assert 'preview.addEventListener("dblclick", (event) => event.preventDefault())' in preview_source
    assert "const controls = document.createElement" not in preview_source
    assert 'mute.type = "checkbox"' in preview_source
    assert "mute.checked = this.properties.muted ?? false;" in preview_source
    assert "this.properties.muted = mute.checked;" in preview_source
    assert 'muteLabel.append(mute, " Mute")' in preview_source
    assert "if (!mute.checked) preview.muted = false;" in preview_source
    assert "if (this.properties?.muted && !preview.muted) preview.muted = true;" in preview_source
    assert "this.dasiwaMuteCheckbox = mute;" in preview_source
    assert "this.dasiwaMuteCheckbox.checked = this.properties?.muted ?? false;" in preview_source

    assert 'preview.dataset.filename = video.filename' in preview_source
    assert "this.dasiwaVideoPreviewWidget = previewWidget;" in preview_source
    on_executed_source = preview_source.split("nodeType.prototype.onExecuted", 1)[1]
    assert "this.dasiwaVideoPreviewWidget.aspectRatio" in on_executed_source
    assert "preview.addEventListener(\"error\"" in preview_source
    assert "Preview unavailable (PyAV transcode or browser decoder failed)" in preview_source
    assert "const originalUrl = videoUrl(video);" in on_executed_source
    assert "preview.src = shouldUseTranscodedPreview(video) ? transcodedVideoUrl(video) : originalUrl;" in on_executed_source
    assert 'download.textContent = "Download"' in preview_source
    assert "download.download = video.filename;" in on_executed_source
    assert "function showHelpDialog()" in preview_source
    assert "Enhanced Video Combine Help" in preview_source
    assert "Animated WebP and Animated AVIF are manual image-animation outputs" in preview_source
    assert "onDrawForeground" in preview_source
    assert "isHelpIconHit" in preview_source
    node_source = (Path(__file__).parents[1] / "nodes" / "nodes_enhanced_video_combine.py").read_text(encoding="utf-8")
    assert "ProgressBar(_encoded_frame_count(images, pingpong))" in node_source
    assert "helper_pyav_video.encode_attempt" in node_source
    assert "asyncio.to_thread(helper_pyav_video.transcode_preview" in node_source
    assert "web.FileResponse" in node_source




def test_preview_source_path_accepts_only_output_assets(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    video = output_dir / "preview.mp4"
    video.write_bytes(b"video")
    calls = []

    def directory_for_type(output_type):
        calls.append(output_type)
        return str(output_dir)

    monkeypatch.setattr(enhanced_video_combine.folder_paths, "get_directory_by_type", directory_for_type, raising=False)

    assert enhanced_video_combine._preview_source_path("preview.mp4", "", "output") == str(video)
    assert enhanced_video_combine._preview_source_path("preview.mp4", "", "input") is None
    assert enhanced_video_combine._preview_source_path("preview.mp4", "../", "output") is None
    assert calls == ["output"]


def test_auto_bit_depth_distinguishes_8_and_10_bit_quantization():
    eight_bit = torch.tensor([0, 64, 127, 255], dtype=torch.float32).reshape(1, 2, 2, 1) / 255
    ten_bit = torch.tensor([0, 256, 511, 1023], dtype=torch.float32).reshape(1, 2, 2, 1) / 1023

    assert enhanced_video_combine.detect_bit_depth(eight_bit) == 8
    assert enhanced_video_combine.detect_bit_depth(ten_bit) == 10


def test_validate_inputs_accepts_comfyui_positional_signature():
    node = enhanced_video_combine.DaSiWa_EnhancedVideoCombine()
    assert node.VALIDATE_INPUTS(images=object()) is True
    assert node.validate_inputs("images", "IMAGE", object(), object()) is True
    assert enhanced_video_combine.DaSiWa_EnhancedVideoCombine.__dict__["validate_inputs"](
        node, "images", "IMAGE", object(), object()
    ) is True


def test_output_node_is_changed_for_each_queued_prompt():
    assert math.isnan(enhanced_video_combine.DaSiWa_EnhancedVideoCombine.IS_CHANGED())


def test_10_bit_frame_data_uses_rgb48le_values():
    images = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32)
    payload = enhanced_video_combine._frame_bytes(images, 10)

    assert len(payload) == 6
    assert torch.frombuffer(bytearray(payload), dtype=torch.uint16).tolist() == [0, 32768, 65472]


def test_frame_byte_chunks_are_bounded_and_preserve_frame_order():
    images = torch.tensor([0, 1, 2, 3], dtype=torch.float32).reshape(4, 1, 1, 1).repeat(1, 1, 1, 3) / 255

    chunks = list(enhanced_video_combine._iter_frame_byte_chunks(images, 8, False, max_chunk_bytes=6))

    assert [len(chunk) for chunk in chunks] == [6, 6]
    assert torch.frombuffer(bytearray(b"".join(chunks)), dtype=torch.uint8).reshape(-1, 3)[:, 0].tolist() == [0, 1, 2, 3]


def test_frame_byte_chunks_emit_pingpong_frames_without_materializing_a_batch():
    images = torch.tensor([0, 1, 2, 3], dtype=torch.float32).reshape(4, 1, 1, 1).repeat(1, 1, 1, 3) / 255

    chunks = enhanced_video_combine._iter_frame_byte_chunks(images, 8, True, max_chunk_bytes=3)

    assert torch.frombuffer(bytearray(b"".join(chunks)), dtype=torch.uint8).reshape(-1, 3)[:, 0].tolist() == [0, 1, 2, 3, 2, 1]


def test_pyav_streaming_encode_writes_all_frames(tmp_path):
    av = pytest.importorskip("av")
    images = torch.tensor([0, 1, 2, 3], dtype=torch.float32).reshape(4, 1, 1, 1).repeat(1, 16, 16, 3) / 255
    output_path = tmp_path / "frames.mkv"

    helper_pyav_video.encode_attempt(
        str(output_path), "MKV", "libx264", 16, 16, 24, 8, 20,
        lambda: enhanced_video_combine._frame_arrays(images, 8, False),
    )

    with av.open(str(output_path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 4


def test_pyav_outputs_have_keyframes_at_least_every_second(tmp_path):
    av = pytest.importorskip("av")
    images = torch.zeros((73, 16, 16, 3), dtype=torch.float32)
    output_path = tmp_path / "seekable.mp4"
    helper_pyav_video.encode_attempt(
        str(output_path), "MP4", "libx264", 16, 16, 24, 8, 20,
        lambda: enhanced_video_combine._frame_arrays(images, 8, False),
    )

    with av.open(str(output_path)) as container:
        keyframe_times = [
            float(packet.pts * packet.time_base)
            for packet in container.demux(video=0)
            if packet.pts is not None and packet.is_keyframe
        ]
    assert keyframe_times == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=0.05)


def test_pyav_round_trips_audio_metadata_and_preview(tmp_path):
    av = pytest.importorskip("av")
    images = torch.zeros((4, 16, 16, 3), dtype=torch.float32)
    source = tmp_path / "source.mkv"
    audio = (torch.zeros((2, 8000), dtype=torch.float32).numpy(), 8000)
    helper_pyav_video.encode_attempt(
        str(source), "MKV", "libx264", 16, 16, 4, 8, 20,
        lambda: enhanced_video_combine._frame_arrays(images, 8, False),
        {"prompt": {"text": "test"}}, audio, "aac", "128k",
    )
    preview = helper_pyav_video.transcode_preview(str(source))
    try:
        with av.open(str(source)) as container:
            assert [stream.type for stream in container.streams] == ["video", "audio"]
            assert json.loads(container.metadata["PROMPT"]) == {"text": "test"}
        with av.open(preview) as container:
            assert [(stream.type, stream.codec_context.name) for stream in container.streams] == [("video", "h264"), ("audio", "aac")]
    finally:
        os.unlink(preview)


def test_pyav_animated_webp_and_avif_preserve_multiple_frames(tmp_path):
    av = pytest.importorskip("av")
    images = torch.rand((3, 32, 32, 3), dtype=torch.float32)
    for container_name, encoder, suffix in (("Animated WebP", "libwebp_anim", ".webp"), ("Animated AVIF", "libsvtav1", ".avif")):
        if encoder not in helper_pyav_video.available_encoders():
            pytest.skip(f"{encoder} is unavailable in this PyAV build")
        output = tmp_path / f"animation{suffix}"
        helper_pyav_video.encode_attempt(
            str(output), container_name, encoder, 32, 32, 8, 8, 20,
            lambda: enhanced_video_combine._frame_arrays(images, 8, False),
        )
        with Image.open(output) as animation:
            assert getattr(animation, "n_frames", 1) == 3


def test_frame_exports_are_written_as_pngs_beside_the_video(tmp_path):
    images = torch.tensor([
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
    ])
    output_path = tmp_path / "video_00001-audio.mp4"

    exports = enhanced_video_combine._save_frame_exports(images, str(output_path), True, True)

    assert exports == [
        str(tmp_path / "video_00001-audio-first-frame.png"),
        str(tmp_path / "video_00001-audio-last-frame.png"),
    ]
    assert all(Path(path).is_file() for path in exports)


def test_encoder_priority_prefers_nvenc_then_other_hardware_then_software():
    assert enhanced_video_combine._ENCODER_NAMES["H.264"][:2] == ("h264_nvenc", "h264_qsv")
    assert enhanced_video_combine._ENCODER_NAMES["H.265 (HEVC)"][:2] == ("hevc_nvenc", "hevc_qsv")
    assert enhanced_video_combine._ENCODER_NAMES["AV1"][:2] == ("av1_nvenc", "av1_qsv")
    assert enhanced_video_combine._ENCODER_NAMES["VP9"] == ("vp9_qsv", "vp9_vaapi", "libvpx-vp9")


def test_auto_codec_prioritizes_av1_then_browser_compatible_fallbacks():
    assert enhanced_video_combine._codec_candidates("Auto") == ("AV1", "VP9", "H.264")
    assert enhanced_video_combine._codec_candidates("H.264") == ("H.264",)


def test_auto_codec_forces_eight_bit_browser_compatible_output():
    images = torch.tensor([0, 256, 511, 1023], dtype=torch.float32).reshape(1, 2, 2, 1).repeat(1, 1, 1, 3) / 1023

    assert enhanced_video_combine._selected_bit_depth("Auto", "Auto", images) == 8
    assert enhanced_video_combine._selected_bit_depth("H.264", "Auto", images) == 10
    assert enhanced_video_combine._selected_bit_depth("Auto", "10-bit", images) == 10


def test_auto_container_prioritizes_webm_then_mkv_then_mp4_for_av1_and_vp9():
    assert enhanced_video_combine._container_candidates("AV1", "Auto") == ("WebM", "MKV", "MP4")
    assert enhanced_video_combine._container_candidates("VP9", "Auto") == ("WebM", "MKV", "MP4")
    assert enhanced_video_combine._container_candidates("H.264", "Auto") == ("MP4", "MKV")
    assert enhanced_video_combine._container_candidates("H.265 (HEVC)", "MKV") == ("MKV",)
    assert "Animated WebP" not in enhanced_video_combine._container_candidates("AV1", "Auto")
    assert "Animated AVIF" not in enhanced_video_combine._container_candidates("AV1", "Auto")


def test_browser_compatible_auto_containers_exclude_mkv():
    assert enhanced_video_combine._auto_container_candidates("AV1", "Auto") == ("WebM",)
    assert enhanced_video_combine._auto_container_candidates("VP9", "Auto") == ("WebM",)
    assert enhanced_video_combine._auto_container_candidates("H.264", "Auto") == ("MP4",)
    assert enhanced_video_combine._auto_container_candidates("AV1", "MKV") == ("MKV",)


def test_animated_image_outputs_are_manual_only_and_use_dedicated_encoders():
    assert enhanced_video_combine._animated_image_settings("Animated WebP") == (".webp", "libwebp_anim")
    assert enhanced_video_combine._animated_image_settings("Animated AVIF") == (".avif", "libaom-av1")
    assert enhanced_video_combine._animated_image_settings("Auto") is None
    assert enhanced_video_combine._animated_image_encoder_candidates("Animated AVIF") == (
        "av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1",
    )
    assert enhanced_video_combine._animated_image_encoder_candidates("Animated WebP") == ("libwebp_anim",)


def test_animated_avif_prefers_nvenc_over_software(monkeypatch):
    captured = []
    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _backend=None: {"av1_nvenc", "libsvtav1"})
    monkeypatch.setattr(
        helper_pyav_video, "encode_attempt",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    assert enhanced_video_combine._encode_animated_image(
        None, "Animated AVIF", 8, 1024, 1280, 8, lambda: iter(()), "output.avif", 20,
    ) == "av1_nvenc"
    assert captured[0][0][2] == "av1_nvenc"


def test_pingpong_appends_reverse_interior_frames():
    images = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)

    assert enhanced_video_combine._pingpong_frames(images, False).flatten().tolist() == [0, 1, 2, 3, 4]
    assert enhanced_video_combine._pingpong_frames(images, True).flatten().tolist() == [0, 1, 2, 3, 4, 3, 2, 1]


def test_filename_prefix_expands_comfyui_date_format(monkeypatch):
    real_datetime = enhanced_video_combine.datetime.datetime

    class FixedDatetime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 7, 18, 13, 4, 5)

    monkeypatch.setattr(enhanced_video_combine.datetime, "datetime", FixedDatetime)

    assert enhanced_video_combine._format_filename_prefix(
        "video/%date:yyyy-MM-dd%/%date:hhmmss%_%seed%", seed=123456789
    ) == "video/2026-07-18/130405_123456789"
    assert enhanced_video_combine._format_filename_prefix("video_%seed%") == "video_%seed%"


def test_audio_file_converts_comfyui_audio_to_planar_float32():
    audio_data, duration = enhanced_video_combine._audio_file({
        "waveform": torch.tensor([[[0.0, 0.5], [-0.5, 1.0]]]),
        "sample_rate": 2,
    })

    assert audio_data[1] == 2
    assert duration == 1.0
    assert audio_data[0].tolist() == [[0.0, 0.5], [-0.5, 1.0]]


def test_audio_encode_passes_audio_crop_and_bitrate_to_pyav(monkeypatch):
    captured = []
    audio = (torch.zeros((2, 48000)).numpy(), 48000)
    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _backend=None: {"libx264", "libmp3lame"})
    monkeypatch.setattr(helper_pyav_video, "encode_attempt", lambda *args, **kwargs: captured.append((args, kwargs)))

    assert enhanced_video_combine._encode_with_available_encoder(
        None, "H.264", 8, 2, 2, 24, lambda: iter(()), "output.mp4", "MP4", 20, 20,
        None, audio, 1.0, True, "MP3", "128k",
    ) == "libx264"
    args = captured[0][0]
    assert args[11] == "libmp3lame"
    assert args[12] == "128k"
    assert args[13] is True


def test_audio_fallbacks_are_container_compatible():
    assert enhanced_video_combine._audio_encoder_candidates("AAC", "WebM") == ("aac", "libopus")
    assert enhanced_video_combine._audio_encoder_candidates("MP3", "MP4") == ("libmp3lame", "aac")
    assert enhanced_video_combine._audio_encoder_candidates("Auto", "MKV") == ("aac", "libopus", "libmp3lame", "pcm_s16le")


def test_legacy_boolean_audio_codec_uses_auto_selection():
    assert enhanced_video_combine._audio_encoder_candidates(True, "MP4") == ("aac", "libmp3lame")
    assert enhanced_video_combine._audio_encoder_candidates(False, "WebM") == ("libopus",)


def test_audio_encode_falls_back_when_requested_encoder_fails(monkeypatch):
    captured = []
    audio = (torch.zeros((2, 48000)).numpy(), 48000)
    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _backend=None: {"libx264", "libmp3lame", "aac"})

    def encode(*args, **kwargs):
        captured.append(args[11])
        if args[11] == "libmp3lame":
            raise RuntimeError("requested audio encoder is unavailable")

    monkeypatch.setattr(helper_pyav_video, "encode_attempt", encode)
    assert enhanced_video_combine._encode_with_available_encoder(
        None, "H.264", 8, 2, 2, 24, lambda: iter(()), "output.mp4", "MP4", 20, 20,
        None, audio, 1.0, False, "MP3", "128k",
    ) == "libx264"
    assert captured == ["libmp3lame", "aac"]


def test_encoder_listing_uses_pyav(monkeypatch):
    monkeypatch.setattr(helper_pyav_video, "available_encoders", lambda: {"h264_nvenc", "libx264"})
    assert enhanced_video_combine._available_encoders() == {"h264_nvenc", "libx264"}


def test_basic_encode_log_reports_the_actual_audio_codec(monkeypatch, capsys):
    audio = (torch.zeros((2, 48000)).numpy(), 48000)
    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _backend=None: {"libx264", "aac"})
    monkeypatch.setattr(helper_pyav_video, "encode_attempt", lambda *args, **kwargs: None)

    enhanced_video_combine._encode_with_available_encoder(
        None, "H.264", 8, 2, 2, 24, lambda: iter(()), "output.mp4", "MP4", 20, 20, None,
        audio=audio, audio_codec="Auto", audio_bitrate="192k",
    )
    assert "audio=aac/192k" in capsys.readouterr().out


def test_output_and_selected_frame_exports_are_published_to_comfyui_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(enhanced_video_combine, "_encode_with_available_encoder", lambda *args, **kwargs: "libx264")
    monkeypatch.setattr(enhanced_video_combine.folder_paths, "get_output_directory", lambda: str(tmp_path))
    images = torch.rand((2, 4, 6, 3), dtype=torch.float32)

    result = enhanced_video_combine.DaSiWa_EnhancedVideoCombine().combine(
        images, 24.0, "H.264", "MP4", "8-bit", 20, False, False, "asset-video_%seed%", True, False,
        save_first_frame=True, save_last_frame=True, seed=987654321,
    )

    # The video itself is not PIL-openable and must not be published under "images" -
    # /view would try to decode it as a still image and raise UnidentifiedImageError.
    assert result["ui"]["images"] == [
        {"filename": "asset-video_987654321_00001-first-frame.png", "subfolder": "", "type": "output", "format": "image/png", "width": 6, "height": 4},
        {"filename": "asset-video_987654321_00001-last-frame.png", "subfolder": "", "type": "output", "format": "image/png", "width": 6, "height": 4},
    ]
    assert result["ui"]["gifs"] == [
        {"filename": "asset-video_987654321_00001.mp4", "subfolder": "", "type": "output", "format": "video/mp4", "codec": "H.264", "bit_depth": 8, "container": "MP4", "width": 6, "height": 4, "fps": 24.0},
    ]


def test_hevc_output_uses_original_asset_for_streaming_browser_preview(tmp_path, monkeypatch):
    encode_calls = []
    monkeypatch.setattr(
        enhanced_video_combine,
        "_encode_with_available_encoder",
        lambda *args, **kwargs: encode_calls.append(args) or "mock-encoder",
    )
    monkeypatch.setattr(enhanced_video_combine.folder_paths, "get_output_directory", lambda: str(tmp_path))
    images = torch.rand((2, 4, 6, 3), dtype=torch.float32)

    result = enhanced_video_combine.DaSiWa_EnhancedVideoCombine().combine(
        images, 24.0, "H.265 (HEVC)", "MP4", "8-bit", 20, False, False,
        "hevc-video", True, False,
    )

    assert len(encode_calls) == 1
    assert result["ui"]["gifs"] == [{
        "filename": "hevc-video_00001.mp4",
        "subfolder": "",
        "type": "output",
        "format": "video/mp4",
        "codec": "H.265 (HEVC)",
        "bit_depth": 8,
        "container": "MP4",
        "width": 6,
        "height": 4,
        "fps": 24.0,
    }]


def test_hardware_failure_falls_back_to_software(monkeypatch):
    attempts = []
    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _backend=None: {"h264_nvenc", "libx264"})

    def encode(*args, **kwargs):
        attempts.append(args[2])
        if args[2] == "h264_nvenc":
            raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(helper_pyav_video, "encode_attempt", encode)
    assert enhanced_video_combine._encode_with_available_encoder(
        None, "H.264", 8, 16, 16, 24, lambda: iter(()), "video.mp4", "MP4", 20, 20, None,
    ) == "libx264"
    assert attempts == ["h264_nvenc", "libx264"]


def test_audio_encoder_coerces_bool_to_auto():
    # Guards the positional widgets_values drift: a bool must never reach encoding.
    assert enhanced_video_combine._audio_encoder(True, "WebM") == "libopus"
    assert enhanced_video_combine._audio_encoder(True, "MP4") == "aac"
    assert enhanced_video_combine._audio_encoder("Auto", "WebM") == "libopus"
    assert enhanced_video_combine._audio_encoder("AAC", "MP4") == "aac"


def test_audio_codec_combo_schema_is_stable():
    controls = enhanced_video_combine.DaSiWa_EnhancedVideoCombine.INPUT_TYPES()["required"]
    assert controls["audio_codec"][0] == ["Auto", "AAC", "Opus", "MP3"]
    assert controls["audio_codec"][1]["default"] == "Auto"
