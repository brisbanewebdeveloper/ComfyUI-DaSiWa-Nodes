"""Contracts for Director-owned publication through the existing saver nodes."""


def test_image_publication_uses_metadata_saver_and_passes_automatic_metadata():
    from nodes.helper_media_output_v2 import publish_media_output

    calls = []

    class Saver:
        def save_images(self, **kwargs):
            calls.append(kwargs)
            return {"ui": {"images": [{"filename": "image.png", "type": "output"}]}}

    result = publish_media_output(
        images="image-batch",
        frame_rate=24.0,
        save_settings={"output_kind": "image", "filename_prefix": "Director"},
        metadata={"text_positive": "resolved prompt", "text_seed": 42},
        saver_factory=Saver,
    )

    assert result["ui"]["images"][0]["filename"] == "image.png"
    assert calls == [{
        "filename_prefix": "Director",
        "file_format": "png",
        "compression": 0,
        "save_output": True,
        "images": "image-batch",
        "metadata_config": {"text_positive": "resolved prompt", "text_seed": 42},
        "prompt": None,
        "extra_pnginfo": None,
    }]


def test_video_publication_reuses_enhanced_video_combine_with_automatic_metadata():
    from nodes.helper_media_output_v2 import publish_media_output

    calls = []

    class Combine:
        def combine(self, **kwargs):
            calls.append(kwargs)
            return {"ui": {"gifs": [{"filename": "clip.webm", "type": "output"}]}}

    result = publish_media_output(
        images="video-batch",
        frame_rate=24.0,
        save_settings={"output_kind": "video", "filename_prefix": "Director/video"},
        metadata={"text_positive": "resolved prompt", "text_seed": 42},
        combine_factory=Combine,
    )

    assert result["ui"]["gifs"][0]["filename"] == "clip.webm"
    assert calls == [{
        "images": "video-batch",
        "frame_rate": 24.0,
        "codec": "Auto",
        "container": "Auto",
        "bit_depth": "Auto",
        "quality": 20,
        "pingpong": False,
        "save_metadata": True,
        "filename_prefix": "Director/video",
        "save_output": True,
        "pass_frames": False,
        "crop_to_audio": False,
        "audio_codec": "Auto",
        "audio_bitrate": "192k",
        "save_first_frame": False,
        "save_last_frame": False,
        "audio": None,
        "prompt": None,
        "extra_pnginfo": None,
    }]
