"""Shared image/video output publication via DaSiWa's existing saver nodes."""


def publish_media_output(
    images,
    frame_rate,
    save_settings,
    metadata,
    audio=None,
    prompt=None,
    extra_pnginfo=None,
    saver_factory=None,
    combine_factory=None,
):
    settings = save_settings or {}
    output_kind = settings.get("output_kind", "image")
    if output_kind == "video":
        if combine_factory is None:
            from .nodes_enhanced_video_combine import DaSiWa_EnhancedVideoCombine
            combine_factory = DaSiWa_EnhancedVideoCombine
        return combine_factory().combine(
            images=images,
            frame_rate=float(frame_rate),
            codec=settings.get("codec", "Auto"),
            container=settings.get("container", "Auto"),
            bit_depth=settings.get("bit_depth", "Auto"),
            quality=int(settings.get("quality", 20)),
            pingpong=bool(settings.get("pingpong", False)),
            save_metadata=bool(settings.get("save_workflow", True)),
            filename_prefix=settings.get("filename_prefix", "Director/video"),
            save_output=bool(settings.get("save_output", True)),
            pass_frames=False,
            crop_to_audio=bool(settings.get("crop_to_audio", False)),
            audio_codec=settings.get("audio_codec", "Auto"),
            audio_bitrate=settings.get("audio_bitrate", "192k"),
            save_first_frame=bool(settings.get("save_first_frame", False)),
            save_last_frame=bool(settings.get("save_last_frame", False)),
            audio=audio,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )

    if saver_factory is None:
        from .nodes_metadata import DaSiWa_MetadataImageSaver
        saver_factory = DaSiWa_MetadataImageSaver
    return saver_factory().save_images(
        filename_prefix=settings.get("filename_prefix", "Director"),
        file_format=settings.get("file_format", "png"),
        compression=int(settings.get("compression", 0)),
        save_output=bool(settings.get("save_output", True)),
        images=images,
        metadata_config=dict(metadata or {}),
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
    )
