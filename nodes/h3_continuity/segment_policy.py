"""Pure global-timeline boundaries for H3 continuation segments."""


def render_layout(source_frames, extension_frames):
    delta = delta_layout(source_frames, extension_frames)
    # Core stitches 5-token/17-frame chunks. 51 frames also give integral 40-Hz
    # audio boundaries; >=102 original frames exceed BigVGAN's local receptive field.
    offset = max(0, (source_frames - 102) // 51 * 51)
    return dict(frame_offset=offset, video_start=offset // 17 * 5,
                audio_start=round(offset * 40 / 24),
                render_frames=delta['frames'] - offset,
                crop_start_frame=source_frames - offset,
                crop_frames=extension_frames,
                crop_start_sample=round(source_frames * 32000 / 24) - round(offset * 32000 / 24),
                crop_samples=round(delta['frames'] * 32000 / 24) - round(source_frames * 32000 / 24))


def delta_layout(source_frames, extension_frames):
    if type(source_frames) is not int or source_frames < 5 or (source_frames - 5) % 17:
        raise ValueError("Source frames must satisfy the H3 17k+5 grid.")
    if type(extension_frames) is not int or not 17 <= extension_frames <= 357 or extension_frames % 17:
        raise ValueError("Added frames must be a multiple of 17 between 17 and 357.")
    total = source_frames + extension_frames
    audio_start = round(source_frames * 40 / 24)
    return dict(segment_start_frame=source_frames, segment_frames=extension_frames,
                frames=total, video_token_start=(source_frames - 5) // 17 * 5 + 2,
                video_token_count=extension_frames // 17 * 5,
                audio_token_start=audio_start, audio_token_count=round(total * 40 / 24) - audio_start)
