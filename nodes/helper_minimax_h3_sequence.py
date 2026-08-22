"""Typed sequence plans and AV continuity operations for MiniMax H3."""

from __future__ import annotations

import json
from typing import Any

import torch
import torchaudio
from comfy.nested_tensor import NestedTensor

from .helper_minimax_h3_director import (
    AUDIO_LATENT_FPS,
    FPS,
    align_frame_count,
    normalize_guide,
    video_latent_t,
)

CONTEXT_FRAME_CHOICES = (5, 22, 39, 56)


def sequence_segment(
    guide: dict,
    run: bool,
    cache_key: str,
    seed_offset: int,
    source_images=None,
    source_audio=None,
) -> dict[str, Any]:
    """Build one independently selectable sequence segment."""
    normalize_guide(guide)
    key = str(cache_key or "").strip()
    if len(key) > 128:
        raise ValueError("MiniMax H3 sequence cache_key must be at most 128 characters")
    return {
        "version": 1,
        "guide": guide,
        "run": bool(run),
        "cache_key": key,
        "seed_offset": int(seed_offset),
        "source_images": source_images,
        "source_audio": source_audio,
    }


def sequence_plan(segments) -> dict[str, Any]:
    """Validate an ordered set of segments and return the executor plan."""
    rows = list(segments or [])
    if not rows:
        raise ValueError("MiniMax H3 sequence needs at least one segment")
    canvas = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("version") != 1:
            raise ValueError(f"MiniMax H3 sequence segment {index + 1} is invalid")
        state = normalize_guide(row.get("guide"))
        current = (state.width, state.height)
        if canvas is None:
            canvas = current
        elif current != canvas:
            raise ValueError(
                "MiniMax H3 sequence segments must share one canvas; "
                f"segment {index + 1} is {current[0]}x{current[1]}, expected {canvas[0]}x{canvas[1]}"
            )
    return {"version": 1, "segments": rows, "width": canvas[0], "height": canvas[1]}


def sequence_summary(plan: dict) -> str:
    """Serialize the execution-relevant, tensor-free portion of a plan."""
    rows = []
    for index, segment in enumerate(validate_sequence_plan(plan)):
        state = normalize_guide(segment["guide"])
        rows.append({
            "index": index + 1,
            "mode": state.mode,
            "frames": state.length,
            "run": segment["run"],
            "cache_key": segment["cache_key"],
            "seed_offset": segment["seed_offset"],
            "source": segment.get("source_images") is not None,
        })
    return json.dumps({"version": 1, "segments": rows}, ensure_ascii=False)


def validate_sequence_plan(plan: dict) -> list[dict]:
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("MiniMax H3 sequence plan is invalid")
    rows = plan.get("segments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("MiniMax H3 sequence plan has no segments")
    return rows


def snap_context_frames(requested: int, available: int) -> int:
    """Choose the largest supported H3 context window that fits the prior segment."""
    choices = [value for value in CONTEXT_FRAME_CHOICES if value <= int(available)]
    if not choices:
        raise ValueError("MiniMax H3 continuity needs at least 5 previous frames")
    requested = int(requested)
    return min(choices, key=lambda value: (abs(value - requested), -value))


def extend_guide(guide: dict, overlap_frames: int) -> tuple[dict, int]:
    """Extend a guide so trimming its inherited prefix keeps the requested body."""
    body_frames = normalize_guide(guide).length
    extended = dict(guide)
    extended["length"] = align_frame_count(body_frames + int(overlap_frames))
    return extended, body_frames


def _av_streams(latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        samples = latent["samples"]
    except (KeyError, TypeError) as exc:
        raise ValueError("MiniMax H3 continuity expects a LATENT dictionary") from exc
    if not getattr(samples, "is_nested", False):
        raise ValueError("MiniMax H3 continuity expects nested AV samples")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError("MiniMax H3 continuity expects exactly two AV streams")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("MiniMax H3 continuity received invalid AV stream shapes")
    return video, audio


def latent_to_cpu(latent: dict | None) -> dict | None:
    """Detach a sampled AV latent so only a CPU handoff survives the segment."""
    if latent is None:
        return None
    video, audio = _av_streams(latent)
    return {
        "samples": NestedTensor((
            video.detach().cpu().contiguous(),
            audio.detach().cpu().contiguous(),
        ))
    }


def prepare_continuation(target: dict, source_cpu: dict, overlap_frames: int) -> dict:
    """Pin a synchronized source AV tail at the start of a fresh target latent."""
    target_video, target_audio = _av_streams(target)
    source_video, source_audio = _av_streams(source_cpu)
    if source_video.shape[3:] != target_video.shape[3:]:
        raise ValueError("MiniMax H3 continuity requires matching segment canvases")

    video_steps = video_latent_t(int(overlap_frames))
    audio_steps = round(int(overlap_frames) / FPS * AUDIO_LATENT_FPS)
    if source_video.shape[2] < video_steps or source_audio.shape[-1] < audio_steps:
        raise ValueError("MiniMax H3 continuity source is shorter than the selected context")
    if target_video.shape[2] < video_steps or target_audio.shape[-1] < audio_steps:
        raise ValueError("MiniMax H3 continuity target is shorter than the selected context")

    target_video[:, :, :video_steps].copy_(
        source_video[:, :, -video_steps:].to(target_video)
    )
    target_audio[..., :audio_steps].copy_(
        source_audio[..., -audio_steps:].to(target_audio)
    )
    video_mask = torch.ones(
        (target_video.shape[0], 1, target_video.shape[2], 1, 1),
        dtype=torch.float32,
        device=target_video.device,
    )
    audio_mask = torch.ones(
        (target_audio.shape[0], 1, 1, target_audio.shape[-1]),
        dtype=torch.float32,
        device=target_audio.device,
    )
    video_mask[:, :, :video_steps] = 0.0
    audio_mask[..., :audio_steps] = 0.0
    output = {key: value for key, value in target.items() if key not in {"samples", "noise_mask"}}
    output["samples"] = NestedTensor((target_video, target_audio))
    output["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return output


def audio_to_cpu(audio: dict | None) -> dict:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        return {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 32000}
    return {
        "waveform": audio["waveform"].detach().cpu().contiguous(),
        "sample_rate": int(audio.get("sample_rate") or 32000),
    }


def trim_segment(images, audio: dict, overlap_frames: int, body_frames: int, fps: float):
    """Remove the inherited prefix and align video/audio to the planned body."""
    start = int(overlap_frames)
    end = start + int(body_frames)
    if int(images.shape[0]) < end:
        raise ValueError("MiniMax H3 segment decode is shorter than its planned body")
    trimmed_images = images[start:end].detach().cpu().contiguous()
    audio = audio_to_cpu(audio)
    sample_rate = audio["sample_rate"]
    audio_start = round(start / float(fps) * sample_rate)
    audio_length = round(int(body_frames) / float(fps) * sample_rate)
    waveform = audio["waveform"]
    trimmed_audio = waveform[..., audio_start:audio_start + audio_length].contiguous()
    return trimmed_images, {"waveform": trimmed_audio, "sample_rate": sample_rate}


def concat_images(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("MiniMax H3 sequence produced no images")
    shape = tuple(parts[0].shape[1:])
    if any(tuple(part.shape[1:]) != shape for part in parts[1:]):
        raise ValueError("MiniMax H3 sequence image canvases do not match")
    return torch.cat(parts, dim=0)


def concat_audio(parts: list[dict]) -> dict:
    usable = [audio_to_cpu(part) for part in parts]
    usable = [part for part in usable if part["waveform"].shape[-1] > 0]
    if not usable:
        return {"waveform": torch.zeros((1, 1, 0)), "sample_rate": 32000}
    target_rate = usable[0]["sample_rate"]
    channels = max(int(part["waveform"].shape[1]) for part in usable)
    waveforms = []
    for part in usable:
        waveform = part["waveform"]
        if part["sample_rate"] != target_rate:
            waveform = torchaudio.functional.resample(waveform, part["sample_rate"], target_rate)
        if waveform.shape[1] == 1 and channels > 1:
            waveform = waveform.expand(-1, channels, -1)
        elif waveform.shape[1] != channels:
            raise ValueError("MiniMax H3 sequence audio channel counts do not match")
        waveforms.append(waveform)
    return {"waveform": torch.cat(waveforms, dim=-1), "sample_rate": target_rate}
