"""Composable multi-segment execution nodes for DaSiWa MiniMax H3."""

from __future__ import annotations

import gc
import json

import comfy.model_management
import comfy.samplers
import torch
from comfy_api.latest import io

from .helper_logging import log_dasiwa
from .helper_minimax_h3_director import normalize_guide
from .helper_minimax_h3_sequence import (
    CONTEXT_FRAME_CHOICES,
    audio_to_cpu,
    concat_audio,
    concat_images,
    extend_guide,
    latent_to_cpu,
    prepare_continuation,
    sequence_plan,
    sequence_segment,
    sequence_summary,
    snap_context_frames,
    trim_segment,
    validate_sequence_plan,
)
from .helper_minimax_h3_sequence_cache import (
    load_segment,
    save_segment,
    segment_fingerprint,
)
from .nodes_minimax_h3_director_guide import MiniMaxH3DirectorGuide, _native_node
from .nodes_minimax_h3_executor import _sample_and_decode, _unpack_node_output

DirectorGuide = io.Custom("MINIMAX_H3_DIRECTOR_GUIDE")
DirectorSegment = io.Custom("DASIWA_MINIMAX_H3_SEGMENT")
DirectorSequence = io.Custom("DASIWA_MINIMAX_H3_SEQUENCE")
CACHE_MODES = ["off", "read_write", "read_only", "write_only", "refresh"]


class MiniMaxH3SequenceSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DaSiWaMiniMaxH3SequenceSegment",
            display_name="DaSiWa MiniMax H3 Sequence Segment",
            category="DaSiWa/MiniMax H3",
            description="Wrap one Director guide with run selection, cache identity, and optional source passthrough.",
            inputs=[
                DirectorGuide.Input("guide"),
                io.Boolean.Input("run", default=True),
                io.String.Input("cache_key", default="", tooltip="Required for disk cache. Change it when reference or source media changes."),
                io.Int.Input("seed_offset", default=0, min=-0x7FFFFFFF, max=0x7FFFFFFF),
                io.Image.Input("source_images", optional=True, tooltip="Used when run is off and no valid cache exists."),
                io.Audio.Input("source_audio", optional=True),
            ],
            outputs=[DirectorSegment.Output("segment")],
        )

    @classmethod
    def execute(cls, guide, run, cache_key, seed_offset, source_images=None, source_audio=None):
        return io.NodeOutput(sequence_segment(
            guide, run, cache_key, seed_offset, source_images, source_audio
        ))


class MiniMaxH3SequencePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            input=DirectorSegment.Input("segment"),
            prefix="segment_",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="DaSiWaMiniMaxH3SequencePlan",
            display_name="DaSiWa MiniMax H3 Sequence Plan",
            category="DaSiWa/MiniMax H3",
            description="Combine ordered Director segments into one explicit execution plan.",
            inputs=[io.Autogrow.Input("segments", template=template)],
            outputs=[DirectorSequence.Output("plan"), io.String.Output("plan_json")],
        )

    @classmethod
    def execute(cls, segments: io.Autogrow.Type):
        plan = sequence_plan(segments.values())
        return io.NodeOutput(plan, sequence_summary(plan))


def _context_audio(audio: dict | None, frames: int, fps: float) -> dict | None:
    audio = audio_to_cpu(audio)
    waveform = audio["waveform"]
    sample_count = round(int(frames) / float(fps) * audio["sample_rate"])
    if sample_count < 1 or waveform.shape[-1] < sample_count:
        return None
    return {"waveform": waveform[..., -sample_count:].contiguous(), "sample_rate": audio["sample_rate"]}


def _cleanup_segment_vram(enabled: bool) -> None:
    if not enabled:
        return
    gc.collect()
    try:
        comfy.model_management.cleanup_models_gc()
        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()
    except RuntimeError as exc:
        log_dasiwa("MiniMax H3 Sequence", f"VRAM cleanup skipped: {exc}")


def _segment_seed(base_seed: int, index: int, offset: int) -> int:
    return (int(base_seed) + int(index) + int(offset)) & 0xFFFFFFFFFFFFFFFF


class MiniMaxH3SequenceExecutor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DaSiWaMiniMaxH3SequenceExecutor",
            display_name="DaSiWa MiniMax H3 Sequence Executor",
            category="DaSiWa/MiniMax H3",
            description="Execute, resume, and concatenate an ordered MiniMax H3 AV sequence.",
            inputs=[
                DirectorSequence.Input("plan"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=25, min=1, max=200),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler", options=list(comfy.samplers.KSampler.SAMPLERS), default="res_multistep"),
                io.Combo.Input("scheduler", options=list(comfy.samplers.KSampler.SCHEDULERS), default="simple"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Boolean.Input("continuity", default=False),
                io.Combo.Input("context_frames", options=list(CONTEXT_FRAME_CHOICES), default=22),
                io.Combo.Input("cache_mode", options=CACHE_MODES, default="read_write"),
                io.Boolean.Input("clear_vram_between_segments", default=True),
                io.Model.Input("fl2va_model", optional=True, lazy=True),
                io.Model.Input("ref2va_model", optional=True, lazy=True),
                io.Vae.Input("audio_vae", optional=True),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Audio.Output("audio"),
                io.Float.Output("fps"),
                io.Int.Output("frame_count"),
                io.String.Output("report"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, plan, fl2va_model=None, ref2va_model=None, **_kwargs):
        rows = validate_sequence_plan(plan)
        needed = {normalize_guide(row["guide"]).mode for row in rows if row.get("run", True)}
        missing = []
        if any(mode != "REF2VA" for mode in needed) and fl2va_model is None:
            missing.append("fl2va_model")
        if "REF2VA" in needed and ref2va_model is None:
            missing.append("ref2va_model")
        return missing

    @classmethod
    def execute(
        cls,
        plan,
        clip,
        video_vae,
        seed,
        steps,
        cfg,
        sampler,
        scheduler,
        shift_video,
        shift_audio,
        continuity,
        context_frames,
        cache_mode,
        clear_vram_between_segments,
        fl2va_model=None,
        ref2va_model=None,
        audio_vae=None,
    ):
        frame_rate = 24.0
        rows = validate_sequence_plan(plan)
        hidden_id = None if cls.hidden is None else cls.hidden.unique_id
        node_id = str(hidden_id or "dasiwa_h3_sequence")
        image_parts, audio_parts, report_rows = [], [], []
        previous_latent = previous_images = previous_audio = None
        can_read = cache_mode in {"read_write", "read_only"}
        can_write = cache_mode in {"read_write", "write_only", "refresh"}

        for index, segment in enumerate(rows):
            guide = segment["guide"]
            state = normalize_guide(guide)
            segment_seed = _segment_seed(seed, index, segment.get("seed_offset", 0))
            use_continuity = bool(continuity and index > 0)
            fingerprint = segment_fingerprint(
                segment, index, seed=segment_seed, steps=steps, cfg=cfg,
                sampler=sampler, scheduler=scheduler, shift_video=shift_video,
                shift_audio=shift_audio, continuity=use_continuity,
                context_frames=context_frames,
            )
            cached = None
            if can_read:
                cached = load_segment(node_id, segment["cache_key"], index, fingerprint)

            if cached is not None:
                images, audio, sampled_cpu = cached["images"], cached["audio"], cached["latent"]
                origin = "cache"
            elif not segment.get("run", True):
                if segment.get("source_images") is None:
                    raise ValueError(
                        f"MiniMax H3 sequence segment {index + 1} is skipped but has no matching cache or source_images"
                    )
                images, audio = trim_segment(
                    segment["source_images"], segment.get("source_audio"), 0,
                    state.length, frame_rate,
                )
                sampled_cpu = None
                origin = "source"
            else:
                model = ref2va_model if state.mode == "REF2VA" else fl2va_model
                if model is None:
                    model_name = "ref2va_model" if state.mode == "REF2VA" else "fl2va_model"
                    raise ValueError(f"MiniMax H3 sequence segment {index + 1} needs {model_name}")
                if state.mode == "REF2VA" and audio_vae is None:
                    raise ValueError("audio_vae is required for REF2VA sequence segments")

                overlap = 0
                positive = target = None
                execution_guide = guide
                if use_continuity:
                    overlap = snap_context_frames(context_frames, int(previous_images.shape[0]))
                    execution_guide, _ = extend_guide(guide, overlap)
                    positive, target = MiniMaxH3DirectorGuide().apply(
                        clip, video_vae, execution_guide, audio_vae
                    )
                    if previous_latent is not None:
                        target = prepare_continuation(target, previous_latent, overlap)
                    else:
                        context_audio = _context_audio(previous_audio, overlap, frame_rate)
                        positive = _unpack_node_output(_native_node("MiniMaxH3AddGuide").execute(
                            positive, target, 0, vae=video_vae, audio_vae=audio_vae,
                            image=previous_images[-overlap:], audio=context_audio,
                        ))[0]

                result = _sample_and_decode(
                    model, clip, video_vae, execution_guide, segment_seed, steps,
                    cfg, sampler, scheduler, shift_video, shift_audio, frame_rate,
                    audio_vae, positive=positive, latent=target,
                )
                images, audio = trim_segment(result[0], result[1], overlap, state.length, frame_rate)
                sampled_cpu = latent_to_cpu(result[4])
                origin = "generated"
                if can_write and segment["cache_key"]:
                    save_segment(
                        node_id, segment["cache_key"], index, fingerprint,
                        images, audio, sampled_cpu,
                    )
                del result, positive, target
                _cleanup_segment_vram(clear_vram_between_segments)

            images = images.detach().cpu().contiguous()
            audio = audio_to_cpu(audio)
            image_parts.append(images)
            audio_parts.append(audio)
            previous_images, previous_audio, previous_latent = images, audio, sampled_cpu
            report_rows.append({
                "segment": index + 1,
                "mode": state.mode,
                "origin": origin,
                "frames": int(images.shape[0]),
                "seed": segment_seed,
            })

        images = concat_images(image_parts)
        audio = concat_audio(audio_parts)
        report = json.dumps({"segments": report_rows, "frame_count": int(images.shape[0])}, ensure_ascii=False)
        return io.NodeOutput(images, audio, float(frame_rate), int(images.shape[0]), report)


NODE_CLASS_MAPPINGS = {
    "DaSiWaMiniMaxH3SequenceSegment": MiniMaxH3SequenceSegment,
    "DaSiWaMiniMaxH3SequencePlan": MiniMaxH3SequencePlan,
    "DaSiWaMiniMaxH3SequenceExecutor": MiniMaxH3SequenceExecutor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWaMiniMaxH3SequenceSegment": "DaSiWa MiniMax H3 Sequence Segment",
    "DaSiWaMiniMaxH3SequencePlan": "DaSiWa MiniMax H3 Sequence Plan",
    "DaSiWaMiniMaxH3SequenceExecutor": "DaSiWa MiniMax H3 Sequence Executor",
}
