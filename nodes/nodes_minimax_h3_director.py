"""MiniMax H3 Director guide node."""
import json

from .helper_logging import log_dasiwa
from .helper_minimax_h3_director import (
    align_frame_count, assemble_prompt, audio_duration, load_audio,
    load_embedded_video_audio, load_image, load_video, normalize_guide,
    scale_input_media, validate_reference_limits,
)
from .helper_minimax_h3_prompt_builder import (
    build_prompt, default_builder_state, migrate_legacy_prompt, normalize_ref_schema,
    validate_builder_state,
)

BASE_MODES = {"T2VA", "I2VA", "FL2VA", "L2VA"}


def _describe_model(model) -> str:
    if model is None:
        return "none"
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__name__}"


class MiniMaxH3Director:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA", "Image Inpaint"], {"default": "FL2VA"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1344, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 768, "min": 16, "max": 8192, "step": 16}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "timeline_data": ("STRING", {"default": "{\"version\":1,\"items\":[],\"prompt_blocks\":[]}", "multiline": False, "hidden": True}),
                "builder_state": ("STRING", {"default": "", "multiline": False, "hidden": True}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
            },
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True}),
                "ref2va_model": ("MODEL", {"lazy": True}),
                "external_width_overwrite": ("INT", {"min": 1, "max": 8192, "step": 1, "forceInput": True}),
                "external_height_overwrite": ("INT", {"min": 1, "max": 8192, "step": 1, "forceInput": True}),
                "external_prompt_overwrite": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_DIRECTOR_GUIDE", "INT", "STRING", "INT", "INT", "MODEL", "BOOLEAN", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("guide", "duration", "positive_prompt", "width", "height", "model", "fl2va_requested", "inpaint_requested", "frame_rate")
    FUNCTION = "build_guide"
    CATEGORY = "DaSiWa/MiniMax H3"

    def check_lazy_status(self, mode, prompt, width, height, duration, ref_image_size, timeline_data, builder_state,
                          fl2va_model=None, ref2va_model=None, external_width_overwrite=None,
                          external_height_overwrite=None, external_prompt_overwrite=None, frame_rate=24.0):
        selected_name = "ref2va_model" if mode == "REF2VA" else "fl2va_model"
        selected_model = ref2va_model if mode == "REF2VA" else fl2va_model
        return [selected_name] if selected_model is None else []

    def build_guide(self, mode, prompt, width, height, duration, ref_image_size, timeline_data, builder_state="",
                    fl2va_model=None, ref2va_model=None, external_width_overwrite=None,
                    external_height_overwrite=None, external_prompt_overwrite=None, frame_rate=24.0):
        # Preserve direct Python callers that used the pre-builder positional model argument.
        if builder_state is not None and not isinstance(builder_state, str):
            if fl2va_model is None:
                fl2va_model = builder_state
                builder_state = ""
            else:
                raise ValueError("builder_state must be JSON text")
        if mode not in BASE_MODES | {"REF2VA", "Image Inpaint"}:
            raise ValueError(f"unsupported MiniMax Director mode: {mode}")
        # A non-numeric frame_rate (e.g. a stale 9th widgets_value shifted in by an
        # older save, or an empty string) falls back to the default instead of crashing
        # the queue; genuinely out-of-range numbers still raise.
        try:
            frame_rate = float(frame_rate)
        except (TypeError, ValueError):
            frame_rate = 24.0
        if not 0.1 <= frame_rate <= 240.0:
            raise ValueError("MiniMax Director frame_rate must be between 0.1 and 240")
        external_canvas = external_width_overwrite is not None or external_height_overwrite is not None
        if external_canvas:
            if external_width_overwrite is None or external_height_overwrite is None:
                raise ValueError("both external width overwrite and external height overwrite are required")
            width, height = int(external_width_overwrite), int(external_height_overwrite)
            if width < 1 or height < 1:
                raise ValueError("external width overwrite and external height overwrite must be positive")
        length = align_frame_count(int(duration) * 24)
        try:
            state = json.loads(timeline_data or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"MiniMax Director timeline_data is invalid JSON: {exc}") from exc
        if not isinstance(state, dict):
            raise ValueError("MiniMax Director timeline_data must contain an object")
        input_scaling = "Off" if external_canvas else (state.get("resolution") or {}).get("input_scaling", "Auto")
        try:
            builder = json.loads(builder_state) if builder_state else state.get("builder_state", {})
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"MiniMax Director builder_state is invalid JSON: {exc}") from exc
        if not isinstance(builder, dict):
            builder = {}
        merged = default_builder_state(mode)
        merged.update(builder)
        merged["ref"] = {**default_builder_state(mode)["ref"], **(builder.get("ref") or {})}
        normalize_ref_schema(merged["ref"])
        merged["mode"] = mode
        merged["duration"] = duration
        migrated_legacy_prompt = migrate_legacy_prompt(merged, state, prompt)

        items = sorted(enumerate(state.get("items", [])), key=lambda pair: (int(pair[1].get("order", pair[0])), pair[0]))
        items = [pair for pair in items if pair[1].get("enabled", True)]
        first_frame = last_frame = None
        ref_images, ref_videos, ref_video_audios, ref_audios = {}, {}, {}, {}
        images, videos, audios = [], [], []
        try:
            import folder_paths
            input_directory = folder_paths.get_input_directory()
        except (ImportError, AttributeError):
            input_directory = None

        if mode == "Image Inpaint":
            image_items = [pair for pair in items if pair[1].get("type") == "image"]
            incompatible_items = [pair[1].get("type") for pair in items if pair[1].get("type") != "image"]
            if incompatible_items:
                raise ValueError("Image Inpaint accepts image references only; video and audio references are not supported")
            if len(image_items) != 1:
                raise ValueError("Image Inpaint requires exactly one enabled image reference")
            value = image_items[0][1].get("value", image_items[0][1].get("tensor"))
            if isinstance(value, str) and input_directory:
                value = load_image(value, input_directory)
            first_frame = scale_input_media(value, input_scaling, width, height)
            last_frame = None
            length = 5

        if mode in BASE_MODES:
            image_items = sorted((pair for pair in items if pair[1].get("type") == "image"), key=lambda pair: (pair[1].get("slot", pair[0]), pair[0]))
            if mode == "T2VA":
                image_items = []
            elif mode in {"I2VA", "L2VA"}:
                image_items = image_items[:1]
            else:
                image_items = image_items[:2]
            for index, (_, item) in enumerate(image_items):
                value = item.get("value", item.get("tensor"))
                if isinstance(value, str) and input_directory:
                    value = load_image(value, input_directory)
                value = scale_input_media(value, input_scaling, width, height)
                if mode == "I2VA":
                    first_frame = value
                elif mode == "L2VA" or (mode == "FL2VA" and item.get("slot", index) == 1):
                    last_frame = value
                else:
                    first_frame = value
        elif mode != "Image Inpaint":
            type_order = {"image": 0, "video": 1, "audio": 2}
            for _, item in sorted(items, key=lambda pair: (type_order.get(pair[1].get("type"), 3), pair[1].get("slot", pair[0]), pair[0])):
                kind, value = item.get("type"), item.get("value", item.get("tensor"))
                if value is None:
                    continue
                trim_start = float(item.get("trim_start", 0))
                trim_end = item.get("trim_end")
                trim_end = float(trim_end) if trim_end is not None else None
                video_mode = item.get("media_mode", "video")
                if kind == "image":
                    value = load_image(value, input_directory) if isinstance(value, str) and input_directory else value
                    value = scale_input_media(value, input_scaling, width, height)
                    ref_images[f"ref_image_{len(ref_images) + 1}"] = value
                    images.append(item)
                elif kind == "audio":
                    value = load_audio(value, input_directory, trim_start=trim_start, trim_end=trim_end) if isinstance(value, str) and input_directory else value
                    ref_audios[f"ref_audio_{len(ref_audios) + 1}"] = value
                    audios.append({**item, "duration": audio_duration(value) if isinstance(value, dict) else item.get("duration")})
                elif kind == "video":
                    if video_mode not in {"video", "audio", "video_audio"}:
                        raise ValueError(f"unsupported video media mode: {video_mode}")
                    if video_mode in {"video", "video_audio"}:
                        video = load_video(value, input_directory, trim_start=trim_start, trim_end=trim_end) if isinstance(value, str) and input_directory else value
                        video = scale_input_media(video, input_scaling, width, height)
                        ref_videos[f"ref_video_{len(ref_videos) + 1}"] = video
                        video_duration = float(video.shape[0]) / 24.0 if hasattr(video, "shape") else item.get("duration")
                        videos.append({**item, "duration": video_duration})
                    if video_mode in {"audio", "video_audio"}:
                        audio = load_embedded_video_audio(value, input_directory, trim_start=trim_start, trim_end=trim_end) if isinstance(value, str) and input_directory else item.get("audio")
                        if video_mode == "video_audio":
                            ref_video_audios[f"ref_video_audio_{len(ref_videos)}"] = audio
                        else:
                            ref_audios[f"ref_audio_{len(ref_audios) + 1}"] = audio
                        audios.append({**item, "duration": audio_duration(audio) if isinstance(audio, dict) else item.get("duration")})
                    attached_audio = item.get("audio")
                    if attached_audio is not None and video_mode not in {"audio", "video_audio"}:
                        if isinstance(attached_audio, str) and input_directory:
                            attached_audio = load_audio(attached_audio, input_directory, trim_start=trim_start, trim_end=trim_end)
                        if video_mode == "video" and ref_videos:
                            ref_video_audios[f"ref_video_audio_{len(ref_videos)}"] = attached_audio
                        else:
                            ref_audios[f"ref_audio_{len(ref_audios) + 1}"] = attached_audio
                        audios.append({**item, "duration": audio_duration(attached_audio) if isinstance(attached_audio, dict) else item.get("duration")})
            validate_reference_limits(images=images, videos=videos, audios=audios)

        blocks = state.get("prompt_blocks", [])
        if isinstance(external_prompt_overwrite, str) and external_prompt_overwrite.strip():
            resolved = external_prompt_overwrite
        else:
            resolved = build_prompt(merged)
            if (not migrated_legacy_prompt and
                    not any(str(merged.get(key) or "").strip() for key in ("imd", "soundscape")) and
                    mode != "REF2VA"):
                resolved = assemble_prompt(prompt, blocks)
        for issue in validate_builder_state(merged):
            log_dasiwa("MiniMax H3 Director", f"[{issue['level'].upper()}] {issue['msg']}")
        guide = {
            "version": 2, "mode": mode, "prompt": prompt, "prompt_blocks": blocks, "resolved_prompt": resolved,
            "width": width, "height": height, "length": length, "ref_image_size": ref_image_size, "input_scaling": input_scaling,
            "first_frame": first_frame, "last_frame": last_frame, "ref_images": ref_images, "ref_videos": ref_videos,
            "ref_video_audios": ref_video_audios, "ref_audios": ref_audios, "builder_state": merged,
            "timeline": [{key: item.get(key) for key in ("id", "type", "start", "duration", "order", "trim_start", "trim_end") if key in item} for _, item in items],
            "prompt_payload": {"mode": mode, "full_prompt": resolved, "is_ref_mode": mode == "REF2VA", "subject_definitions": merged["ref"]["subject_defs"], "summary": merged["ref"]["summary_text"], "retention_analysis": merged["ref"]["retention"], "detailed_description": {"style_line": merged["ref"]["style_line"], "detail": merged["ref"]["detail"]}, "overall_soundscape": merged["ref"]["soundscape"] if mode == "REF2VA" else merged["soundscape"], "non_diegetic_music": merged["ref"]["music"] if mode == "REF2VA" else merged["music"], "imd": merged.get("imd", ""), "p2_shot": merged.get("p2_shot", ""), "last_shot": merged.get("last_shot", "")},
        }
        normalize_guide(guide)
        selected_model = ref2va_model if mode == "REF2VA" else fl2va_model
        log_dasiwa("MiniMax H3 Director", f"mode={mode}; requested_model={'ref2va_model' if mode == 'REF2VA' else 'fl2va_model'}; passed_model={_describe_model(selected_model)}; canvas={width}x{height}; frames={length}; fps={frame_rate}; refs=images:{len(ref_images)},videos:{len(ref_videos)},video_audio:{len(ref_video_audios)},audio:{len(ref_audios)}; timeline_items={len(items)}")
        return guide, length, resolved, int(width), int(height), selected_model, mode in BASE_MODES or mode == "Image Inpaint", mode == "Image Inpaint", frame_rate


NODE_CLASS_MAPPINGS = {"DaSiWaMiniMaxH3Director": MiniMaxH3Director}
NODE_DISPLAY_NAME_MAPPINGS = {"DaSiWaMiniMaxH3Director": "DaSiWa MiniMax H3 Director"}
