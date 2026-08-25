"""Director-owned native MiniMax H3 Image Inpaint execution."""
from copy import deepcopy

DEFAULT_POSTPROCESS_RECIPE = (
    {"id": "frame_interpolation", "enabled": False, "factor": 2, "model": "rife_v4.26.safetensors"},
    {"id": "torch_resize", "enabled": False, "size_mode": "Multiplier", "scale_multiplier": 2, "interpolation": "Lanczos"},
    {"id": "model_upscale", "enabled": False, "model_name": "2x-AnimeSharpV4_RCAN.safetensors"},
    {"id": "rtx_refiner", "enabled": False, "denoise": False, "denoise_quality": "Ultra", "deblur": False, "deblur_quality": "Ultra", "upscale": "VSR", "upscale_quality": "Ultra", "resize_type": "Scale", "scale": 2.0, "megapixels": 2.0, "width": 1920, "height": 1080, "divisible_by": "8", "ratio_preset": "16:9", "resize_method": "Center Crop (Fill)", "device_id": 0, "empty_cache": False, "use_mmap": False, "auto_unload_models": True},
    {"id": "watermark", "enabled": False, "watermark_path": "", "position": "bottom-right"},
)


def normalize_postprocess_recipe(recipe):
    supplied = {stage.get("id"): stage for stage in recipe or () if isinstance(stage, dict)}
    normalized = []
    for default in DEFAULT_POSTPROCESS_RECIPE:
        stage = deepcopy(default)
        candidate = supplied.get(stage["id"], {})
        for key, value in candidate.items():
            if key in stage:
                stage[key] = value
        stage["enabled"] = bool(stage["enabled"])
        normalized.append(stage)
    return tuple(normalized)


def _first(result):
    return result[0]


def _apply_model_patches(model, clip, settings):
    from nodes import LoraLoader
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    lora = settings.get("lora") or {}
    if lora.get("enabled"):
        model, clip = LoraLoader().load_lora(
            model, clip, lora["name"], float(lora.get("strength_model", 1.0)),
            float(lora.get("strength_clip", 1.0)),
        )
    if settings.get("comfy_kitchen_attention", False):
        from .nodes_comfy_kitchen_attention import PathchComfyKitchenAttentionDaSiWa
        model = _first(PathchComfyKitchenAttentionDaSiWa().patch(model))
    cache = settings.get("cache") or {}
    if cache.get("enabled"):
        from .nodes_minimax_h3_cache import MiniMaxH3Cache
        model = _first(MiniMaxH3Cache().patch(
            model, float(cache.get("reuse_threshold", 0.05)), float(cache.get("start_percent", 0.15)),
            float(cache.get("end_percent", 0.90)), int(cache.get("max_steps", 2)),
            cache.get("device", "auto"), bool(cache.get("verbose", False)),
        ))
    return _first(MiniMaxH3SigmaShift.execute(
        model, float(settings.get("shift_video", 11.0)), float(settings.get("shift_audio", 4.0)),
    )), clip


def _postprocess(images, recipe):
    for stage in recipe:
        if not stage["enabled"]:
            continue
        stage_id = stage["id"]
        if stage_id == "torch_resize":
            from .nodes_scaling import DaSiWa_TorchResize
            images = _first(DaSiWa_TorchResize().resize(
                images, stage["size_mode"], "Fit", 1024, 1024, float(stage["scale_multiplier"]),
                stage["interpolation"], True, 16, "0, 0, 0", "center", 0, 16.0, 64,
            ))
        elif stage_id == "model_upscale":
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
            model = _first(UpscaleModelLoader().load_model(stage["model_name"]))
            images = _first(ImageUpscaleWithModel().upscale(model, images))
        elif stage_id == "rtx_refiner":
            from .nodes_rtx_upscaler_refiner import DaSiWa_RTX_UpscalerRefiner
            images = _first(DaSiWa_RTX_UpscalerRefiner().execute(
                images,
                stage["denoise"], stage["denoise_quality"],
                stage["deblur"], stage["deblur_quality"],
                stage["upscale"], stage["upscale_quality"],
                stage["resize_type"], stage["scale"], stage["megapixels"],
                stage["width"], stage["height"], stage["divisible_by"],
                stage["ratio_preset"], stage["resize_method"], stage["device_id"],
                empty_cache=stage["empty_cache"],
                use_mmap=stage["use_mmap"],
                auto_unload_models=stage["auto_unload_models"],
            ))
        elif stage_id == "watermark":
            from .nodes_watermark import DaSiWa_Watermark
            if not stage["watermark_path"]:
                raise ValueError("Watermark postprocess requires watermark_path")
            images = _first(DaSiWa_Watermark().apply_watermark(
                images, stage["watermark_path"], stage["position"], 0.12, "lanczos", 1.0,
                0.0, 20, 20, False, 0.4,
            ))
        elif stage_id == "frame_interpolation":
            raise ValueError("Frame interpolation is unavailable for Image Inpaint's single-frame result")
    return images


def execute_image_inpaint(guide, model, clip, vae, seed, settings=None):
    """Run H3 at its valid five-frame minimum and return only decoded frame zero."""
    if guide.get("mode") != "Image Inpaint":
        raise ValueError("Director internal execution currently supports Image Inpaint only")
    if guide.get("first_frame") is None:
        raise ValueError("Image Inpaint requires one image keyframe")
    settings = settings or {}
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    from nodes import VAEDecode

    model, clip = _apply_model_patches(model, clip, settings)
    from .helper_minimax_h3_director_preview_v2 import attach_step_preview
    attach_step_preview(
        model, settings,
        preview_tiny_vae=settings.get("preview_tiny_vae"),
        preview_vae=settings.get("preview_vae"),
        unique_id=settings.get("unique_id"),
        client_id=settings.get("_client_id"),
    )
    positive, latent = MiniMaxH3ImageToVideo.execute(
        clip, vae, guide["resolved_prompt"], int(guide["width"]), int(guide["height"]), 5,
        guide["first_frame"], None,
    )
    guider = _first(BasicGuider.execute(model, positive))
    sampler = _first(KSamplerSelect.execute(settings.get("sampler", "res_multistep")))
    sigmas = _first(BasicScheduler.execute(model, settings.get("scheduler", "simple"), int(settings.get("steps", 25)), 1.0))
    noise = _first(RandomNoise.execute(int(seed)))
    sampled = _first(SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent))
    decoded_images = _first(VAEDecode().decode(vae, sampled))
    images = decoded_images if (settings.get("save") or {}).get("output_kind") == "video" else decoded_images[:1]
    return _postprocess(images, normalize_postprocess_recipe(settings.get("postprocess_recipe")))


def execute_h3(guide, model, clip, vae, audio_vae, seed, settings=None):
    """Execute every Director mode with native H3 conditioning and sampling."""
    settings = settings or {}
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
    from nodes import VAEDecode

    model, clip = _apply_model_patches(model, clip, settings)
    from .helper_minimax_h3_director_preview_v2 import attach_step_preview
    attach_step_preview(
        model, settings,
        preview_tiny_vae=settings.get("preview_tiny_vae"),
        preview_vae=settings.get("preview_vae"),
        unique_id=settings.get("unique_id"),
        client_id=settings.get("_client_id"),
    )
    length = 5 if guide.get("mode") == "Image Inpaint" else int(guide["length"])
    if guide.get("mode") == "REF2VA":
        if audio_vae is None:
            raise ValueError("REF2VA internal execution requires audio_vae")
        positive, latent = MiniMaxH3ReferenceToVideo.execute(
            clip, vae, audio_vae, guide["resolved_prompt"], int(guide["width"]), int(guide["height"]), length,
            guide.get("ref_image_size", "match"), guide.get("ref_images"), guide.get("ref_videos"),
            guide.get("ref_video_audios"), guide.get("ref_audios"),
        )
    else:
        positive, latent = MiniMaxH3ImageToVideo.execute(
            clip, vae, guide["resolved_prompt"], int(guide["width"]), int(guide["height"]), length,
            guide.get("first_frame"), guide.get("last_frame"),
        )
    guider = _first(BasicGuider.execute(model, positive))
    sampler = _first(KSamplerSelect.execute(settings.get("sampler", "res_multistep")))
    sigmas = _first(BasicScheduler.execute(model, settings.get("scheduler", "simple"), int(settings.get("steps", 25)), 1.0))
    sampled = _first(SamplerCustomAdvanced.execute(_first(RandomNoise.execute(int(seed))), guider, sampler, sigmas, latent))
    images = _first(VAEDecode().decode(vae, sampled))
    if guide.get("mode") == "Image Inpaint":
        images = images[:1]
    audio = None
    if audio_vae is not None:
        av_samples = sampled.get("samples") if isinstance(sampled, dict) else None
        if getattr(av_samples, "is_nested", False):
            audio_latent = av_samples.unbind()[-1]
            waveform = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
            audio = {"waveform": waveform, "sample_rate": getattr(audio_vae, "audio_sample_rate", 32000)}
    return _postprocess(images, normalize_postprocess_recipe(settings.get("postprocess_recipe"))), audio
