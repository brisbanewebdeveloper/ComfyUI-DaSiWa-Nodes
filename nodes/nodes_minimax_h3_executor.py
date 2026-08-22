"""Optional single-stage executor for DaSiWa MiniMax H3 Director guides."""

from __future__ import annotations

from typing import Any

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import torch
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_lt import LTXVSeparateAVLatent
from nodes import VAEDecode

from .helper_minimax_h3_director import normalize_guide
from .nodes_minimax_h3_director_guide import MiniMaxH3DirectorGuide, _native_node


def _unpack_node_output(output):
    if hasattr(output, "args") and output.args:
        return output.args
    if isinstance(output, (tuple, list)):
        return output
    raise RuntimeError(f"Unexpected node output type: {type(output)!r}")


def _empty_audio() -> dict[str, Any]:
    return {"waveform": torch.zeros(1, 1, 0), "sample_rate": 44100}


class MiniMaxH3DirectorExecutor:
    """Sample and decode one guide while leaving the planner independently composable."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "guide": ("MINIMAX_H3_DIRECTOR_GUIDE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {"audio_vae": ("VAE",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "LATENT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "samples", "report")
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/MiniMax H3"
    DESCRIPTION = (
        "Optional one-node sampling and decode path for a DaSiWa MiniMax H3 Director guide. "
        "Keep using Director Guide with external sampler nodes when custom sampling is needed."
    )

    def execute(
        self,
        model,
        clip,
        video_vae,
        guide,
        seed,
        steps,
        cfg,
        sampler,
        scheduler,
        shift_video,
        shift_audio,
        frame_rate,
        audio_vae=None,
    ):
        return _sample_and_decode(
            model, clip, video_vae, guide, seed, steps, cfg, sampler, scheduler,
            shift_video, shift_audio, frame_rate, audio_vae,
        )


def _sample_and_decode(
    model,
    clip,
    video_vae,
    guide,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    frame_rate: float,
    audio_vae=None,
    *,
    positive=None,
    latent=None,
):
    """Run one guide, optionally using sequence-prepared conditioning and latent."""
    state = normalize_guide(guide)
    if positive is None or latent is None:
        positive, latent = MiniMaxH3DirectorGuide().apply(clip, video_vae, guide, audio_vae)
    shifted = _unpack_node_output(
        _native_node("MiniMaxH3SigmaShift").execute(model, float(shift_video), float(shift_audio))
    )[0]

    latent_image = comfy.sample.fix_empty_latent_channels(
        shifted,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    noise = comfy.sample.prepare_noise(latent_image, int(seed), latent.get("batch_index"))
    callback = latent_preview.prepare_callback(shifted, int(steps))
    samples = comfy.sample.sample(
        shifted,
        noise,
        int(steps),
        float(cfg),
        sampler,
        scheduler,
        positive,
        [],
        latent_image,
        denoise=1.0,
        noise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(seed),
    )

    sampled = latent.copy()
    sampled.pop("downscale_ratio_spacial", None)
    sampled.pop("downscale_ratio_temporal", None)
    sampled["samples"] = samples

    video_latent, audio_latent = _unpack_node_output(
        LTXVSeparateAVLatent.execute(sampled)
    )[:2]
    images, = VAEDecode().decode(video_vae, video_latent)
    if audio_vae is None:
        audio = _empty_audio()
    else:
        audio = _unpack_node_output(VAEDecodeAudio.execute(audio_vae, audio_latent))[0]

    frame_count = int(images.shape[0])
    report = (
        f"{state.mode}: {state.width}x{state.height}, {frame_count} frames at "
        f"{float(frame_rate):g} fps; {int(steps)} steps, {sampler}/{scheduler}"
    )
    return images, audio, float(frame_rate), frame_count, sampled, report


NODE_CLASS_MAPPINGS = {"DaSiWaMiniMaxH3DirectorExecutor": MiniMaxH3DirectorExecutor}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWaMiniMaxH3DirectorExecutor": "DaSiWa MiniMax H3 Director Executor"
}
