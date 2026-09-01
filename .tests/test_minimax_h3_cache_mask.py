"""Per-token denoise-mask parity tests for the MiniMax H3 cache node.

These run the node's patched_forward against the LIVE Core (comfy.ldm.minimax.model)
and assert it honors denoise_mask / audio_denoise_mask exactly like Core's _forward.
They skip when Core is not importable (plain suite without PYTHONPATH=Core).

Run with Core:
    cd /data/GitHub/ComfyUI-DaSiWa-Nodes
    PYTHONPATH=/data/GitHub/ComfyUI /data/GitHub/ComfyUI/venv/bin/python -m pytest .tests/test_minimax_h3_cache_mask.py -v
"""
import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_minimax_h3_cache.py"

minimax_model = pytest.importorskip("comfy.ldm.minimax.model")


def _load_module():
    spec = importlib.util.spec_from_file_location("minimax_h3_cache_mask_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model():
    torch.manual_seed(3)
    model = minimax_model.MiniMaxH3Model(
        hidden_size=8, num_layers=1, token_refiner_num_layers=1,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
        latents_dim=2, audio_latents_dim=2, patch_size=(1, 2, 2),
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, rope_inv_freq_len=1,
        dtype=torch.float32, device=torch.device("cpu"), operations=torch.nn,
    )
    model.eval()
    model.requires_grad_(False)  # in-place RoPE is inference-only; scale params must be grad-free
    model.rope.inv_freq.data = torch.linspace(0.0, 1.0, model.rope.inv_freq.shape[0])
    return model


def _inputs():
    video = torch.randn(1, 2, 2, 4, 4)    # [B, C, T, H, W]
    audio = torch.randn(1, 2, 2, 2)       # [B, C, ch=2, T_audio]
    context = torch.randn(1, 4, 8)        # [B, text_len, text_dim]
    # sigma_v = 500/1000 = 0.5: a mid-sigma where preserved (m=0 -> t pinned ~0.999)
    # and generated (m=1 -> t_v=0.5) rows actually separate. At a near-clean sigma
    # (e.g. 0.5/1000) the clamp collapses both to t_v and the mask is invisible.
    timestep = torch.tensor([500.0])
    sigmas = torch.linspace(1.0, 0.0, 11)
    return video, audio, context, timestep, sigmas


class _CaptureLoop:
    """Identity block_loop replacement that records the mod_segments the node computed.

    Per-token rows are normalized to lists so segments are plain-data comparable.
    """

    def __init__(self):
        self.mod_segments = None

    def __call__(self, block_args, replacement_dict):
        self.mod_segments = [
            (a, b, r.tolist() if torch.is_tensor(r) else r)
            for a, b, r in block_args["mod_segments"]
        ]
        original = replacement_dict["original_block"]
        return {"img": original(block_args)["img"]}


def _run_capture(model, video, audio, context, timestep, sigmas, **masks):
    module = _load_module()
    capture = _CaptureLoop()
    opts = {"sample_sigmas": sigmas, "patches_replace": {"dit": {("block_loop", 0): capture}}}
    with torch.inference_mode():
        module.build_h3_block_loop_forward(True)(
            model, x=[video, audio], timestep=timestep, context=context,
            transformer_options=opts, minimax_payload={}, **masks,
        )
    return capture.mod_segments


def _list_rows(segs):
    """(start, end, row) entries whose row is a per-token list, in packed-sequence order."""
    return [(a, b, r) for a, b, r in segs if isinstance(r, list)]


def test_absent_and_full_masks_keep_scalar_segments():
    """Regression guard: no mask and all-generate masks must leave the no-mask mod_segments untouched."""
    model = _tiny_model()
    video, audio, context, timestep, sigmas = _inputs()
    base = _run_capture(model, video, audio, context, timestep, sigmas)
    assert all(isinstance(r, int) for _, _, r in base), "no-mask forward must use scalar mod rows"

    full_v = torch.ones(1, 1, 2, 4, 4)
    full_a = torch.ones(1, 1, 4)
    full = _run_capture(model, video, audio, context, timestep, sigmas,
                        denoise_mask=full_v, audio_denoise_mask=full_a)
    assert full == base, "all-generate masks must be invisible to the modulation"


def test_mixed_mask_emits_per_token_mod_rows():
    """A mask that mixes preserved/generated rows must yield per-token mod rows, not the shared scalar."""
    model = _tiny_model()
    video, audio, context, timestep, sigmas = _inputs()
    base = _run_capture(model, video, audio, context, timestep, sigmas)

    mask_v = torch.zeros(1, 1, 2, 4, 4)
    mask_v[0, 0, 1] = 1.0  # first latent frame preserved, second generated
    mask_a = torch.tensor([[[0.0, 0.0, 0.5, 1.0]]])
    mixed = _run_capture(model, video, audio, context, timestep, sigmas,
                        denoise_mask=mask_v, audio_denoise_mask=mask_a)

    rows = _list_rows(mixed)
    assert len(rows) == 2, "mixed video+audio mask must produce exactly two per-token rows (video, audio)"
    # packed order is [text | cond | audio | video] -> audio row precedes video row
    audio_seg, video_seg = rows
    assert len(set(video_seg[2])) > 1, "mixed video mask must map to more than one mod row"
    assert len(set(audio_seg[2])) > 1, "mixed audio mask must map to more than one mod row"
    assert mixed != base, "mixed mask must change the mod_segments"


def test_masked_forward_matches_core_forward():
    """The node with a mixed mask must reproduce Core's _forward output, and unmasked must too."""
    model = _tiny_model()
    video, audio, context, timestep, sigmas = _inputs()
    module = _load_module()
    patched = module.build_h3_block_loop_forward(True)
    opts = {"sample_sigmas": sigmas}

    mask_v = torch.zeros(1, 1, 2, 4, 4)
    mask_v[0, 0, 1] = 1.0
    mask_a = torch.tensor([[[0.0, 0.0, 0.5, 1.0]]])

    with torch.inference_mode():
        n_v, n_a = patched(model, x=[video, audio], timestep=timestep, context=context,
                           transformer_options=opts, minimax_payload={},
                           denoise_mask=mask_v, audio_denoise_mask=mask_a)
        c_v, c_a = model._forward([video, audio], timestep, context, transformer_options=opts,
                                  minimax_payload={}, denoise_mask=mask_v, audio_denoise_mask=mask_a)
    assert torch.isfinite(n_v).all() and torch.isfinite(n_a).all()
    assert torch.allclose(n_v, c_v, atol=1e-5), "masked node output diverges from Core"
    assert torch.allclose(n_a, c_a, atol=1e-5), "masked audio output diverges from Core"

    with torch.inference_mode():
        n_v0, n_a0 = patched(model, x=[video, audio], timestep=timestep, context=context,
                             transformer_options=opts, minimax_payload={})
        c_v0, c_a0 = model._forward([video, audio], timestep, context, transformer_options=opts,
                                     minimax_payload={})
    assert torch.allclose(n_v0, c_v0, atol=1e-5), "unmasked node output diverges from Core"
    assert torch.allclose(n_a0, c_a0, atol=1e-5), "unmasked audio output diverges from Core"
    assert not torch.allclose(n_v, c_v0, atol=1e-3), "mask must actually change the video output"
