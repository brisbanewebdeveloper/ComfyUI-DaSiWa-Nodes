"""CPU tests for the generic image Tiled Diffusion wrapper; model_function is a test double."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'ComfyUI'))
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")
spec = importlib.util.spec_from_file_location("dasiwa_tiled_diffusion", ROOT / "nodes/nodes_tiled_diffusion.py")
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)


@pytest.mark.parametrize("total,tile,overlap,expected", [
    (8, 8, 2, [0]), (4, 8, 2, [0]), (16, 8, 2, [0, 6, 8]), (14, 8, 2, [0, 6]),
    (13, 8, 2, [0, 5]), (9, 8, 0, [0, 1]), (16, 8, 0, [0, 8]),
])
def test_tile_starts_cover_without_overrun(total, tile, overlap, expected):
    starts = td.tile_starts(total, tile, overlap)
    assert starts == expected
    assert all(s + tile <= max(total, tile) for s in starts)
    assert starts[-1] + tile >= total


def test_weights_positive_and_symmetric():
    w = td.tile_weight(5, 8, torch.device("cpu"))
    assert w.shape == (1, 1, 5, 8) and w.dtype == torch.float32
    assert (w > 0).all()
    assert torch.equal(w, w.flip(-1)) and torch.equal(w, w.flip(-2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiles_blend_to_full_prediction_and_slice_spatial_conds(dtype):
    x = torch.linspace(-2, 2, 2 * 3 * 13 * 19, dtype=dtype).reshape(2, 3, 13, 19)
    mask = torch.rand(2, 1, 13, 19)
    concat = torch.rand(2, 4, 13, 19)
    options = {"sentinel": object()}
    c = dict(denoise_mask=mask, c_concat=concat, transformer_options=options, c_crossattn=torch.ones(2, 4, 8))
    args = dict(input=x, timestep=torch.ones(2), c=c, cond_or_uncond=[0, 1])
    original = x.clone()
    calls = []

    def double(t, timestep, **ct):
        calls.append(tuple(t.shape))
        assert ct["denoise_mask"].shape[-2:] == t.shape[-2:] and ct["denoise_mask"].shape[-2:] != (13, 19)
        assert ct["c_concat"].shape == (2, 4, *t.shape[-2:])
        assert ct["c_crossattn"] is c["c_crossattn"]
        assert ct["transformer_options"]["sentinel"] is options["sentinel"]
        ct["transformer_options"]["block_index"] = 1  # native forward writes here
        return t * 2

    wrapper = td.TiledDiffusion(64, 64, 16, 8)  # 8x8 latent tiles, 2 overlap
    out = wrapper(double, args)
    assert out.dtype == dtype and out.shape == x.shape
    torch.testing.assert_close(out, x * 2)
    assert len(calls) == len(td.tile_starts(13, 8, 2)) * len(td.tile_starts(19, 8, 2))
    assert "block_index" not in options
    assert torch.equal(original, x)
    assert all(not isinstance(v, torch.Tensor) for v in vars(wrapper).values())


def test_small_latent_calls_once_unchanged():
    x = torch.randn(1, 4, 8, 8)
    c = {"denoise_mask": torch.ones(1, 1, 8, 8)}
    calls = []

    def fn(t, timestep, **ct):
        calls.append((t, ct))
        return t * 2

    out = td.TiledDiffusion(64, 64, 16, 8)(fn, dict(input=x, timestep=torch.ones(1), c=c))
    assert len(calls) == 1 and calls[0][0] is x and calls[0][1]["denoise_mask"] is c["denoise_mask"]
    torch.testing.assert_close(out, x * 2)


def test_controlnet_and_bad_geometry_rejected():
    args = dict(input=torch.zeros(1, 4, 32, 32), timestep=torch.ones(1), c={"control": object()})
    with pytest.raises(ValueError, match="ControlNet"):
        td.TiledDiffusion(64, 64, 16, 8)(lambda t, ts, **c: t, args)
    with pytest.raises(ValueError):
        td.TiledDiffusion(64, 64, 64, 8)  # overlap not smaller than tile
    with pytest.raises(ValueError):
        td.TiledDiffusion(72, 64, 16, 16)  # not a multiple of the latent scale
    with pytest.raises(ValueError, match="4-D"):
        td.TiledDiffusion(64, 64, 16, 8)(lambda t, ts, **c: t, dict(input=torch.zeros(1, 4, 2, 32, 32), timestep=torch.ones(1), c={}))


def test_cancellation_checked_per_tile(monkeypatch):
    checks = []

    def interrupt():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("test cancellation")
    monkeypatch.setattr(td.comfy.model_management, "throw_exception_if_processing_interrupted", interrupt)
    calls = []
    args = dict(input=torch.zeros(1, 4, 16, 16), timestep=torch.ones(1), c={})
    with pytest.raises(RuntimeError, match="test cancellation"):
        td.TiledDiffusion(64, 64, 16, 8)(lambda t, ts, **c: calls.append(1) or t, args)
    assert len(calls) == 1


def test_node_patches_clone_with_model_latent_scale():
    class Patcher:
        def __init__(self):
            self.model = type("M", (), {"latent_format": type("F", (), {"spacial_downscale_ratio": 16})()})()
            self.wrapper = None

        def clone(self):
            return Patcher()

        def set_model_unet_function_wrapper(self, w):
            self.wrapper = w

    node = td.DaSiWaTiledDiffusion()
    src = Patcher()
    (patched,) = node.patch(src, 1024, 768, 128)
    assert patched is not src and src.wrapper is None
    assert (patched.wrapper.tile_w, patched.wrapper.tile_h, patched.wrapper.overlap) == (64, 48, 8)
    with pytest.raises(ValueError):
        node.patch(src, 1032, 768, 128)
