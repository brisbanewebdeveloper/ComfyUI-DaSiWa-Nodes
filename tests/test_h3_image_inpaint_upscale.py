"""Native five-frame I2V + real CPU H3 sampling/upscale networks.

Small randomly initialized checkpoints and CLIP/VAE doubles test execution,
not pretrained image quality. No GPU or live model reclamation is used.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'ComfyUI'))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_base
import comfy.model_management as mm
import comfy.model_patcher
import comfy.supported_models
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

PACKAGE = '_dasiwa_h3_inpaint_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'nodes')]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.node', ROOT / 'nodes/nodes_minimax_h3_tiled_upscale.py')
u = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = u
spec.loader.exec_module(u)
backend = __import__(PACKAGE + '.h3_latent_upscale', fromlist=['*'])


class ClipDouble:
    def tokenize(self, prompt, **kwargs):
        assert len(kwargs['images']) == 1
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 3, 32), {}]]


class RecordingVAE:
    def __init__(self):
        self.shapes = []

    def encode(self, pixels):
        self.shapes.append(tuple(pixels.shape))
        return torch.ones(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)


@pytest.fixture
def native_case(monkeypatch, tmp_path):
    import folder_paths
    from safetensors.torch import save_file
    monkeypatch.setattr(mm, 'get_torch_device', lambda: torch.device('cpu'))
    def free_memory(device, torch_free_too=False):
        available = 64 * 1024**3
        return (available, available) if torch_free_too else available
    monkeypatch.setattr(mm, 'get_free_memory', free_memory)
    # Any model reclamation is limited to this test's own CPU patchers.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = comfy.supported_models.MiniMaxH3(dict(
        image_model='minimax_h3', hidden_size=32, num_layers=1,
        token_refiner_num_layers=0, num_attention_heads=4, attention_head_dim=8,
        ffn_hidden_size=64, text_dim=32, timestep_input_dim=8,
        time_embed_hidden_size=32, time_embed_dim=16, rope_inv_freq_len=1))
    config.set_inference_dtype(torch.float32, None)
    base = comfy.model_base.MiniMaxH3(config, device=torch.device('cpu'))
    base.eval().requires_grad_(False)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(13)
        for p in base.diffusion_model.parameters():
            p.normal_(0, 0.02, generator=generator)
        base.diffusion_model.rope.inv_freq.fill_(1)
    model = comfy.model_patcher.ModelPatcher(base, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
    monkeypatch.setitem(folder_paths.folder_names_and_paths, backend._FOLDER,
                        ([str(tmp_path)], set(backend._EXTENSIONS)))
    network = backend._LatentResizer3D(channels=32, in_blocks=2, out_blocks=1,
                                     temporal_every=1, temporal_kernel=3, dropout=0)
    save_file(network.state_dict(), str(tmp_path / 'short_h3.safetensors'))
    vae = RecordingVAE()
    first = torch.ones(1, 32, 32, 3)
    conditioning, latent = MiniMaxH3ImageToVideo.execute(
        ClipDouble(), vae, 'Refine the still', 32, 32, 5, first, None)
    # A completed initial-pass fixture, not a claim of pretrained generation.
    latent['samples'].tensors[0].normal_(generator=torch.Generator().manual_seed(42))
    latent['sentinel'] = 'retain'
    try:
        yield model, conditioning, latent, vae, {'mode': 'Image Inpaint', 'first_frame': first}
    finally:
        mm.unload_model_and_clones(model, unload_additional_models=False)
        torch.set_num_threads(threads)


@pytest.mark.parametrize('selection', ['interpolation', 'short_h3.safetensors'])
@pytest.mark.parametrize('denoise', [0, 0.2])
@pytest.mark.parametrize('tiled', [False, True])
def test_image_inpaint_upscales_complete_short_pass(native_case, monkeypatch, selection, denoise, tiled):
    model, conditioning, latent, vae, guide = native_case
    video, audio = latent['samples'].tensors
    assert video.shape == (1, 24, 2, 2, 2)
    assert audio.shape == (1, 32, 2, 8)
    original = video.clone()
    keyframe = conditioning[0][1]['minimax_keyframes'][0]['latent']
    calls = []
    forwards = []
    def record_forward(args, extra):
        forwards.append(args['layout'].signature)
        return extra['original_block'](args)
    model.set_model_patch_replace(record_forward, 'dit', 'double_block', 0)
    real_sample = u.sample_chunk
    def record_sample(*args, **kwargs):
        working, pc, _, _, samples, mask = args[:6]
        assert samples.tensors[0].shape == (1, 24, 2, 4, 4)
        assert pc[0][1]['minimax_keyframes'][0]['latent'].shape == (1, 24, 1, 4, 4)
        assert torch.count_nonzero(mask.tensors[1]) == 0
        calls.append(True)
        return real_sample(*args, **kwargs)
    monkeypatch.setattr(u, 'sample_chunk', record_sample)
    if tiled:
        helpers = __import__(PACKAGE + '.h3_tiled_sampling', fromlist=['*'])
        # Force real per-step spatial tiling on the tiny canvas, never temporal splitting.
        monkeypatch.setattr(helpers, 'plan_tiles', lambda *a, **k: dict(
            tile_width=32, tile_height=32, overlap=0, chunk_tokens=2,
            temporal_overlap_tokens=0, explanation='TEST forced short-pass tiles'))
    output, report = u.DaSiWaH3TiledUpscale().upscale(
        model, conditioning, latent, scale=2, upscale_model=selection,
        denoise=denoise, steps=1, seed=123, director_guide=guide, vae=vae)
    result = output['samples'].tensors[0]
    assert result.shape == (1, 24, 2, 4, 4)
    assert torch.isfinite(result).all()
    assert output['samples'].tensors[1] is audio
    assert output['sentinel'] == 'retain'
    assert torch.equal(video, original)
    assert conditioning[0][1]['minimax_keyframes'][0]['latent'] is keyframe
    assert keyframe.shape == (1, 24, 1, 2, 2)
    assert vae.shapes == ([(1, 32, 32, 3), (1, 64, 64, 3)] if denoise else [(1, 32, 32, 3)])
    assert len(calls) == (1 if denoise else 0)
    assert len(forwards) == ((4 if tiled else 1) if denoise else 0)
    if forwards:
        assert all(signature[2:4] == ((2, 2) if tiled else (4, 4)) for signature in forwards)
    assert 'temporal=1 chunks' in report
    assert 'model_function_wrapper' not in model.model_options


def test_image_inpaint_does_not_bypass_source_mask_guard(native_case):
    model, conditioning, latent, vae, guide = native_case
    latent['noise_mask'] = torch.ones(1, 1, 2, 2, 2)
    with pytest.raises(ValueError, match='Masked/inpaint source latents'):
        u.DaSiWaH3TiledUpscale().upscale(model, conditioning, latent, director_guide=guide, vae=vae)


def test_image_inpaint_still_requires_packed_av_latent(native_case):
    model, conditioning, latent, vae, guide = native_case
    latent['samples'] = latent['samples'].tensors[0]
    with pytest.raises(ValueError, match='packed video\\+audio LATENT'):
        u.DaSiWaH3TiledUpscale().upscale(model, conditioning, latent, director_guide=guide, vae=vae)
