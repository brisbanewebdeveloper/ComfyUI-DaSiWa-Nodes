import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_minimax_h3_cache.py"
PACKAGE_PATH = Path(__file__).parents[1] / "__init__.py"


def load_module():
    spec = importlib.util.spec_from_file_location("minimax_h3_cache_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_reuses_saved_block_stack_residual_within_limits():
    module = load_module()
    cache = module.H3BlockStackCache(
        reuse_threshold=0.05,
        start_percent=0.0,
        end_percent=1.0,
        max_steps=2,
        device="auto",
        verbose=False,
    )
    cache.begin(4)
    calls = []

    def original(block_args):
        calls.append(block_args["img"].clone())
        return {"img": block_args["img"] + 2}

    first = cache(
        {"img": torch.zeros(4, 8), "timestep": torch.tensor([1000.0]), "cache_ranges": ((0, 4),)},
        {"original_block": original},
    )
    second = cache(
        {"img": torch.zeros(4, 8), "timestep": torch.tensor([900.0]), "cache_ranges": ((0, 4),)},
        {"original_block": original},
    )

    assert len(calls) == 1
    assert torch.equal(first["img"], torch.full((4, 8), 2.0))
    assert torch.equal(second["img"], torch.full((4, 8), 2.0))


def test_cache_resets_for_changed_tensor_layout():
    module = load_module()
    cache = module.H3BlockStackCache(0.05, 0.0, 1.0, 2, "auto", False)
    cache.begin(4)
    calls = []

    def original(block_args):
        calls.append(block_args["img"].shape)
        return {"img": block_args["img"] + 1}

    cache({"img": torch.zeros(4, 8), "timestep": torch.tensor([1000.0])}, {"original_block": original})
    result = cache({"img": torch.zeros(6, 8), "timestep": torch.tensor([900.0])}, {"original_block": original})

    assert calls == [torch.Size((4, 8)), torch.Size((6, 8))]
    assert torch.equal(result["img"], torch.ones(6, 8))


def test_node_clones_and_registers_model_scoped_patches(monkeypatch):
    module = load_module()
    def patched_forward(*_args, **_kwargs):
        return None

    def builder_stub(with_pdd_args: bool = False):
        return patched_forward

    monkeypatch.setattr(module, "build_h3_block_loop_forward", builder_stub)

    class _StubFinalLayer:
        def forward(self, x, t_emb, video_seg, audio_seg):
            return None

    class MiniMaxH3Model:
        blocks = [object()]
        final_layer = _StubFinalLayer()

    class Clone:
        def __init__(self):
            self.model = type("Model", (), {"diffusion_model": MiniMaxH3Model()})()
            self.model_options = {"transformer_options": {}}
            self.object_patches = []
            self.replacements = []
            self.wrappers = []

        def add_object_patch(self, name, value):
            self.object_patches.append((name, value))

        def set_model_patch_replace(self, patch, name, block_name, number):
            self.replacements.append((patch, name, block_name, number))

        def add_wrapper(self, wrapper_type, wrapper):
            self.wrappers.append((wrapper_type, wrapper))

    class Model:
        def __init__(self):
            self.clone_calls = 0
            self.result = Clone()

        def clone(self):
            self.clone_calls += 1
            return self.result

    model = Model()
    result = module.MiniMaxH3Cache().patch(model, 0.05, 0.15, 0.90, 2, "auto", False)

    assert result == (model.result,)
    assert model.clone_calls == 1
    assert model.result.object_patches[0][0] == "diffusion_model._forward"
    assert model.result.object_patches[0][1].__self__ is model.result.model.diffusion_model
    assert len(model.result.replacements) == 1
    assert model.result.replacements[0][1:] == ("dit", "block_loop", 0)
    assert len(model.result.wrappers) == 1


def test_node_rejects_invalid_range_and_non_h3_model():
    module = load_module()

    with pytest.raises(ValueError, match="start_percent"):
        module.MiniMaxH3Cache().patch(object(), 0.05, 0.91, 0.90, 2, "auto", False)

    class Clone:
        model = type("Model", (), {"diffusion_model": object()})()

    class Model:
        def clone(self):
            return Clone()

    with pytest.raises(ValueError, match="MiniMax H3"):
        module.MiniMaxH3Cache().patch(Model(), 0.05, 0.15, 0.90, 2, "auto", False)


def test_package_registers_mini_max_h3_cache_node():
    module = load_module()
    source = PACKAGE_PATH.read_text(encoding="utf-8")

    assert "MiniMaxH3Cache" in source
    assert '"MiniMax H3 Cache"' in source
    assert module.MiniMaxH3Cache.CATEGORY == "DaSiWa/MiniMax H3"


class _RecordingFinalLayer:
    def __init__(self, arity: int):
        self.arity = arity
        self.calls = []

    def __call__(self, *args):
        assert len(args) == self.arity, f"expected {self.arity} args, got {len(args)}"
        self.calls.append(args)
        return ("video", "audio")


def test_final_layer_call_helper_dispatches_by_pdd_arity():
    module = load_module()

    sigmas = torch.tensor([1.0, 0.5, 0.0])

    layer4 = _RecordingFinalLayer(4)
    result = module.final_layer_call(
        layer4, "hidden", "t_emb", "vseg", "aseg",
        torch.tensor(0.5), {"sample_sigmas": sigmas}, 12.0, 3.0, False,
    )
    assert result == ("video", "audio")
    assert len(layer4.calls[0]) == 4

    layer7 = _RecordingFinalLayer(7)
    module.final_layer_call(
        layer7, "hidden", "t_emb", "vseg", "aseg",
        torch.tensor(0.5), {"sample_sigmas": sigmas}, 12.0, 3.0, True,
    )
    args7 = layer7.calls[0]
    assert len(args7) == 7
    assert args7[4] == torch.tensor(0.5)          # sigma
    assert args7[5] is sigmas                      # sample_sigmas from transformer_options
    assert args7[6] == (12.0, 3.0)                 # shifts tuple


def _patch_fixture(monkeypatch, final_layer_arity: int):
    module = load_module()
    seen = {}

    def capture_builder(*args, **kwargs):
        if args:
            seen["with_pdd_args"] = args[0]
        else:
            seen["with_pdd_args"] = kwargs.get("with_pdd_args", False)

        def patched_forward(*_a, **_k):
            return None

        return patched_forward

    monkeypatch.setattr(module, "build_h3_block_loop_forward", capture_builder)

    if final_layer_arity == 7:
        class FinalLayer:
            def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
                return None
    else:
        class FinalLayer:
            def forward(self, x, t_emb, video_seg, audio_seg):
                return None

    class MiniMaxH3Model:
        blocks = [object()]
        final_layer = FinalLayer()

    class Clone:
        def __init__(self):
            self.model = type("M", (), {"diffusion_model": MiniMaxH3Model()})()
            self.model_options = {"transformer_options": {}}
            self.object_patches = []
            self.replacements = []
            self.wrappers = []

        def add_object_patch(self, name, value):
            self.object_patches.append((name, value))

        def set_model_patch_replace(self, patch, name, block_name, number):
            self.replacements.append((patch, name, block_name, number))

        def add_wrapper(self, wrapper_type, wrapper):
            self.wrappers.append((wrapper_type, wrapper))

    class Model:
        def clone(self):
            return Clone()

    return module, Model(), seen


def test_patch_detects_seven_arg_final_layer(monkeypatch):
    module, model, seen = _patch_fixture(monkeypatch, final_layer_arity=7)
    module.MiniMaxH3Cache().patch(model, 0.05, 0.15, 0.90, 2, "auto", False)
    assert seen["with_pdd_args"] is True


def test_patch_detects_four_arg_final_layer(monkeypatch):
    module, model, seen = _patch_fixture(monkeypatch, final_layer_arity=4)
    module.MiniMaxH3Cache().patch(model, 0.05, 0.15, 0.90, 2, "auto", False)
    assert seen["with_pdd_args"] is False
