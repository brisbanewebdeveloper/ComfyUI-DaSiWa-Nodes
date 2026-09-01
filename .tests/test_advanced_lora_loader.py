import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_advanced_lora_loader.py"
PACKAGE_PATH = Path(__file__).parents[1] / "__init__.py"


class _Routes:
    @staticmethod
    def get(_path):
        return lambda handler: handler


@pytest.fixture
def loader_module(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda _category: []
    folder_paths.get_full_path = lambda _category, name: name

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_lora = types.ModuleType("comfy.lora")
    comfy_sd = types.ModuleType("comfy.sd")
    comfy_utils.load_torch_file = lambda *_args, **_kwargs: ({}, None)
    comfy_lora.load_lora_for_models = lambda model, clip, _weights, _model_strength, _clip_strength: (model, clip)
    comfy.utils = comfy_utils
    comfy.lora = comfy_lora

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(json_response=lambda payload: payload)
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    helper_logging = types.ModuleType("helper_logging")
    helper_logging.log_dasiwa = lambda *_args, **_kwargs: None

    for name, module in {
        "folder_paths": folder_paths,
        "comfy": comfy,
        "comfy.utils": comfy_utils,
        "comfy.lora": comfy_lora,
        "comfy.sd": comfy_sd,
        "aiohttp": aiohttp,
        "server": server,
        "helper_logging": helper_logging,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("nodes_advanced_lora_loader_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_exposes_the_universal_model_type_selector(loader_module):
    controls = loader_module.DaSiWa_AdvancedLoRALoader.INPUT_TYPES()["required"]

    assert controls["model_type"][0] == [
        "Basic",
        "LTX-2.3",
    ]
    assert controls["model_type"][1]["default"] == "Basic"
    assert loader_module.DaSiWa_AdvancedLoRALoader.CATEGORY == "DaSiWa/loaders/lora"


def test_package_keeps_node_id_and_uses_universal_display_name():
    source = PACKAGE_PATH.read_text(encoding="utf-8")

    assert '"DaSiWa_LTX2LoraLoader"' in source
    assert '"Advanced LoRA Loader"' in source


def test_basic_mode_applies_every_weight_once(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "mixed.safetensors"
    lora_path.touch()
    weights = {
        "diffusion_model.block.lora_A.weight": object(),
        "adapter.audio_projection.lora_B.weight": object(),
    }
    calls = []

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: (weights, None)
    monkeypatch.setattr(
        loader_module,
        "_load_lora",
        lambda model, clip, loaded_weights, model_strength, clip_strength: (
            calls.append((model, clip, loaded_weights, model_strength, clip_strength))
            or (f"{model}:loaded", f"{clip}:loaded")
        ),
    )

    result = loader_module._apply_slot(
        "model", "clip", str(lora_path), 0.8, 0.5, 1.7, "Basic",
    )

    assert result == ("model:loaded", "clip:loaded")
    assert calls == [("model", "clip", weights, 0.4, 0.4)]


def test_basic_mode_ignores_audio_multiplier(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "basic.safetensors"
    lora_path.touch()
    weights = {"adapter.audio_projection.lora_A.weight": object()}
    calls = []

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: (weights, None)
    monkeypatch.setattr(
        loader_module,
        "_load_lora",
        lambda model, clip, loaded_weights, model_strength, clip_strength: (
            calls.append((loaded_weights, model_strength, clip_strength)) or (model, clip)
        ),
    )

    loader_module._apply_slot("model", "clip", str(lora_path), 0.8, 0.5, 0.0, "Basic")
    loader_module._apply_slot("model", "clip", str(lora_path), 0.8, 0.5, 2.0, "Basic")

    assert calls == [(weights, 0.4, 0.4), (weights, 0.4, 0.4)]


def test_ltx23_separates_audio_keys_and_applies_independent_strengths(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "ltx.safetensors"
    lora_path.touch()
    video_weight = object()
    audio_weight = object()
    calls = []
    weights = {
        "transformer.video_block.lora_A.weight": video_weight,
        "transformer.audio_block.lora_A.weight": audio_weight,
    }

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: (weights, None)
    monkeypatch.setattr(
        loader_module,
        "_load_lora",
        lambda model, clip, loaded_weights, model_strength, clip_strength: (
            calls.append((model, clip, loaded_weights, model_strength, clip_strength))
            or (f"{model}:loaded", f"{clip}:loaded")
        ),
    )

    loader_module._apply_slot("model", "clip", str(lora_path), 0.8, 0.5, 1.5, "LTX-2.3")

    assert calls[0][:3] == ("model", "clip", {"transformer.video_block.lora_A.weight": video_weight})
    assert calls[1][:3] == ("model:loaded", "clip:loaded", {"transformer.audio_block.lora_A.weight": audio_weight})
    assert math.isclose(calls[0][3], 0.4) and math.isclose(calls[0][4], 0.4)
    assert math.isclose(calls[1][3], 1.2) and math.isclose(calls[1][4], 1.2)


def test_basic_mode_passes_lora_metadata_to_core_loader(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "pdd.safetensors"
    lora_path.touch()
    weights = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": object(),
               "diffusion_model.final_layer.video_out.set_weight": object()}
    metadata = {"pdd_num_steps": 32, "pdd_block_size": 4, "converted_layout": "comfyui_minimax_h3"}
    calls = []

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: (weights, metadata)
    monkeypatch.setattr(
        loader_module, "_load_lora",
        lambda model, clip, loaded_weights, model_strength, clip_strength, lora_metadata=None: (
            calls.append((loaded_weights, model_strength, clip_strength, lora_metadata))
            or (model, clip)
        ),
    )

    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic")

    assert calls == [(weights, 1.0, 1.0, metadata)]


def test_schema_default_is_cache_off(loader_module):
    controls = loader_module.DaSiWa_AdvancedLoRALoader.INPUT_TYPES()["required"]
    assert controls["use_cache"][0] == "BOOLEAN"
    assert controls["use_cache"][1]["default"] is False


def test_cache_off_by_default_reads_every_slot(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "big.safetensors"
    lora_path.touch()
    weights = {"diffusion_model.final_layer.video_out.set_weight": object()}
    reader_calls = {"n": 0}

    def fake_read(*_a, **_k):
        reader_calls["n"] += 1
        return (weights, None)

    loader_module.comfy.utils.load_torch_file = fake_read
    monkeypatch.setattr(loader_module, "_load_lora", lambda *a, **k: (a[0], a[1]))
    loader_module._LORA_FILE_CACHE.clear()

    # use_cache defaults to False -> the LRU cache is never used
    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic")
    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic")

    assert reader_calls["n"] == 2   # no cache when off (existing behavior)


def test_cache_on_reads_once_per_unique_path(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "big.safetensors"
    lora_path.touch()
    weights = {"diffusion_model.final_layer.video_out.set_weight": object()}
    reader_calls = {"n": 0}

    def fake_read(*_a, **_k):
        reader_calls["n"] += 1
        return (weights, None)

    loader_module.comfy.utils.load_torch_file = fake_read
    monkeypatch.setattr(loader_module, "_load_lora", lambda *a, **k: (a[0], a[1]))
    loader_module._LORA_FILE_CACHE.clear()

    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic", True)
    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic", True)

    assert reader_calls["n"] == 1   # second slot reused the cached read


def test_frontend_has_mode_selector_and_disables_unavailable_audio_controls():
    source = (Path(__file__).parents[1] / "js" / "advanced_lora_loader_ui.js").read_text(encoding="utf-8")

    assert "MODEL_TYPES" in source
    assert "use_cache" in source
    assert "syncCacheWidget" in source
    assert "CONTROL_DESCRIPTIONS.cache" in source
    assert '"MiniMax H3 (prepared)"' not in source
    assert "hasSeparatedAudio" in source
    assert "syncModeWidget" in source
    assert '"VIS"' in source
    assert "Visual multiplier" in source
    assert "toggleAll" in source
    assert "ALL✓" in source
    assert "-5.0, 5.0" in source
    assert "openValueEditor" in source
    assert "closeValueEditor" in source
    assert "positionValueEditor" in source
    assert "position:fixed" in source
    assert "transform-origin:0 0" in source
    assert "_viewState" in source
    assert "getBoundingClientRect" in source
    assert "graph-canvas" in source
    assert "requestAnimationFrame(track)" in source
    assert "LoRA Strength" in source
    assert "Visual Multiplier" in source
    assert "Audio Multiplier" in source
    assert "onCommit" in source
    assert '"H3: keys TBD"' not in source
    assert "Audio separation awaits published MiniMax H3 tensor keys" not in source


def test_readme_documents_basic_visual_control_and_toggle_all():
    source = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "Basic mode" in source
    assert "VIS" in source
    assert "Toggle All" in source
    assert "−5.0 to +5.0" in source


class _T:
    """Fake tensor exposing only .shape, for PDD head-bank tests."""

    def __init__(self, shape):
        self.shape = shape


# ── PDD head-bank guard: detection + model-width read (Task 1) ──────────────
def test_pdd_head_bank_detection(loader_module):
    weights = {
        "diffusion_model.final_layer.video_out.set_weight": _T((3072, 5376)),
        "diffusion_model.final_layer.video_out.set_bias": _T((3072,)),
    }
    meta = {"pdd_num_steps": 32, "pdd_block_size": 4, "converted_layout": "comfyui_minimax_h3"}
    is_pdd, video_width = loader_module._pdd_head_bank_info(weights, meta)
    assert is_pdd is True
    assert video_width == 3072


def test_non_pdd_lora_not_flagged(loader_module):
    normal = {"diffusion_model.to_q.lora_A.weight": _T((64, 512))}
    assert loader_module._pdd_head_bank_info(normal, None)[0] is False


def test_pdd_metadata_flat_and_wrapped_forms(loader_module):
    weights = {"diffusion_model.final_layer.video_out.set_weight": _T((3072, 5376))}
    # Flat form (what load_torch_file(return_metadata=True) actually returns):
    assert loader_module._pdd_head_bank_info(weights, {"pdd_num_steps": 32})[0] is True
    # Wrapped form:
    assert loader_module._pdd_head_bank_info(weights, {"__metadata__": {"pdd_num_steps": 32}})[0] is True
    # No PDD metadata -> not PDD regardless of keys:
    assert loader_module._pdd_head_bank_info(weights, None)[0] is False


def test_model_final_layer_width_read_defensive(loader_module):
    class _Tensor:
        shape = (96, 5376)

    class _VideoOut:
        weight = _Tensor()

    class _Fl:
        video_out = _VideoOut()

    class _DM:
        final_layer = _Fl()

    class _MB:
        diffusion_model = _DM()

    class _MP:
        model = _MB()

    assert loader_module._model_video_out_width(_MP()) == 96


def test_model_final_layer_width_unreadable_returns_none(loader_module):
    assert loader_module._model_video_out_width(object()) is None


# ── PDD head-bank guard: warn-only, crash stands (Task 2) ───────────────────
def _pdd_weights():
    return {
        "diffusion_model.final_layer.video_out.set_weight": _T((3072, 5376)),
        "diffusion_model.final_layer.video_out.set_bias": _T((3072,)),
        "diffusion_model.final_layer.audio_out.set_weight": _T((1024, 5376)),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": _T((64, 512)),
    }


def _cap_logs(loader_module, monkeypatch):
    """Capture log_dasiwa calls; returns the captured list of (component, msg)."""
    calls = []
    monkeypatch.setattr(loader_module, "log_dasiwa",
        lambda component, msg: calls.append((component, msg)))
    return calls


def _guard_slot(loader_module, tmp_path, monkeypatch, meta, model_w, weights=None):
    """Run _apply_slot (Basic mode) with PDD metadata + a fixed model width.

    Returns (keys_passed_to_core, log_calls). The head-bank keys MUST still be
    in keys (warn-only: no circumvention) regardless of model_w.
    """
    lora_path = tmp_path / "pdd.safetensors"
    lora_path.touch()
    if weights is None:
        weights = _pdd_weights()
    passed = {}
    logs = _cap_logs(loader_module, monkeypatch)

    loader_module.comfy.utils.load_torch_file = lambda *_a, **_k: (weights, meta)
    monkeypatch.setattr(loader_module, "_model_video_out_width", lambda _m: model_w)
    monkeypatch.setattr(loader_module, "_load_lora",
        lambda m, c, w, sm, sc, lora_metadata=None: passed.__setitem__("keys", w) or (m, c))

    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic")
    return passed["keys"], logs


def test_warns_and_keeps_pdd_head_bank_when_model_single_head(loader_module, tmp_path, monkeypatch):
    # Proven mismatch: PDD LoRA (bank 3072) on a single-head model (96).
    # The crash is protective, so we keep the head-bank keys in place (no
    # circumvention) and print a console warning instead.
    meta = {"pdd_num_steps": 32, "pdd_block_size": 4}
    keys, logs = _guard_slot(loader_module, tmp_path, monkeypatch, meta, model_w=96)

    # No circumvention: the incompatible head-bank keys still reach core.
    assert "diffusion_model.final_layer.video_out.set_weight" in keys
    assert "diffusion_model.final_layer.audio_out.set_weight" in keys
    # And a console warning was logged:
    assert any("PDD head-bank width" in msg for _c, msg in logs)


def test_keeps_pdd_head_bank_when_model_width_unreadable(loader_module, tmp_path, monkeypatch):
    # Ambiguity -> keep keys, and NO warning (no positive evidence of mismatch).
    meta = {"pdd_num_steps": 32}
    keys, logs = _guard_slot(loader_module, tmp_path, monkeypatch, meta, model_w=None)

    assert "diffusion_model.final_layer.video_out.set_weight" in keys
    assert not any("PDD head-bank width" in msg for _c, msg in logs)


def test_keeps_pdd_head_bank_when_widths_match(loader_module, tmp_path, monkeypatch):
    # Genuine PDD model: width matches -> keep keys, no warning.
    meta = {"pdd_num_steps": 32}
    keys, logs = _guard_slot(loader_module, tmp_path, monkeypatch, meta, model_w=3072)

    assert "diffusion_model.final_layer.video_out.set_weight" in keys
    assert not any("PDD head-bank width" in msg for _c, msg in logs)


def test_no_warning_for_non_pdd_lora_even_on_single_head(loader_module, tmp_path, monkeypatch):
    # A normal (non-PDD) LoRA has no pdd metadata -> never flagged, regardless of model.
    lora_path = tmp_path / "normal.safetensors"
    lora_path.touch()
    weights = {"diffusion_model.final_layer.video_out.set_weight": _T((96, 5376))}
    passed = {}
    logs = _cap_logs(loader_module, monkeypatch)

    loader_module.comfy.utils.load_torch_file = lambda *_a, **_k: (weights, None)
    monkeypatch.setattr(loader_module, "_model_video_out_width", lambda _m: 96)
    monkeypatch.setattr(loader_module, "_load_lora",
        lambda m, c, w, sm, sc, lora_metadata=None: passed.__setitem__("keys", w) or (m, c))

    loader_module._apply_slot("model", "clip", str(lora_path), 1.0, 1.0, 1.0, "Basic")

    assert "diffusion_model.final_layer.video_out.set_weight" in passed["keys"]
    assert not any("PDD head-bank width" in msg for _c, msg in logs)
