import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_ltx2_loader.py"
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
    comfy_utils.load_torch_file = lambda *_args, **_kwargs: {}
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

    spec = importlib.util.spec_from_file_location("nodes_ltx2_loader_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_exposes_the_universal_model_type_selector(loader_module):
    controls = loader_module.DaSiWa_LTX2LoraLoader.INPUT_TYPES()["required"]

    assert controls["model_type"][0] == [
        "Basic",
        "LTX-2.3",
    ]
    assert controls["model_type"][1]["default"] == "Basic"
    assert loader_module.DaSiWa_LTX2LoraLoader.CATEGORY == "DaSiWa/loaders/lora"


def test_package_keeps_node_id_and_uses_universal_display_name():
    source = PACKAGE_PATH.read_text(encoding="utf-8")

    assert '"DaSiWa_LTX2LoraLoader"' in source
    assert '"DaSiWa Advanced LoRA Loader"' in source


def test_basic_mode_applies_every_weight_once(loader_module, tmp_path, monkeypatch):
    lora_path = tmp_path / "mixed.safetensors"
    lora_path.touch()
    weights = {
        "diffusion_model.block.lora_A.weight": object(),
        "adapter.audio_projection.lora_B.weight": object(),
    }
    calls = []

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: weights
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

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: weights
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

    loader_module.comfy.utils.load_torch_file = lambda *_args, **_kwargs: weights
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


def test_frontend_has_mode_selector_and_disables_unavailable_audio_controls():
    source = (Path(__file__).parents[1] / "js" / "ltx2_dynamic_ui.js").read_text(encoding="utf-8")

    assert "MODEL_TYPES" in source
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
