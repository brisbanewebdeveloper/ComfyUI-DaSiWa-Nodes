import os
import gc
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_temp_directory = lambda: "/tmp"
sys.modules.setdefault("folder_paths", folder_paths)
HELPER_PATH = Path(__file__).parents[1] / "nodes" / "helper_batch_output.py"
helper_spec = importlib.util.spec_from_file_location("helper_batch_output", HELPER_PATH)
assert helper_spec is not None and helper_spec.loader is not None
batch_output = importlib.util.module_from_spec(helper_spec)
sys.modules["helper_batch_output"] = batch_output
helper_spec.loader.exec_module(batch_output)
LOGGING_PATH = Path(__file__).parents[1] / "nodes" / "helper_logging.py"
logging_spec = importlib.util.spec_from_file_location("helper_logging", LOGGING_PATH)
assert logging_spec is not None and logging_spec.loader is not None
helper_logging = importlib.util.module_from_spec(logging_spec)
sys.modules["helper_logging"] = helper_logging
logging_spec.loader.exec_module(helper_logging)
MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_rtx_upscaler_refiner.py"
spec = importlib.util.spec_from_file_location("nodes_rtx_upscaler_refiner", MODULE_PATH)
assert spec is not None and spec.loader is not None
rtx_upscaler_refiner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtx_upscaler_refiner)

DaSiWa_RTX_UpscalerRefiner = rtx_upscaler_refiner.DaSiWa_RTX_UpscalerRefiner
_fit_frame_to_target_aspect = rtx_upscaler_refiner._fit_frame_to_target_aspect
_same_aspect = rtx_upscaler_refiner._same_aspect


def test_center_crop_matches_target_aspect_exactly():
    frame = torch.ones((3, 120, 160), dtype=torch.float32)

    fitted = _fit_frame_to_target_aspect(frame, 1920, 1080, "Center Crop (Fill)")

    _, height, width = fitted.shape
    assert _same_aspect(width, height, 1920, 1080)
    assert width <= 160
    assert height <= 120


def test_letterbox_matches_target_aspect_exactly_for_common_ratio():
    frame = torch.ones((3, 120, 160), dtype=torch.float32)

    fitted = _fit_frame_to_target_aspect(frame, 1920, 1080, "Letterbox (Fit)")

    _, height, width = fitted.shape
    assert _same_aspect(width, height, 1920, 1080)
    assert width >= 160
    assert height >= 120
    assert torch.count_nonzero(fitted == 0) > 0


def test_matching_aspect_returns_contiguous_copy_without_resize():
    frame = torch.ones((3, 90, 160), dtype=torch.float32).transpose(1, 2)

    fitted = _fit_frame_to_target_aspect(frame, 1080, 1920, "Center Crop (Fill)")

    assert fitted.shape == frame.shape
    assert fitted.is_contiguous()


def test_validate_inputs_accepts_new_comfyui_positional_signature():
    node = DaSiWa_RTX_UpscalerRefiner()

    assert node.validate_inputs("images", "IMAGE", object(), object()) is True


def test_validate_inputs_accepts_class_level_positional_signature():
    validate_inputs = DaSiWa_RTX_UpscalerRefiner.__dict__["validate_inputs"]

    assert validate_inputs(
        DaSiWa_RTX_UpscalerRefiner, "images", "IMAGE", object(), object()
    ) is True


def test_projected_output_bytes_reports_full_rgb_batch_size():
    assert rtx_upscaler_refiner._projected_output_bytes(480, 7680, 4320, torch.float32) == 191_102_976_000


def test_large_cpu_output_uses_comfy_temp_mmap_with_stable_frame_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (3, 2, 2, 3), torch.float32, torch.device("cpu"), allow_mmap=True, auto_unload_models=False)
    output[0].fill_(1)
    output[1].fill_(2)
    output[2].fill_(3)

    assert storage_path is not None
    assert os.path.dirname(storage_path) == str(tmp_path)
    assert [output[index, 0, 0, 0].item() for index in range(3)] == [1, 2, 3]


def test_mmap_output_fails_cleanly_when_comfy_temp_has_insufficient_space(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_output, "can_allocate_in_ram", lambda _: False)

    with pytest.raises(RuntimeError, match="Not enough free disk space"):
        batch_output.allocate_cpu_output(
            (3, 2, 2, 3), torch.float32, str(tmp_path), has_free_disk_space=lambda *_: False
        )


def test_mmap_output_removes_its_temporary_file_when_tensor_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, torch.device("cpu"), allow_mmap=True)

    assert storage_path is not None
    assert os.path.exists(storage_path)
    del output
    gc.collect()
    assert not os.path.exists(storage_path)


def test_input_schema_has_mmap_and_auto_unload_options():
    optional = DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["optional"]

    assert optional["use_mmap"][0] == "BOOLEAN"
    assert optional["use_mmap"][1]["default"] is False
    assert optional["auto_unload_models"][0] == "BOOLEAN"
    assert optional["auto_unload_models"][1]["default"] is True
    # Legacy key stays at its position for old workflows.
    assert "empty_cache" in optional


def test_denoise_and_deblur_default_to_off():
    required = DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["required"]

    assert required["denoise"][1]["default"] is False
    assert required["deblur"][1]["default"] is False


def test_allow_mmap_uses_ram_when_ram_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: True)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, torch.device("cpu"), allow_mmap=True, auto_unload_models=False)

    assert storage_path is None
    assert output.device.type == "cpu"


def _fake_model_management(calls):
    mm = types.ModuleType("model_management")
    mm.unload_all_models = lambda: calls.append("unload_all_models")
    mm.soft_empty_cache = lambda: calls.append("soft_empty_cache")
    return mm


def test_auto_unload_runs_full_unload_when_vram_is_short(tmp_path, monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(rtx_upscaler_refiner, "_can_fit_in_vram", lambda *_: False)
    calls = []
    monkeypatch.setitem(sys.modules, "model_management", _fake_model_management(calls))
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, device, allow_mmap=True, auto_unload_models=True)

    # VRAM stays short even after the unload -> disk-backed CPU output.
    assert storage_path is not None
    assert os.path.dirname(storage_path) == str(tmp_path)
    assert "unload_all_models" in calls
    del output
    gc.collect()


def test_auto_unload_recheck_can_restore_gpu_output(monkeypatch):
    device = torch.device("cuda:0")
    results = iter([False, True])
    monkeypatch.setattr(rtx_upscaler_refiner, "_can_fit_in_vram", lambda *_: next(results))
    calls = []
    monkeypatch.setitem(sys.modules, "model_management", _fake_model_management(calls))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, device, auto_unload_models=True)

    assert storage_path is None
    assert output.device == device
    assert calls == ["unload_all_models", "soft_empty_cache"]


def test_auto_unload_disabled_falls_back_without_unloading(tmp_path, monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(rtx_upscaler_refiner, "_can_fit_in_vram", lambda *_: False)
    calls = []
    monkeypatch.setitem(sys.modules, "model_management", _fake_model_management(calls))
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, device, allow_mmap=True, auto_unload_models=False)

    assert storage_path is not None
    assert calls == []
    del output
    gc.collect()


def test_use_mmap_off_gpu_uses_ram_when_vram_short(monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(rtx_upscaler_refiner, "_can_fit_in_vram", lambda *_: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: True)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: "/tmp")

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, device, allow_mmap=False, auto_unload_models=False)

    assert storage_path is None
    assert output.device.type == "cpu"


def test_use_mmap_off_gpu_uses_lazy_ram_when_vram_and_ram_are_short(monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(rtx_upscaler_refiner, "_can_fit_in_vram", lambda *_: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: False)

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, device, allow_mmap=False, auto_unload_models=False)

    assert storage_path is None
    assert output.device.type == "cpu"


def test_use_mmap_off_cpu_uses_lazy_ram_when_ram_is_short(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, torch.device("cpu"), allow_mmap=False, auto_unload_models=False)

    assert storage_path is None
    assert output.device.type == "cpu"


def test_use_mmap_on_gpu_prefers_ram_over_disk_when_vram_short(monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(rtx_upscaler_refiner, "_can_fit_in_vram", lambda *_: False)
    monkeypatch.setattr(rtx_upscaler_refiner, "can_allocate_in_ram", lambda _: True)

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor(
        (1, 2, 2, 3), torch.float32, device, allow_mmap=True, auto_unload_models=False)

    assert storage_path is None
    assert output.device.type == "cpu"
