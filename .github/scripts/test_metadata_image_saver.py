"""Metadata Image Saver: auto-detected sampler values must survive the numeric manual defaults (0 = auto);
LoRAs get A1111 `Lora hashes` because Civitai links resources only by hash."""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def saver(monkeypatch, tmp_path):
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"lora bytes")
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path),
        get_temp_directory=lambda: str(tmp_path),
        get_save_image_path=lambda *args: (str(tmp_path), "test", 1, "", "test"),
        get_full_path=lambda kind, name: str(lora) if kind == "loras" and name == "style.safetensors" else None,
    ))
    path = Path(__file__).resolve().parents[2] / "nodes" / "nodes_metadata.py"
    spec = importlib.util.spec_from_file_location("nodes.metadata_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tmp_path


def test_detected_steps_cfg_seed_win_over_zero_defaults(saver):
    module, out = saver
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blur", "clip": ["1", 1]}},
        "L1": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": "style.safetensors", "strength_model": 0.8, "strength_clip": 0.8}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["L1", 0], "positive": ["2", 0], "negative": ["3", 0], "seed": 123, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "karras"}},
        "M": {"class_type": "DaSiWa_MetadataConfig", "inputs": {"save_workflow": True, "model_hash": "", "node_positive": ["2", 0], "node_negative": ["3", 0], "node_model": ["1", 0], "node_latent": ["5", 0]}},
        "7": {"class_type": "DaSiWa_MetadataImageSaver", "inputs": {"images": ["6", 0], "filename_prefix": "t", "file_format": "png", "compression": 44, "save_output": True, "metadata_config": ["M", 0]}},
    }
    config = module.DaSiWa_MetadataConfig().gather(save_workflow=True, model_hash="")[0]
    result = module.DaSiWa_MetadataImageSaver().save_images(
        filename_prefix="t", file_format="png", images=torch.zeros(1, 8, 8, 3),
        metadata_config=config, prompt=prompt, extra_pnginfo={"output_node_id": "7"})
    params = Image.open(out / "test_00001_.png").text["parameters"]
    digest = hashlib.sha256(b"lora bytes").hexdigest()[:10]
    assert params.startswith("a cat, <lora:style:0.8>\nNegative prompt: blur\nSteps: 20, Sampler: euler karras, CFG scale: 7.0, Seed: 123, ")
    assert f'Lora hashes: "style: {digest}", ' in params
    assert "Model: m.safetensors" in params and result["result"][1] == params
