import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nodes import helper_refmod_format as refmods
from nodes import refmod_library


def _save(path, kind="video", key="refmod_meta", **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"kind": kind, "name": path.stem, "description": "desc", "latent_t": 2,
                "latent_h": 4, "latent_w": 6, **meta}
    save_file({"latent": torch.ones(1, 24, 2, 4, 6)}, str(path), metadata={key: json.dumps(metadata)})




def test_refmods_roots_follow_configured_comfy_model_roots(monkeypatch, tmp_path):
    primary_models = tmp_path / "comfy" / "models"
    shared_models = tmp_path / "shared" / "models"
    # A second, shared models root is only discovered when the operator
    # registers its refmods category explicitly — it is never inferred from
    # sibling categories like checkpoints/loras.
    fake_folder_paths = types.SimpleNamespace(
        models_dir=str(primary_models),
        get_folder_paths=lambda name: {
            "refmods": [str(shared_models / "refmods")],
        }.get(name, []),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    # Configured models root + explicitly registered shared refmods category,
    # in registration order (registered category first, then the default root).
    assert refmods.refmods_roots() == [
        str(shared_models / "refmods"),
        str(primary_models / "refmods"),
    ]


def test_refmods_roots_does_not_guess_sibling_categories(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "loras").mkdir(parents=True)
    fake_folder_paths = types.SimpleNamespace(
        models_dir=str(models_dir),
        get_folder_paths=lambda name: {
            "checkpoints": [str(models_dir / "checkpoints")],
            "loras": [str(models_dir / "loras")],
        }.get(name, []),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    # Only the configured models root's refmods subfolder is returned; the
    # presence of sibling categories does not fabricate extra roots.
    assert refmods.refmods_roots() == [str(models_dir / "refmods")]


def test_list_refmods_recurses_skips_invalid_and_deduplicates(monkeypatch, tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    _save(one / "people" / "alice.safetensors")
    _save(two / "people" / "alice.safetensors")
    _save(one / "voice.safetensors", kind="audio")
    _save(one / "bundle.safetensors", kind="bundle")
    _save(one / "graph_presets" / "hidden.safetensors")
    save_file({"latent": torch.ones(1)}, str(one / "invalid.safetensors"))
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(one), str(two)])
    assert refmods.list_refmods() == ["people/alice", "voice"]


def test_read_metadata_header_audio_key_and_sidecar(monkeypatch, tmp_path):
    _save(tmp_path / "audio.safetensors", kind="audio", key="audio_refmod_meta")
    assert refmods.read_refmod_meta(str(tmp_path / "audio"))["kind"] == "audio"
    save_file({"latent": torch.ones(1)}, str(tmp_path / "old.safetensors"))
    (tmp_path / "old.json").write_text(json.dumps({"kind": "image", "name": "old"}))
    assert refmods.read_refmod_meta(str(tmp_path / "old"))["name"] == "old"


def test_refmod_discovery_and_loading_accept_case_insensitive_safetensors_extensions(monkeypatch, tmp_path):
    path = tmp_path / "People" / "Alice.SAFETENSORS"
    _save(path, kind="image")
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(tmp_path)])

    assert refmods.list_refmods() == ["People/Alice"]
    assert refmods.find_mod_path("People/Alice") == str(path)[:-len(".SAFETENSORS")]
    assert refmods.load_refmod("People/Alice")[1]["kind"] == "image"


def test_refmod_bundle_lists_and_expands_every_member(monkeypatch, tmp_path):
    path = tmp_path / "slime_cat_full.safetensors"
    members = [
        {"name": "slime_visual", "kind": "video", "latent_t": 4, "latent_h": 4, "latent_w": 6},
        {"name": "slime_voice", "kind": "audio", "latent_t": 5},
    ]
    save_file(
        {"ref_0": torch.ones(1, 24, 4, 4, 6), "ref_1": torch.ones(1, 32, 2, 5)},
        str(path),
        metadata={"refmod_meta": json.dumps({"_format_version": 5, "kind": "bundle", "name": "slime_cat_full", "members": members})},
    )
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(tmp_path)])

    assert refmods.list_refmods() == ["slime_cat_full"]
    assert [(meta["kind"], latent.shape) for latent, meta in refmods.load_refmods("slime_cat_full")] == [
        ("video", (1, 24, 4, 4, 6)),
        ("audio", (1, 32, 2, 5)),
    ]
    monkeypatch.setattr(refmod_library, "refmods_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(refmod_library, "list_refmods", refmods.list_refmods)
    monkeypatch.setattr(refmod_library, "find_mod_path", refmods.find_mod_path)
    monkeypatch.setattr(refmod_library, "read_refmod_meta", refmods.read_refmod_meta)
    refmod_library._entries_cache.update(sig=None, entries=[])
    assert refmod_library.library_entries()[0]["kinds"] == ["video", "audio"]


@pytest.mark.parametrize("name", ["", "None", "/absolute", "../escape", "a/../b", "a//b", "..%2fescape"])
def test_find_mod_path_rejects_unsafe_names(monkeypatch, tmp_path, name):
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(tmp_path)])
    with pytest.raises(ValueError, match="Invalid RefMod"):
        refmods.find_mod_path(name)


def test_load_refmod_clones_and_rejects_invalid_bundle(monkeypatch, tmp_path):
    _save(tmp_path / "person.safetensors")
    _save(tmp_path / "group.safetensors", kind="bundle")
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(tmp_path)])
    first, meta = refmods.load_refmod("person")
    first.zero_()
    second, _ = refmods.load_refmod("person")
    assert meta["kind"] == "video" and torch.all(second == 1)
    with pytest.raises(ValueError, match="no supported RefMod metadata"):
        refmods.load_refmod("group")


def test_symlinked_refmods_are_discovered_and_loaded_without_looping(monkeypatch, tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    _save(outside / "person.safetensors")
    _save(outside / "nested" / "voice.safetensors", kind="audio")
    (root / "person.safetensors").symlink_to(outside / "person.safetensors")
    (root / "linked").symlink_to(outside / "nested", target_is_directory=True)
    (outside / "nested" / "back").symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(root)])

    assert refmods.list_refmods() == ["linked/voice", "person"]
    assert refmods.find_mod_path("person") == str(root / "person")
    assert refmods.load_refmod("linked/voice")[1]["kind"] == "audio"

    monkeypatch.setattr(refmod_library, "refmods_roots", lambda: [str(root)])
    monkeypatch.setattr(refmod_library, "list_refmods", refmods.list_refmods)
    monkeypatch.setattr(refmod_library, "find_mod_path", refmods.find_mod_path)
    monkeypatch.setattr(refmod_library, "read_refmod_meta", refmods.read_refmod_meta)
    refmod_library._entries_cache.update(sig=None, entries=[])
    assert [entry["name"] for entry in refmod_library.library_entries()] == ["linked/voice", "person"]


def test_library_cache_invalidates_when_sidecar_changes(monkeypatch, tmp_path):
    path = tmp_path / "old.safetensors"
    save_file({"latent": torch.ones(1)}, str(path))
    sidecar = tmp_path / "old.json"
    sidecar.write_text(json.dumps({"kind": "image", "name": "old", "description": "first"}))
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(refmod_library, "refmods_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(refmod_library, "list_refmods", refmods.list_refmods)
    monkeypatch.setattr(refmod_library, "find_mod_path", refmods.find_mod_path)
    monkeypatch.setattr(refmod_library, "read_refmod_meta", refmods.read_refmod_meta)
    refmod_library._entries_cache.update(sig=None, entries=[])
    assert refmod_library.library_entries()[0]["description"] == "first"
    sidecar.write_text(json.dumps({"kind": "image", "name": "old", "description": "changed and longer"}))
    assert refmod_library.library_entries()[0]["description"] == "changed and longer"


def test_library_entries_are_metadata_only_and_cached(monkeypatch, tmp_path):
    _save(tmp_path / "person.safetensors", concept_type="person")
    monkeypatch.setattr(refmods, "refmods_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(refmod_library, "refmods_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(refmod_library, "list_refmods", refmods.list_refmods)
    monkeypatch.setattr(refmod_library, "find_mod_path", refmods.find_mod_path)
    monkeypatch.setattr(refmod_library, "read_refmod_meta", refmods.read_refmod_meta)
    monkeypatch.setattr(refmod_library, "load_refmod", lambda *_: (_ for _ in ()).throw(AssertionError("tensor loaded")), raising=False)
    refmod_library._entries_cache.update(sig=None, entries=[])
    assert refmod_library.library_entries() == [{"name": "person", "kind": "video", "kinds": ["video"], "concept": "person",
        "description": "desc", "tokens": 12, "mtime": pytest.approx((tmp_path / "person.safetensors").stat().st_mtime)}]
