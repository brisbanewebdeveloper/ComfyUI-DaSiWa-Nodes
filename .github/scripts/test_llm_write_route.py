"""POST /dasiwa/llm/write: the Prompt Writer's dialect on the Forge backends, without the graph."""
# Needs ComfyUI on the path: PYTHONPATH=/path/to/ComfyUI python -m pytest .github/scripts/test_llm_write_route.py
import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "_dasiwa_write_nodes"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "nodes")]
sys.modules[PACKAGE] = package
forge = importlib.import_module(f"{PACKAGE}.h3_forge")


def backend(reply):
    b = Mock(base="http://remote.invalid")
    b.can_see.return_value = False
    b.chat.return_value = (reply, {})
    b.unload.return_value = True
    return b


def run(body, b):
    with patch.object(forge, "backends", return_value={"ollama": b}), patch.object(forge, "_is_this_machine", return_value=False):
        return forge.write({"model": "ollama:test", **body})


def test_writes_in_the_dialect_and_spaces_tags():
    b = backend("===SEGMENT: Positive prompt===\n1girl, long_hair, watercolor (medium)")
    result = run({"write_for": "Anima", "idea": "a girl in rain", "detail": 7}, b)
    assert result == {"prompt": "1girl, long hair, watercolor \\(medium\\)", "model": "ollama:test", "saw_images": 0}
    system, user = b.chat.call_args.args[1:3]
    assert "Positive prompt" in system and user.startswith("a girl in rain") and "Detail 7 of 10" in user
    b.unload.assert_called_once_with("test")


def test_blind_model_writes_from_the_idea_and_reads_output_pictures(tmp_path):
    b = backend("===SEGMENT: Positive prompt===\nA red door.")
    assert run({"write_for": "Anima", "idea": "a door", "images": ["a.png"]}, b)["saw_images"] == 0
    assert b.chat.call_args.args[3] == []
    with pytest.raises(forge.ForgeError) as blind:
        run({"write_for": "Anima", "idea": "", "images": ["a.png"]}, backend(""))
    assert blind.value.code == "no_vision"
    (tmp_path / "gen.png").write_bytes(b"png")
    seeing = backend("===SEGMENT: Positive prompt===\nA red door.")
    seeing.can_see.return_value = True
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)), \
         patch.object(forge, "_image_b64", side_effect=lambda path: path):
        assert run({"write_for": "Anima", "idea": "", "images": ["gen.png [output]"]}, seeing)["saw_images"] == 1
    assert seeing.chat.call_args.args[3] == [str(tmp_path / "gen.png")]
    with pytest.raises(forge.ForgeError) as escape:
        run({"write_for": "Anima", "idea": "x", "images": ["../x.png [output]"]}, seeing)
    assert escape.value.code == "bad_references"


def test_refuses_unknown_targets_and_levels():
    with pytest.raises(forge.ForgeError) as target:
        run({"write_for": "MiniMax H3", "idea": "x"}, backend(""))
    assert target.value.code == "bad_mode"
    with pytest.raises(forge.ForgeError) as level:
        run({"write_for": "Krea2", "idea": "x", "creativity": 11}, backend(""))
    assert level.value.code == "bad_prompt"
