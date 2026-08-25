"""Source-level regression tests for MiniMax H3 Director prompt-field height persistence.

The Director v1 and v2 timelines re-render their whole DOM on every mutation
(media add, mode change, resolution change, workflow reload). Resized prompt
text boxes must therefore persist their height in the per-node install closure
(fieldHeights) and in the serialized timeline_data state (state.field_heights).
"""
from pathlib import Path

V1 = Path(__file__).resolve().parent.parent / "js" / "minimax_h3_director.js"
V2 = Path(__file__).resolve().parent.parent / "js" / "minimax_h3_director_v2.js"

# (key, "fieldKey: \"<key>\"") — all ten form-builder call sites in BOTH files.
ALL_KEYS = (
    "imd", "soundscape", "music", "simple_prompt",
    "ref_subject_definitions", "ref_summary", "ref_retention_analysis",
    "ref_detailed_description", "ref_soundscape", "ref_music",
)


def _resizer_asserts(src: str) -> None:
    assert "const key = opts.fieldKey || \"\";" in src
    assert "if (key) fieldHeights[key] = height;" in src
    assert "const saved = Number(fieldHeights[key]);" in src
    assert "if (Number.isFinite(saved) && saved >= 60) area.style.height = `${saved}px`;" in src


def _call_sites_asserts(src: str) -> None:
    for key in ALL_KEYS:
        assert f"fieldKey: \"{key}\"" in src, f"field key {key} missing"
    assert src.count(", fieldHeights);") >= 10


def _persist_asserts(src: str) -> None:
    assert "state.field_heights = { ...fieldHeights };" in src
    assert src.count("for (const [key, value] of Object.entries(state.field_heights || {}))") >= 2


def test_v1_resizer_records_resized_heights_under_stable_field_keys():
    _resizer_asserts(V1.read_text(encoding="utf-8"))


def test_v1_rebuilt_fields_restore_their_persisted_height():
    _resizer_asserts(V1.read_text(encoding="utf-8"))


def test_v1_every_form_builder_passes_the_height_store_and_a_stable_key():
    _call_sites_asserts(V1.read_text(encoding="utf-8"))


def test_v1_emit_persists_field_heights_into_the_serialized_timeline_state():
    _persist_asserts(V1.read_text(encoding="utf-8"))


def test_v2_resizer_records_resized_heights_under_stable_field_keys():
    _resizer_asserts(V2.read_text(encoding="utf-8"))


def test_v2_every_form_builder_passes_the_height_store_and_a_stable_key():
    _call_sites_asserts(V2.read_text(encoding="utf-8"))


def test_v2_emit_persists_field_heights_into_the_serialized_timeline_state():
    _persist_asserts(V2.read_text(encoding="utf-8"))
