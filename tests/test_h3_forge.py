"""Forge request and mocked-model contracts (no model/service required)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_forge as forge


def test_mixed_video_audio_labels_match_native_paired_audio_first():
    refs = [{"kind": "video", "stream": "audio"}, {"kind": "video", "stream": "both"}, {"kind": "audio"}]
    lines, _ = forge.format_references(refs, "REF2VA")
    assert lines[0].startswith("- <Audio 2>")
    assert lines[1].startswith("- <Video 1>")
    assert lines[2].startswith("- <Audio 1>") and "synchronized" in lines[2]
    assert lines[3].startswith("- <Audio 3>")


def test_reference_intent_and_group_notes():
    refs = [
        {"kind": "image", "role": "pose", "instructions": "Only arms", "keep": "stance", "drop": "face"},
        {"kind": "image", "role": "custom", "instructions": "Use room layout"},
        {"kind": "image", "role": "subject", "subject_group": "a", "instructions": "Blue coat"},
        {"kind": "image", "role": "subject", "subject_group": "a", "instructions": "Same person"},
    ]
    lines, pictures = forge.format_references(refs, "REF2VA")
    text = "\n".join(lines)
    assert "pose" in text and "not" in text and "identity" in text
    for note in ("Only arms", "stance", "face", "Use room layout", "Blue coat", "Same person"):
        assert note in text
    assert "instructions (<Picture 3>): Blue coat" in text
    assert len(pictures) == 4


@pytest.mark.parametrize("refs", [
    "bad", [None], [{}], [{"kind": "image", "role": "custom", "instructions": "  "}],
    [{"kind": "image", "instructions": 1}], [{"kind": "image", "keep": "x" * 4001}],
    [{"kind": "image"}] * 33,
])
def test_request_reference_validation(refs):
    with pytest.raises(forge.ForgeError) as exc:
        forge.validate_references(refs)
    assert exc.value.code == "bad_references"


def test_reference_validation_preserves_legacy_optional_fields():
    refs = [{"kind": "image", "role": "subject", "path": "a.png"}]
    assert forge.validate_references(refs) == refs
    assert forge.validate_references(None) == []


class Model:
    base = "http://remote.example"
    def __init__(self, raw, sees=True):
        self.raw, self.sees, self.calls, self.unloads = raw, sees, [], []
    def models(self):
        return [{"id": "openai:test"}]
    def can_see(self, name):
        return self.sees
    def unload(self, name):
        self.unloads.append(name)
        return True
    def chat(self, *args):
        self.calls.append(args)
        return self.raw, {}


@pytest.fixture
def draft_context(tmp_path, monkeypatch):
    from PIL import Image
    from nodes.h3_continuity import media
    inputs, tail = tmp_path / "inputs", tmp_path / "tail"
    inputs.mkdir(); tail.mkdir()
    for directory, name in ((inputs, "ref.png"), (tail, "old.jpg"), (tail, "new.jpg")):
        Image.new("RGB", (8, 8), "red").save(directory / name)
    monkeypatch.setattr(media, "ensure_tail_thumbnails", lambda *a: ["old.jpg", "new.jpg"])
    monkeypatch.setattr(forge, "_image_b64", lambda p: Path(p).name)
    refs = [{"kind": "image", "role": "pose", "path": "ref.png", "instructions": "Only arms"}]
    return inputs, tail, refs


def structured_raw(definitions="<Subject 7>: dancer from <Picture 1>"):
    bodies = [definitions, "Continue dancing", "Preserve stance", "[Shot 1] dancer steps forward", "Footsteps", "N/A"]
    return "\n".join(f"===SEGMENT: {label} ===\n{body}" for (label, _), body in zip(forge._REF_FIELDS, bodies))


def test_structured_continuity_dual_images_and_contract(draft_context, monkeypatch):
    inputs, tail, refs = draft_context
    model = Model(structured_raw())
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate_continuity_draft(
        {"prompt": "Previous", "clip_id": "source"}, "Step forward", str(tail), "openai:test", {},
        input_directory=str(inputs), references=refs, existing_definitions="<Subject 7>: dancer", structured=True,
        options={"mode": "REF2VA"})
    _, system, user, images, *_ = model.calls[0]
    assert images == ["ref.png", "old.jpg", "new.jpg"]
    assert "Only arms" in user and "<Subject 7>: dancer" in user
    assert "Tail frame 1" in user and "Tail frame 2" in user
    assert "NOT <Picture N>" in user
    assert "uninterrupted" in system and "IDs" in system
    for label, _ in forge._REF_FIELDS:
        assert f"===SEGMENT: {label} ===" in system
    assert result["fields"]["ref"]["subject_definitions"].startswith("<Subject 7>")
    assert result["simple_prompt"] == result["prompt"]
    assert result["prompt"].startswith("subject_definitions:")
    assert result["saw_images"] == 3 and model.unloads == ["test"]


@pytest.mark.parametrize("keep,structured", [(True, True), (False, False)])
def test_generate_continuity_request_forwarding(draft_context, monkeypatch, keep, structured):
    from types import SimpleNamespace
    inputs, tail, refs = draft_context
    monkeypatch.setitem(sys.modules, "nodes.h3_continuity.core", SimpleNamespace(
        ClipStore=lambda: object(), safe_id=lambda x: x,
        continuation_timing=lambda *a: {"extension_frames": 119, "added_seconds": 5}))
    monkeypatch.setitem(sys.modules, "nodes.h3_continuity.video_source", SimpleNamespace(
        read_manifest=lambda *a: {"frames": 100, "clip_id": "source"}, manifest_dir=lambda *a: str(tail)))
    model = Model(structured_raw() if structured else "Continue stepping.")
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    body = {"model": "openai:test", "mode": "REF2VA", "duration": 5, "references": refs,
            "output_canvas": {"width": 720, "height": 1280},
            "existing_definitions": "<Subject 7>: dancer", "structured": structured,
            "continuity": {"session": "session", "clip_id": "source", "source_kind": "video", "use_references": keep}}
    result = forge.generate(body, str(inputs))
    assert model.calls[0][3] == (["ref.png", "old.jpg", "new.jpg"] if keep else ["old.jpg", "new.jpg"])
    assert bool(result["fields"]) == structured
    assert result["simple_prompt"] == result["prompt"]
    assert "<Subject 7>: dancer" in model.calls[0][2]
    assert "720x1280 pixels; aspect ratio 9:16" in model.calls[0][2]


@pytest.mark.parametrize("structured", [False, True])
def test_continuation_warns_undefined_current_media(draft_context, monkeypatch, structured):
    inputs, tail, refs = draft_context
    raw = structured_raw("<Subject 7>: <Picture 9> and <Video 2> and <Audio 1>") if structured else "Follow <Picture 9>."
    model = Model(raw, False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate_continuity_draft({}, "", str(tail), "openai:test", {},
        references=refs, input_directory=str(inputs), structured=structured)
    assert model.calls[0][3] == []
    assert "Do not invent" in model.calls[0][2]
    assert any("<Picture 9>" in w and "undefined" in w for w in result["warnings"])
    if structured:
        assert any("<Video 2>" in w for w in result["warnings"])
        assert any("<Audio 1>" in w for w in result["warnings"])


def test_base_endpoint_pose_cannot_override_alignment():
    text = "\n".join(forge.format_references([{ "kind": "image", "role": "pose", "instructions": "Only arms"}], "I2VA")[0])
    assert "endpoint" in text and "Only arms" in text


def test_plain_references_distinguish_image_evidence(draft_context, monkeypatch):
    inputs, tail, refs = draft_context
    model = Model("Continue stepping.")
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    forge.generate_continuity_draft({}, "", str(tail), "openai:test", {}, input_directory=str(inputs), references=refs)
    user = model.calls[0][2]
    assert "The attached images, if present, are chronological frames" not in user
    assert "Tail frame 1" in user and "Only arms" in user


@pytest.mark.parametrize("canvas,ratio", [
    ({"width": 1280, "height": 720}, "16:9"),
    ({"width": 768, "height": 1360}, "48:85"),
    ({"width": 1024, "height": 1024}, "1:1"),
])
def test_normal_forge_director_canvas_is_authoritative(monkeypatch, canvas, ratio):
    model = Model(structured_raw(), False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    forge.generate({"mode": "REF2VA", "brief": "Dance in 21:9", "model": "openai:test", "output_canvas": canvas})
    user = model.calls[0][2]
    assert f"{canvas['width']}x{canvas['height']} pixels; aspect ratio {ratio}" in user
    assert "over conflicting format requests" in user
    assert "reference-image dimensions and example formats are not" in user


def test_unknown_canvas_omits_ratio_in_normal_and_continuity_requests(draft_context, monkeypatch):
    _, tail, _ = draft_context
    model = Model(structured_raw(), False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    forge.generate({"mode": "REF2VA", "brief": "Dance", "model": "openai:test", "output_canvas": None})
    forge.generate_continuity_draft({}, "Continue", str(tail), "openai:test", {})
    for call in model.calls:
        assert "Omit aspect ratio and resolution" in call[2]


def test_continuity_uses_director_canvas_not_source_canvas(draft_context, monkeypatch):
    _, tail, _ = draft_context
    model = Model(structured_raw(), False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    forge.generate_continuity_draft({"width": 1920, "height": 1080}, "Continue", str(tail), "openai:test", {},
        options={"output_canvas": {"width": 720, "height": 1280}, "creativity": "wild", "detail": 9}, structured=True)
    user = model.calls[0][2]
    assert "720x1280 pixels; aspect ratio 9:16" in user
    assert "one uninterrupted continuation: no cuts" in user


@pytest.mark.parametrize("canvas", [{}, "16:9", {"width": True, "height": 720},
    {"width": 0, "height": 720}, {"width": 1280.5, "height": 720}, {"width": 9000, "height": 720}])
def test_invalid_output_canvas_is_rejected(canvas):
    with pytest.raises(forge.ForgeError) as exc:
        forge.output_canvas_context(canvas)
    assert exc.value.code == "bad_canvas"


def test_shot_policy_separates_detail_creativity_and_explicit_instructions():
    bundle = forge.load_bundle()
    assert bundle is not None
    for spec in bundle["modes"].values():
        assert "Detail controls descriptive precision only, not shot count" in spec["system"]
        assert "single-shot/no-cut or multi-shot instructions" in spec["system"]
        assert "Single-shot description example" in spec["system"]
        assert "not the default shot count" in spec["system"]
        assert "If the canvas is unknown, omit aspect ratio" in spec["system"]
    for entry in bundle["detail_levels"].values():
        assert "own cut or camera change" not in entry["rule"]
        assert "not shot count" in entry["rule"] or "without adding cuts" in entry["rule"]
    for key in ("literal", "precise", "faithful"):
        assert "no cuts" in bundle["creativity_presets"][key]["rule"]
    for key in ("inventive", "creative", "daring", "wild"):
        assert "unexpected" in bundle["creativity_presets"][key]["rule"] or "surprising" in bundle["creativity_presets"][key]["rule"]


def test_normal_forge_notes_are_bounded_and_text_only(monkeypatch):
    model = Model(structured_raw(), False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate({"mode": "REF2VA", "brief": "Dance", "model": "openai:test",
        "existing_definitions": "<Subject 9>: approved dancer",
        "references": [{"kind": "image", "role": "custom", "instructions": "Only layout", "path": "a.png"}]})
    user = model.calls[0][2]
    assert "<Subject 9>: approved dancer" in user and "IDs" in user
    assert "refine/override" in user and "Do not invent" in user
    assert "A reference picture is attached" not in user
    assert result["saw_images"] == 0


def test_saved_reference_is_not_claimed_attached(draft_context, monkeypatch):
    inputs, _, refs = draft_context
    refs.append({"kind": "image", "role": "subject", "saved_reference": True, "instructions": "Saved dancer"})
    model = Model(structured_raw())
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate({"mode": "REF2VA", "brief": "Dance", "model": "openai:test", "references": refs}, str(inputs))
    user = model.calls[0][2]
    assert "pictures for <Picture 1>, <Picture 2> are attached" not in user
    assert "picture for <Picture 1> is attached" in user
    assert "<Picture 2>" in user and "not visible" in user
    assert result["saw_images"] == 1


@pytest.mark.parametrize("structured", [False, True])
def test_continuity_reports_output_format(draft_context, monkeypatch, structured):
    _, tail, _ = draft_context
    model = Model(structured_raw() if structured else "Continue.", False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate_continuity_draft({}, "", str(tail), "openai:test", {}, structured=structured)
    assert result["structured"] is structured


@pytest.mark.parametrize("raw,code", [("No markers", "no_segments"),
    ("===SEGMENT: Summary ===\nSummary only", "missing_segments"),
    (structured_raw() + "x" * 8000, "bad_prompt")])
def test_structured_invalid_output_unloads(draft_context, monkeypatch, raw, code):
    _, tail, _ = draft_context
    model = Model(raw, False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    with pytest.raises(forge.ForgeError) as exc:
        forge.generate_continuity_draft({}, "", str(tail), "openai:test", {}, structured=True)
    assert exc.value.code == code
    assert model.unloads == ["test"]


@pytest.mark.parametrize("path", ["../outside.png", "/etc/passwd", "missing.png"])
def test_reference_paths_are_constrained(draft_context, monkeypatch, path):
    inputs, tail, refs = draft_context
    refs[0]["path"] = path
    model = Model("Continue.")
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    with pytest.raises(forge.ForgeError) as exc:
        forge.generate_continuity_draft({}, "", str(tail), "openai:test", {},
            references=refs, input_directory=str(inputs))
    assert exc.value.code == "bad_references"
    assert not model.calls


def test_plain_default_contract_unchanged(draft_context, monkeypatch):
    _, tail, _ = draft_context
    model = Model("<think>private</think>Continue stepping.", False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate_continuity_draft({"prompt": "prior"}, "next", str(tail), "openai:test", {})
    assert model.calls[0][1].startswith(forge.CONTINUATION_SYSTEM)
    assert "must not replace or suppress" in model.calls[0][1]
    assert "Current conditioning references" not in model.calls[0][2]
    assert result["prompt"] == "Continue stepping." and result["fields"] == {}


def test_image_rejection_retries_text_and_unloads(draft_context, monkeypatch):
    from urllib.error import HTTPError
    inputs, tail, refs = draft_context
    model = Model(structured_raw(), None)
    def chat(*args):
        model.calls.append(args)
        if len(model.calls) == 1:
            raise HTTPError("http://remote", 400, "No vision", {}, None)
        return model.raw, {}
    model.chat = chat
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate_continuity_draft({}, "", str(tail), "openai:test", {},
        references=refs, input_directory=str(inputs), structured=True)
    assert model.calls[1][3] == []
    assert "Ignore attachment claims" in model.calls[1][2]
    assert result["saw_images"] == 0 and model.unloads == ["test"]


def test_cancellation_during_generation_unloads(draft_context, monkeypatch):
    import threading
    _, tail, _ = draft_context
    stop = threading.Event()
    model = Model("Continue.", False)
    def chat(*args):
        stop.set()
        return model.raw, {}
    model.chat = chat
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    with pytest.raises(forge.ForgeError) as exc:
        forge.generate_continuity_draft({}, "", str(tail), "openai:test", {}, cancel=stop)
    assert exc.value.code == "cancelled" and model.unloads == ["test"]


def test_normal_forge_invalid_path_is_request_error(draft_context, monkeypatch):
    inputs, _, refs = draft_context
    refs[0]["path"] = "../outside.png"
    model = Model(structured_raw())
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    with pytest.raises(forge.ForgeError) as exc:
        forge.generate({"mode": "REF2VA", "brief": "Dance", "model": "openai:test", "references": refs}, str(inputs))
    assert exc.value.code == "bad_references"


def test_normal_forge_checks_existing_media_citations(monkeypatch):
    model = Model(structured_raw("<Subject 7>: from <Picture 8>"), False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate({"mode": "REF2VA", "brief": "Dance", "model": "openai:test",
        "existing_definitions": "<Subject 9>: <Audio 2>"})
    assert any("<Picture 8>" in w for w in result["warnings"])
    assert any("<Audio 2>" in w for w in result["warnings"])


def test_structured_group_split_warning(draft_context, monkeypatch):
    _, tail, _ = draft_context
    refs = [{"kind": "image", "role": "subject", "subject_group": "a"}] * 2
    model = Model(structured_raw("<Subject 1>: <Picture 1>\n<Subject 2>: <Picture 2>"), False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.generate_continuity_draft({}, "", str(tail), "openai:test", {}, references=refs, structured=True)
    assert any("separate Subjects" in w for w in result["warnings"])


@pytest.mark.parametrize("options", [{"structured": "true"}, {"existing_definitions": 1},
    {"existing_definitions": "x" * 12001}, {"structured": True, "options": {"mode": "I2VA"}}])
def test_invalid_continuity_options_rejected(draft_context, monkeypatch, options):
    _, tail, _ = draft_context
    model = Model("Continue.", False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    with pytest.raises(forge.ForgeError):
        forge.generate_continuity_draft({}, "", str(tail), "openai:test", {}, **options)
    assert not model.calls


def test_reference_note_does_not_define_a_numbered_media_slot():
    refs = [{"kind": "image", "instructions": "Ignore <Picture 9>; use only this image"},
            {"kind": "video", "stream": "both"}, {"kind": "audio"}]
    warnings = forge.media_citation_warnings("<Picture 1> <Picture 9> <Video 1> <Audio 1> <Audio 2>", refs)
    assert len(warnings) == 1 and "<Picture 9>" in warnings[0]
