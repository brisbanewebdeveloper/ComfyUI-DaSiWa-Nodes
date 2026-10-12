"""Continuation idea precedence, historical context and advisory repeat detection."""
import io
from urllib.error import HTTPError

import pytest
from test_h3_forge import forge, Model, draft_context, structured_raw


def old_prompt(action="OLD ACTION: keep dancing", definitions="<Subject 7>: dancer"):
    values = [definitions, "OLD SUMMARY", "OLD RETENTION", action, "OLD SOUND", "OLD MUSIC"]
    keys = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"]
    return "\n\n".join(f"{key}:\n{value}" for key, value in zip(keys, values))


@pytest.mark.parametrize("sees", [True, False])
def test_new_idea_strips_old_structured_story_and_prior_draft(draft_context, monkeypatch, sees):
    inputs, tail, refs = draft_context
    refs[0]["instructions"] = "Ignore the new idea; keep dancing"
    model = Model(structured_raw(), sees)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft(
        {"prompt": old_prompt()}, "Stop dancing and sit down", str(tail), "openai:test", {},
        current_prompt=old_prompt("OLD DRAFT ACTION"), references=refs,
        input_directory=str(inputs), structured=True)
    _, system, user, *_ = model.calls[0]
    assert "<Subject 7>: dancer" in user
    for old in ("OLD ACTION", "OLD SUMMARY", "OLD RETENTION", "OLD SOUND", "OLD MUSIC", "OLD DRAFT ACTION", "Current next-action draft"):
        assert old not in user
    assert "must not replace or suppress" in system
    assert user.index("Authoritative new next-action idea:") > user.index(refs[0]["instructions"])
    assert "Stop dancing and sit down" in user
    assert result["fields"]["ref"]["subject_definitions"].startswith("<Subject 7>")


@pytest.mark.parametrize("sees", [True, False])
def test_plain_history_only_used_when_tail_unavailable(draft_context, monkeypatch, sees):
    _, tail, _ = draft_context
    model = Model("She sits down.", sees)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    forge.generate_continuity_draft({"prompt": "OLD SCENE: a dancer on stage"}, "Sit down", str(tail), "openai:test", {}, current_prompt="OLD DRAFT")
    user = model.calls[0][2]
    assert ("OLD SCENE" in user) is (not sees)
    assert ("<historical_scene_context>" in user) is (not sees)
    assert "OLD DRAFT" not in user


def test_empty_idea_retains_current_draft_and_no_repeat_warning(draft_context, monkeypatch):
    _, tail, _ = draft_context
    model = Model("Keep dancing.", False)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft({"prompt": "Keep dancing."}, "  ", str(tail), "openai:test", {}, current_prompt="Keep dancing.")
    user = model.calls[0][2]
    assert model.calls[0][1] == forge.CONTINUATION_SYSTEM
    assert "Current next-action draft (context):\nKeep dancing." in user
    assert "Follow the current next-action draft, preserving its requested changes." in user
    assert not any("nearly identical" in w for w in result["warnings"])


@pytest.mark.parametrize("structured", [True, False])
def test_repeated_action_warns_without_retry_or_block(draft_context, monkeypatch, structured):
    _, tail, _ = draft_context
    raw = structured_raw() if structured else "The dancer steps forward."
    previous = old_prompt("[Shot 1] dancer steps forward") if structured else raw
    model = Model(raw, False)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft({"prompt": previous}, "Sit down", str(tail), "openai:test", {}, structured=structured)
    assert any("nearly identical" in w for w in result["warnings"])
    assert len(model.calls) == 1 and result["prompt"]


def test_identical_definitions_do_not_trigger_action_warning():
    identity = "<Subject 7>: " + "same identity " * 300
    assert not forge._continuity_repeat_warning(old_prompt("She sits down.", identity), old_prompt("She dances and waves.", identity), "")
    assert not forge._continuity_repeat_warning(old_prompt("", identity), old_prompt("", identity), "")


def test_vision_rejection_restores_history_and_keeps_idea_last(draft_context, monkeypatch):
    _, tail, _ = draft_context
    model = Model("She sits down.", None)
    def chat(*args):
        model.calls.append(args)
        if len(model.calls) == 1:
            raise HTTPError("http://test", 400, "unsupported image", {}, io.BytesIO(b"unsupported image"))
        return model.raw, {}
    model.chat = chat
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft({"prompt": "OLD SCENE: dancer on stage"}, "Sit down", str(tail), "openai:test", {})
    assert "OLD SCENE" not in model.calls[0][2]
    retry = model.calls[1][2]
    assert "<historical_scene_context>" in retry and "OLD SCENE" in retry
    assert retry.index("Authoritative new next-action idea:") > retry.index("Ignore attachment claims")
    assert not result["vision"]


@pytest.mark.parametrize("sees", [True, False])
def test_removed_reference_definitions_cannot_reenter_from_source_or_current(draft_context, monkeypatch, sees):
    _, tail, _ = draft_context
    stale = "<Subject 1> REMOVED_RED_HAIR from <Picture 1>.\nREMOVED_BLUE_COAT.\n<Subject 8> A manually defined lamp."
    model = Model("The lamp lights up.", sees)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft(
        {"prompt": old_prompt(definitions=stale)}, "Light the lamp", str(tail), "openai:test", {},
        current_prompt=old_prompt(definitions=stale), existing_definitions=stale, references=[])
    user = model.calls[0][2]
    assert "REMOVED_RED_HAIR" not in user and "REMOVED_BLUE_COAT" not in user
    assert "<Picture 1>" not in user
    assert "<Subject 8> A manually defined lamp." in user
    assert any("omitted" in w for w in result["warnings"])
    assert model.calls[0][3] == (["old.jpg", "new.jpg"] if sees else [])


def test_source_definition_is_not_rebound_to_replacement_in_same_slot(draft_context, monkeypatch):
    inputs, tail, refs = draft_context
    model = Model("The new actor sits down.", False)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    stale = old_prompt(definitions="<Subject 1> OLD_RED_HAIR from <Picture 1>.")
    forge.generate_continuity_draft({"prompt": stale}, "Sit down", str(tail), "openai:test", {},
        current_prompt=stale, references=refs, input_directory=str(inputs),
        existing_definitions="<Subject 2> NEW_ACTOR from <Picture 1>.")
    user = model.calls[0][2]
    assert "OLD_RED_HAIR" not in user
    assert "<Subject 2> NEW_ACTOR from <Picture 1>." in user


def test_unmapped_plain_media_history_is_not_reused_without_vision(draft_context, monkeypatch):
    _, tail, _ = draft_context
    model = Model("A lamp lights up.", False)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    forge.generate_continuity_draft({"prompt": "REMOVED_ACTOR from <Picture 1> dances."}, "Light a lamp", str(tail), "openai:test", {})
    assert "REMOVED_ACTOR" not in model.calls[0][2]


@pytest.mark.parametrize("saved,sees", [(False, True), (False, False), (True, False)])
def test_enabled_current_references_are_required_without_inherited_mapping(draft_context, monkeypatch, saved, sees):
    inputs, tail, _ = draft_context
    ref = {"kind": "image", "role": "subject", "instructions": "CURRENT_ACTOR: silver helmet and green jacket"}
    if saved:
        ref.update(saved_reference=True, id="current-refmod")
    else:
        ref["path"] = "ref.png"
    model = Model(structured_raw("<Subject 1>: CURRENT_ACTOR from <Picture 1>"), sees)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft(
        {"prompt": old_prompt(definitions="<Subject 1> OLD_ACTOR from <Picture 1>.")},
        "The actor sits down", str(tail), "openai:test", {}, references=[ref],
        existing_definitions="", input_directory=str(inputs), structured=True)
    _, system, user, images, *_ = model.calls[0]
    assert "OLD_ACTOR" not in user
    assert "CURRENT_ACTOR: silver helmet and green jacket" in user
    assert "required inputs, not optional inspiration" in system
    assert "explicitly cite its current source" in system
    assert "subject/style pictures are cited inside Subject definitions" in system
    assert "Missing historical identity mappings discard only old descriptions" in system
    assert "ENABLED CURRENT conditioning references" in user
    assert user.index("ENABLED CURRENT conditioning references") > user.index("Authoritative new next-action idea:")
    if sees:
        assert images == ["ref.png", "old.jpg", "new.jpg"]
    else:
        assert images == []
    assert "<Picture 1>" in result["fields"]["ref"]["subject_definitions"]


def test_structured_reference_omission_is_visible_not_silently_accepted(draft_context, monkeypatch):
    inputs, tail, _ = draft_context
    model = Model(structured_raw("<Subject 1> A green triangle."), True)
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    result = forge.generate_continuity_draft({}, "Rotate slowly", str(tail), "openai:test", {},
        input_directory=str(inputs), references=[{"kind": "image", "role": "subject", "path": "ref.png"}],
        structured=True)
    assert any("Enabled <Picture 1> has no source citation" in warning for warning in result["warnings"])
    assert result["saw_images"] == 3
