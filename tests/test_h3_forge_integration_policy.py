"""Integration policy: idea priority and cancellation before a reader starts."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from nodes import h3_forge as forge, h3_prompting as hp


def test_card_traits_are_reference_baselines_not_outfit_overrides():
    refs = [{"kind": "image", "path": "a.png", "easy_role": "character-1"}]
    cast = hp.easy_cast(refs)
    message = hp.build_user_message(hp.load_bundle(), "Character 1 now wears a red jacket", "REF2VA", 5, 5, "balanced", refs, False, cast=cast, cards={1: "An adult in a blue jacket."})
    assert "Explicit appearance, outfit, setting or style changes" in message
    assert "take priority over conflicting card traits" in message
    fields = {"Detailed description": "[Shot 1] <Subject 1> wears a red jacket.", "Subject definitions": "N/A"}
    hp.easy_segments(cast, fields, [], {"<Subject 1>": "An adult in a blue jacket."})
    assert "explicit changes in detailed_description override conflicting traits" in fields["Subject definitions"]
    assert "except traits explicitly changed" in fields["Retention analysis"]
    assert fields["Detailed description"] == "[Shot 1] <Subject 1> wears a red jacket."


@pytest.mark.parametrize("reader", [forge.find_people, forge.read_cast])
def test_queued_reads_can_be_cancelled_without_loading_a_model(monkeypatch, reader):
    entered = threading.Event()
    released = threading.Event()

    class HeldLock:
        def acquire(self, timeout):
            entered.set()
            assert released.wait(2)
            return True

        def release(self):
            pass

    monkeypatch.setattr(forge, "_REQUEST_LOCK", HeldLock())
    monkeypatch.setattr(forge, "backends", lambda *_: pytest.fail("Cancelled read must not discover/load models"))
    body = {"request_id": "queued-read-policy"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(reader, body, "/unused")
        try:
            assert entered.wait(2)
            assert forge.cancel(body["request_id"])
        finally:
            released.set()
        with pytest.raises(hp.ForgeError) as exc:
            result.result(timeout=2)
        assert exc.value.code == "cancelled"
    assert body["request_id"] not in forge._CANCELS


class ServerModel:
    base = "http://remote.invalid"
    def __init__(self):
        self.unloads = []
    def can_see(self, name):
        return False
    def unload(self, name):
        self.unloads.append(name)
        return True
    def loaded(self):
        return []


@pytest.fixture
def warm_state(monkeypatch):
    class Timer:
        def __init__(self, *args, **kwargs): self.cancelled = False
        def start(self): pass
        def cancel(self): self.cancelled = True
    monkeypatch.setattr(forge.threading, "Timer", Timer)
    monkeypatch.setattr(forge, "_WARM", {})
    monkeypatch.setattr(forge, "_FORGE_LOADED", set())


@pytest.mark.parametrize("missing_image", [False, True])
def test_failed_draft_preparation_keeps_warm_cleanup(warm_state, monkeypatch, tmp_path, missing_image):
    backend = ServerModel()
    forge._keep_warm("openai", backend, "r4")
    timer = next(iter(forge._WARM.values()))[2]
    next_backend = ServerModel()
    next_backend.can_see = lambda name: True
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": next_backend})
    body = {"model": "openai:r4", "mode": "REF2VA" if missing_image else "T2VA", "brief": "An adult waves.", "duration": 5, "num_ctx": "bad"}
    if missing_image:
        body.update(easy=True, see_pictures=True, references=[{"kind": "image", "path": "missing.png", "easy_role": "character-1"}])
    with pytest.raises(hp.ForgeError if missing_image else ValueError):
        forge.generate(body, str(tmp_path))
    assert len(forge._WARM) == 1 and not timer.cancelled
    forge.release_warm()
    assert backend.unloads == ["r4"] and not forge._FORGE_LOADED


def test_repeated_warming_replaces_backend_ownership(warm_state):
    models = [ServerModel() for _ in range(3)]
    for backend in models:
        forge._FORGE_LOADED.add((backend, "r4"))
        forge._keep_warm("openai", backend, "r4")
    assert forge._FORGE_LOADED == {(models[-1], "r4")}
    forge.release_warm()
    assert not forge._FORGE_LOADED
    assert models[-1].unloads == ["r4"]


def test_director_unloads_owned_models_without_loaded_enumeration(warm_state):
    model = ServerModel()
    forge._FORGE_LOADED.add((model, "r4"))
    forge.unload_forge_models()
    assert model.unloads == ["r4"]


@pytest.mark.parametrize("text", ["A child in a white dress.", "A girl with red hair.", "A teenage boy."])
def test_generated_and_edited_cards_cannot_bypass_age_policy(text):
    from nodes import h3_cast
    for validate in (h3_cast.written_look, lambda t: h3_cast.given_cast({1: t})):
        with pytest.raises(hp.ForgeError) as exc:
            validate(text)
        assert exc.value.code == "bad_cast"


def test_reader_rejects_minor_output_before_a_card_is_returned(tmp_path, monkeypatch):
    from PIL import Image
    Image.new("RGB", (32, 32)).save(tmp_path / "a.png")
    model = ServerModel()
    model.can_see = lambda _: True
    def chat(name, system, *args, **kwargs):
        assert "Every human you describe is an adult." in system
        return "A child in a white dress.\nKept: white dress", {}
    model.chat = chat
    monkeypatch.setattr(forge, "backends", lambda _: {"openai": model})
    with pytest.raises(hp.ForgeError) as exc:
        forge.read_cast({"model": "openai:r4", "mode": "REF2VA", "character": 1, "references": [{"kind": "image", "path": "a.png", "easy_role": "character-1"}]}, str(tmp_path))
    assert exc.value.code == "bad_cast"
    assert model.unloads == ["r4"]
