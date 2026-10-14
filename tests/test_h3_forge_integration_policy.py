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
