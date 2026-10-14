"""People boxes: who is in a picture and where, from a box round each character."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_forge as forge
from nodes import h3_people as people
from nodes import h3_prompting as hp
from nodes.llm_backends import ForgeError

# Five people side by side, as in a band picture.
BAND = [{"n": n, "box": [0.2 * (n - 1) + 0.01, 0.1, 0.2 * n - 0.01, 0.95]} for n in range(1, 6)]
# A piggyback: the one carried overlaps the one carrying by most of her box.
PIGGYBACK = [{"n": 1, "box": [0.30, 0.25, 0.70, 1.0]}, {"n": 2, "box": [0.35, 0.05, 0.75, 0.70]}]


def pic(easy_role="character-1", **extra):
    return {"kind": "image", "role": "subject", "easy_role": easy_role, "path": "p.png", **extra}


def test_boxes_are_checked_not_guessed():
    ref = {"boxes": [
        {"n": 2, "box": [0.6, 0.9, 0.1, 0.2]},       # corners swapped: put right
        {"n": 2, "box": [0, 0, 1, 1]},               # number taken twice: the second goes
        {"n": 3, "box": [0.1, 0.1, 0.105, 0.5]},     # a mis-drag, too thin
        {"n": 0, "box": [0, 0, 1, 1]},               # no Character 0
        {"n": True, "box": [0, 0, 1, 1]},            # not a number
        {"n": 4, "box": [0, 0, 1]},                  # three corners
        {"n": 5, "box": [-0.2, 0.1, 1.4, 0.9]},      # past the edges: clamped
    ]}
    assert people.boxes_of(ref) == [{"n": 2, "box": [0.1, 0.2, 0.6, 0.9]}, {"n": 5, "box": [0.0, 0.1, 1.0, 0.9]}]
    assert people.boxes_of({}) == [] and people.boxes_of({"boxes": "1,2"}) == []


def test_boxes_order_people_left_to_right():
    boxes = [{"n": 3, "box": [0.6, 0, 0.9, 1]}, {"n": 1, "box": [0.0, 0, 0.3, 1]}, {"n": 2, "box": [0.3, 0, 0.6, 1]}]
    assert people.who_from_boxes(boxes) == [1, 2, 3]
    assert hp.picture_who(pic(boxes=boxes, who=[3, 2, 1], picture_kinds=["character"])) == [1, 2, 3]


def test_a_pose_keeps_its_takers_whatever_boxes_it_holds():
    ref = pic("pose", role="pose", picture_kinds=["pose"], who=[2, 1], boxes=BAND)
    assert hp.picture_who(ref) == [2, 1]


def test_crops_only_where_a_box_shows_one_person():
    band = people.boxes_of({"boxes": BAND})
    assert people.crop_for(band, 3) == band[2]["box"]
    assert people.crop_for(band, 9) is None
    piggyback = people.boxes_of({"boxes": PIGGYBACK})
    assert people.crop_for(piggyback, 1) is None and people.crop_for(piggyback, 2) is None


def test_positions_go_past_four_when_boxes_give_the_order():
    assert people.boxed_places(5) == ("furthest left", "second from the left", "third from the left",
                                      "fourth from the left", "furthest right")
    assert hp.place_word("second from the left") == "second from the left"
    assert hp.place_word("furthest right") == "furthest right"
    cast = hp.easy_cast([pic(picture_kinds=["character"], who=[1], boxes=BAND)])
    places = [s["placements"][0]["position"] for s in cast["subjects"]]
    assert places == list(people.boxed_places(5))
    lines = "\n".join(hp.easy_lines(cast))
    assert '<Subject 2> is "Character 2" in the brief, a character, shown second from the left in <Picture 1>' in lines
    assert '<Subject 5> is "Character 5" in the brief, a character, shown furthest right in <Picture 1>' in lines


def test_two_to_four_boxes_use_the_usual_words_whatever_order_was_picked():
    two = [{"n": 2, "box": [0.0, 0, 0.4, 1]}, {"n": 1, "box": [0.5, 0, 0.9, 1]}]
    cast = hp.easy_cast([pic(picture_kinds=["character"], who=[1, 2], who_axis="z", boxes=two)])
    where = {s["number"]: s["placements"][0]["position"] for s in cast["subjects"]}
    assert where == {2: "left", 1: "right"}


def test_frame_boxes_place_the_brief_names():
    two = [{"n": 1, "box": [0.55, 0, 0.95, 1]}, {"n": 2, "box": [0.05, 0, 0.45, 1]}]
    refs = [{"kind": "image", "role": "keyframe", "path": "f.png", "who": [1, 2], "boxes": two}]
    brief = hp.framed_brief("Character 1 hugs Character 2.", refs, "I2VA")
    assert brief == "the character on the right in the first frame hugs the character on the left in the first frame."


def test_frames_with_many_boxes_place_everyone():
    refs = [{"kind": "image", "role": "keyframe", "path": "f.png", "boxes": BAND}]
    brief = hp.framed_brief("Character 5 high-fives Character 2.", refs, "L2VA")
    assert brief == "the character furthest right in the last frame high-fives the character second from the left in the last frame."


@pytest.mark.parametrize("boxes", ["1,2", [{"n": 1, "box": [0, 0, 1]}], [{"n": 1}], [{"n": i, "box": [0, 0, 1, 1]} for i in range(33)]])
def test_bad_boxes_are_refused(boxes):
    with pytest.raises(ForgeError):
        hp.validate_references([pic(boxes=boxes)])


def test_find_people_answers_are_read_leniently():
    # A stray ")" inside the JSON, as a 4B wrote on a five-person picture.
    text = ('```json\n[{"label": "girl, red hair", "bbox_2d": [520, 40, 900, 990]}),\n'
            '{"label": "boy", "bbox_2d": [100, 60, 480, 995]}]\n```')
    found = people.parse_people(text)
    assert [p["label"] for p in found] == ["boy", "girl, red hair"]
    assert found[0]["box"] == [0.1, 0.06, 0.48, 0.995]
    # Past 1000: pixels of the picture that was sent.
    pixels = people.parse_people('[{"label": "a", "bbox_2d": [0, 0, 1536, 512]}]', (2048, 1024))
    assert pixels[0]["box"] == [0.0, 0.0, 0.75, 0.5]
    assert people.parse_people("I see two people.") == []


def test_crop_image_cuts_the_box(tmp_path):
    from PIL import Image
    path = tmp_path / "p.png"
    Image.new("RGB", (400, 200), "blue").save(path)
    import base64
    import io
    cut = Image.open(io.BytesIO(base64.b64decode(people.crop_image_b64(str(path), [0.25, 0.0, 0.5, 1.0]))))
    assert cut.size == (116, 200)  # a quarter of the width plus a 2% margin each side


class Seeing:
    def __init__(self, raw, sees=None):
        self.raw, self.sees, self.calls, self.unloads, self.base = raw, sees, [], [], "http://127.0.0.1:8099"

    def can_see(self, name):
        return self.sees

    def unload(self, name):
        self.unloads.append(name)
        return True

    def chat(self, *args):
        self.calls.append(args)
        return self.raw, {}


@pytest.fixture
def picture(tmp_path):
    from PIL import Image
    Image.new("RGB", (2048, 1024), "white").save(tmp_path / "band.png")
    return tmp_path


def test_find_people_boxes_with_the_picked_model_and_unloads(picture, monkeypatch):
    model = Seeing('[{"label": "a", "bbox_2d": [600, 0, 900, 1000]}, {"label": "b", "bbox_2d": [100, 0, 400, 1000]}]')
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.find_people({"model": "openai:qwen-vl", "path": "band.png"}, str(picture))
    assert [p["label"] for p in result["people"]] == ["b", "a"]
    _, system, user, images, sampling, *_ = model.calls[0]
    assert user == people.FIND_PEOPLE_PROMPT and len(images) == 1 and sampling["temperature"] == 0
    assert model.unloads == ["qwen-vl"] and result["unloaded"]
    assert "raw" not in result


def test_find_people_says_why_when_nobody_is_boxed(picture, monkeypatch):
    model = Seeing("There are two people: a boy and a girl.")
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    result = forge.find_people({"model": "openai:writer", "path": "band.png"}, str(picture))
    assert result["people"] == [] and result["raw"].startswith("There are two people")
    assert model.unloads == ["writer"]


def test_find_people_refuses_a_model_that_cannot_see(picture, monkeypatch):
    model = Seeing("", sees=False)
    monkeypatch.setattr(forge, "backends", lambda settings: {"ollama": model})
    with pytest.raises(ForgeError) as err:
        forge.find_people({"model": "ollama:text-only", "path": "band.png"}, str(picture))
    assert err.value.code == "no_vision" and "Add character" in err.value.message
    assert not model.calls
