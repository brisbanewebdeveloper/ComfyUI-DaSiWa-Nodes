"""People boxes: a box round each character in a picture of several.

A port of PromptForge's server/people-boxes.mjs. The person draws the boxes in
the Forge panel, or the model finds them ("Find people"), and either way they
are edited the same: dragged, resized, renumbered. A box does two jobs. Its
place left to right orders the people, so the Who buttons and the
left/top/front order are not needed on that picture. And a reader can
describe that person from a crop of the box rather than the whole picture.

Measured in PromptForge, 13 Oct 2026, on a five-girl band in matching
uniforms: read from the whole picture each girl got the same uniform
sentence; read from her crop, the details that tell them apart came back. On
a piggyback the crop of the girl riding on his back held his head and chest
too, and the reader described him - so heavily overlapping boxes are read
from the whole picture by position instead (crop_for).

Boxes are fractions of the picture: [x1, y1, x2, y2], 0 to 1.
"""

import base64
import io
import re

# The Forge's Who buttons go to 32 characters; boxes stop there too.
MAX_BOXES = 32
# A box smaller than this, as a fraction of the picture, is a mis-drag.
MIN_SIDE = 0.02
# Overlap past this share of the smaller box means the crop would hold
# someone else as much as its own person. The piggyback's two boxes overlap
# by about 60% of the smaller; the band's neighbours by under 25%.
OVERLAP_LIMIT = 0.35
# Room kept round a crop, as a fraction of the picture.
MARGIN = 0.02


def _clamp01(v):
    return min(1.0, max(0.0, v))


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def boxes_of(ref, limit=MAX_BOXES):
    """A picture's boxes, checked: [{"n", "box"}], one per character number.
    Anything malformed is skipped, never guessed at."""
    raw = ref.get("boxes") if isinstance(ref, dict) else None
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for b in raw[:limit]:
        if not isinstance(b, dict):
            continue
        n, box = b.get("n"), b.get("box")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= limit or n in seen:
            continue
        if not isinstance(box, list) or len(box) != 4 or not all(_number(v) for v in box):
            continue
        x1, y1 = _clamp01(min(box[0], box[2])), _clamp01(min(box[1], box[3]))
        x2, y2 = _clamp01(max(box[0], box[2])), _clamp01(max(box[1], box[3]))
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            continue
        seen.add(n)
        out.append({"n": n, "box": [x1, y1, x2, y2]})
    return out


def who_from_boxes(boxes):
    """The character numbers in a picture, left to right by where their boxes sit."""
    return [b["n"] for b in sorted(boxes, key=lambda b: b["box"][0] + b["box"][2])]


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def boxed_places(count):
    """Positions for boxed people past four, left to right: "furthest left",
    "second from the left" ... "furthest right". Boxes give the order, so
    nobody is left without a place (PromptForge, 13 Oct 2026: five boxed
    people in a frame kept a bare "Character 5", and the writer gave the
    high-five to the wrong girl)."""
    return tuple("furthest left" if i == 0 else "furthest right" if i == count - 1
                 else f"{_ORDINALS[i] if i < len(_ORDINALS) else f'number {i + 1}'} from the left"
                 for i in range(count))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_share(a, b):
    inter = _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return inter / max(1e-9, min(_area(a), _area(b)))


def crop_for(boxes, n):
    """The box to read character `n` from, or None: no box for them, or it
    overlaps another person's box too much to show them alone."""
    own = next((b for b in boxes if b["n"] == n), None)
    if own is None:
        return None
    if any(b is not own and _overlap_share(own["box"], b["box"]) > OVERLAP_LIMIT for b in boxes):
        return None
    return own["box"]


def crop_image_b64(path, box, max_edge=1024):
    """The picture cut to a box (plus a small margin), as base64 JPEG."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        x1, y1 = round(_clamp01(box[0] - MARGIN) * w), round(_clamp01(box[1] - MARGIN) * h)
        x2 = max(x1 + 1, round(_clamp01(box[2] + MARGIN) * w))
        y2 = max(y1 + 1, round(_clamp01(box[3] + MARGIN) * h))
        im = im.crop((x1, y1, x2, y2))
        im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def picture_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


# What the model is asked when the person presses Find people. Qwen-VL models
# answer with bbox_2d on a 0-1000 grid.
FIND_PEOPLE_PROMPT = ("Find every person in this picture. Answer with JSON only: a list of "
                      '{"label": short description, "bbox_2d": [x1, y1, x2, y2]}, one per person, covering their whole body.')

_FOUND = re.compile(r'"label"\s*:\s*"([^"]*)"\s*,\s*"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)')


def parse_people(text, size=None):
    """The model's answer as boxes, left to right: [{"box", "label"}].

    Read leniently, by pulling each label and its four numbers: a 4B put a
    stray ")" inside the JSON on a five-person picture. Qwen-VL answers on a
    0-1000 grid; numbers past 1000 are taken as pixels of a picture `size`
    (width, height) big."""
    found = []
    for m in _FOUND.finditer(str(text or "")):
        try:
            found.append((m.group(1).strip(), [float(v) for v in m.groups()[1:]]))
        except ValueError:
            continue
    pixels = size and any(v > 1000 for _, raw in found for v in raw)
    people = []
    for label, raw in found:
        if pixels:
            x1, y1, x2, y2 = raw[0] / size[0], raw[1] / size[1], raw[2] / size[0], raw[3] / size[1]
        else:
            x1, y1, x2, y2 = (v / 1000 for v in raw)
        box = [_clamp01(min(x1, x2)), _clamp01(min(y1, y2)), _clamp01(max(x1, x2)), _clamp01(max(y1, y2))]
        if box[2] - box[0] >= MIN_SIDE and box[3] - box[1] >= MIN_SIDE:
            people.append({"label": label, "box": box})
    return sorted(people[:MAX_BOXES], key=lambda p: p["box"][0] + p["box"][2])
