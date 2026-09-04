"""Regression tests for MiniMax H3 Director reference-video decoding.

Reference videos must be decoded to the target rate by *presentation time*
(VHS-style force_rate): every 1/fps tick inside the trim window emits the
source frame with the nearest PTS. The old implementation indexed the decoded
source sequentially, so a 60 fps clip played back ~2.5x slow and only the
first ~40% of the trim was ever used.
"""
import av
import numpy as np
import pytest
import torch

from nodes.helper_minimax_h3_director import load_video

WIDTH, HEIGHT = 64, 64
TOLERANCE = 0.01  # x264/yuv420p round-trip error for solid blocks


def _encode_solid_video(path, fps, seconds):
    """Encode `seconds` of `fps` solid-color frames.

    Frame i has red channel (i * 3) % 256 so every frame is individually
    identifiable after the codec round trip.
    """
    total = fps * seconds
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = WIDTH, HEIGHT
    stream.pix_fmt = "yuv420p"
    for i in range(total):
        arr = np.full((HEIGHT, WIDTH, 3), 160, dtype=np.uint8)
        arr[:, :, 0] = (i * 3) % 256
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(
            format="yuv420p", width=WIDTH, height=HEIGHT)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def _decode_source(path, trim_start=0.0, trim_end=None):
    """Independent reference decode: (source_times, source_frames) for a trim."""
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "video")
    times, frames = [], []
    for frame in container.decode(stream):
        t = float(frame.pts * frame.time_base)
        if t < trim_start or (trim_end is not None and t >= trim_end):
            continue
        times.append(t)
        frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()).float() / 255.0)
    container.close()
    return np.asarray(times), torch.stack(frames)


def _nearest_source_index(times, tick):
    """Nearest source frame for one tick — the spec, computed with a plain
    argmin so the test never mirrors the helper's searchsorted implementation."""
    return int(np.argmin(np.abs(times - tick)))


def _nearest_source_index_tie_lower(times, tick):
    """Nearest source frame for one tick per the spec: the source frame whose
    PTS is closest to `tick`; on an exact half-tick tie the lower index wins
    (VHS force_rate convention). Mirrors the helper's searchsorted+<= logic so
    the test asserts the documented tie-break instead of argmin's higher pick."""
    pos = int(np.searchsorted(times, tick))
    if pos >= len(times):
        return len(times) - 1
    if pos == 0:
        return 0
    return pos - 1 if abs(times[pos - 1] - tick) <= abs(times[pos] - tick) else pos


def _match_source_index(frames, frame):
    """Index of the source frame equal to `frame` (colors make matches unique)."""
    for index, reference in enumerate(frames):
        if torch.allclose(frame, reference, atol=TOLERANCE):
            return index
    raise AssertionError("output frame matches no source frame")


@pytest.fixture(scope="module")
def video_input_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("h3_video_input")
    for name, fps, seconds in (("fast", 60, 6), ("native", 24, 3), ("slow", 12, 3)):
        _encode_solid_video(directory / f"{name}.mp4", fps, seconds)
    return directory


def test_high_fps_reference_spans_the_full_trim_window(video_input_dir):
    """The reported bug: 6 s @ 60 fps must yield ~144 frames covering all 6 s,
    not the first 2.4 s stretched into 6 s."""
    source_times, source_frames = _decode_source(video_input_dir / "fast.mp4")
    out = load_video("fast.mp4", str(video_input_dir), trim_start=0.0, trim_end=6.0)

    # Mirror the helper's exact tick lattice (np.arange, not k/24.0):
    # floating accumulation in arange(0, 6, 1/24) drifts ~1e-16 from the
    # naive k/24.0, flipping the nearest source frame at half-ticks.
    ticks = np.arange(0.0, 6.0, 1.0 / 24.0)
    ticks = ticks[ticks < 6.0]

    assert abs(out.shape[0] - 144) <= 1
    for k in range(out.shape[0]):
        expected = _nearest_source_index_tie_lower(source_times, ticks[k])
        assert torch.allclose(out[k], source_frames[expected], atol=TOLERANCE), (
            f"output frame {k} is not the source frame nearest to tick {ticks[k]:.4f}s")
    # The last tick must land in the final second of the clip; the old
    # implementation stopped at ~2.4 s (source index 143 of 359).
    assert _nearest_source_index(source_times, ticks[-1]) >= 300


def test_trim_window_is_covered_end_to_end(video_input_dir):
    source_times, source_frames = _decode_source(
        video_input_dir / "fast.mp4", trim_start=1.0, trim_end=3.0)
    out = load_video("fast.mp4", str(video_input_dir), trim_start=1.0, trim_end=3.0)

    # Mirror the helper's exact tick lattice for the [1, 3) trim (see above).
    ticks = np.arange(1.0, 3.0, 1.0 / 24.0)
    ticks = ticks[ticks < 3.0]

    assert abs(out.shape[0] - 48) <= 1
    for k in range(out.shape[0]):
        expected = _nearest_source_index_tie_lower(source_times, ticks[k])
        assert torch.allclose(out[k], source_frames[expected], atol=TOLERANCE)
    # First/last ticks land at the trimmed window edges (source ~0/~119
    # local). Asserted via the expected index, not the color matcher: the
    # (i*3)%256 red channel wraps inside a 120-frame window, so frames 85
    # apart differ by <1/255 in red and the matcher is not injective here.
    assert _nearest_source_index(source_times, ticks[0]) <= 2
    assert _nearest_source_index(source_times, ticks[-1]) >= len(source_times) - 3


def test_native_24fps_source_is_unchanged(video_input_dir):
    """At a matching source rate the output must be the untouched frame batch."""
    source_times, source_frames = _decode_source(
        video_input_dir / "native.mp4", trim_start=0.0, trim_end=2.5)
    out = load_video("native.mp4", str(video_input_dir), trim_start=0.0, trim_end=2.5)

    assert out.shape[0] == 60
    assert torch.allclose(out, source_frames, atol=TOLERANCE)


def test_slow_source_repeats_frames_at_target_rate(video_input_dir):
    """A 12 fps source at the 24 fps target must duplicate frames in
    monotonic time order rather than crash or skip time."""
    source_times, source_frames = _decode_source(video_input_dir / "slow.mp4")
    out = load_video("slow.mp4", str(video_input_dir), trim_start=0.0, trim_end=3.0)

    assert abs(out.shape[0] - 72) <= 1
    matched = [_match_source_index(source_frames, out[k]) for k in range(out.shape[0])]
    assert matched == sorted(matched), "nearest-PTS sampling must be monotonic in time"
    # Both edges of the 3 s window are covered.
    assert matched[0] <= 1
    assert matched[-1] >= len(source_frames) - 2
