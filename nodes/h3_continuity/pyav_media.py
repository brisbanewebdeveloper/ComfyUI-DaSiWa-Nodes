"""Local, streaming H3 media conversion through PyAV's linked libraries.

Use the same libavfilter operations as the former CLI pipeline. Only decoded
frames and filter look-ahead are resident; import buffers are written to disk.
"""
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import math
import shutil
import time

import av
import numpy as np

MAX_PIXELS = 33_554_432  # Includes UHD 8K; reject oversized headers before decode.
MAX_SECONDS = 3600
_FORMATS = "mov,matroska,webm,avi,mpegts,mpeg,m4v,ogg,flv,asf,h264,hevc,ivf,nut,mjpeg,wav,mp3,flac,aac"


def guard(timeout=1800, interrupt=False):
    deadline = time.monotonic() + timeout
    def check():
        if interrupt:
            from comfy.model_management import throw_exception_if_processing_interrupted
            throw_exception_if_processing_interrupted()
        if time.monotonic() > deadline:
            raise ValueError("Media processing timed out. Use a shorter or repaired source video.")
    check()
    return check


def check_dimensions(width, height):
    if width <= 0 or height <= 0 or width > 32768 or height > 32768 or width * height > MAX_PIXELS:
        raise ValueError(f"Source/canvas dimensions exceed the {MAX_PIXELS:,}-pixel media limit.")


@contextmanager
def open_media(path):
    # Python owns the local file. Nested file/network URLs in playlist-style
    # containers are forbidden; a validated basename must not grant URL access.
    try:
        with Path(path).open("rb") as file:
            with av.open(file, mode="r", options={"protocol_whitelist": "", "format_whitelist": _FORMATS, "enable_drefs": "0"}) as container:
                yield container
    except av.FFmpegError as exc:
        raise ValueError(f"Could not decode source media with PyAV: {exc}") from exc


def video_stream(container, index=None):
    streams = [s for s in container.streams.video if not (s.disposition & av.stream.Disposition.attached_pic)]
    stream = next((s for s in streams if index is None or s.index == index), None)
    if stream is None:
        raise ValueError("The selected file has no decodable video stream.")
    check_dimensions(stream.width, stream.height)
    stream.codec_context.thread_count = 2
    return stream


def decoded(container, stream, check):
    for packet in container.demux(stream):
        check()
        for frame in packet.decode():
            check()
            if isinstance(frame, av.VideoFrame):
                check_dimensions(frame.width, frame.height)
            yield frame


def rotation_filters(frame):
    angle = float(frame.rotation)
    matrix = next((s for s in frame.side_data if s.type.name == "DISPLAYMATRIX"), None)
    if matrix is not None:
        values = np.frombuffer(matrix, dtype=np.int32)
        if len(values) != 9:
            raise ValueError("Invalid source display matrix.")
        linear = values[[0, 1, 3, 4]].astype(float) / 65536
        key = tuple(np.rint(linear).astype(int))
        transforms = {
            (1, 0, 0, 1): [], (-1, 0, 0, 1): [("hflip", None)],
            (1, 0, 0, -1): [("vflip", None)], (-1, 0, 0, -1): [("hflip", None), ("vflip", None)],
            (0, -1, 1, 0): [("transpose", "cclock")], (0, 1, -1, 0): [("transpose", "clock")],
            (0, -1, -1, 0): [("transpose", "cclock"), ("hflip", None)],
            (0, 1, 1, 0): [("transpose", "clock"), ("hflip", None)],
        }
        if key in transforms and np.allclose(linear, key, atol=1e-4):
            return transforms[key]
    if not math.isfinite(angle):
        raise ValueError("Invalid source display rotation.")
    nearest = round(angle / 90) * 90
    if abs(angle - nearest) > .01:
        raise ValueError("Continuity import requires a right-angle display rotation; normalize this source first.")
    return {0: [], 90: [("transpose", "cclock")], 180: [("hflip", None), ("vflip", None)], 270: [("transpose", "clock")]}[nearest % 360]


def graph_for(frame, filters):
    graph = av.filter.Graph()
    graph.threads = 1
    source = graph.add_buffer(template=frame) if isinstance(frame, av.VideoFrame) else graph.add_abuffer(sample_rate=frame.sample_rate, format=frame.format.name, layout=frame.layout.name, time_base=frame.time_base)
    sink = "buffersink" if isinstance(frame, av.VideoFrame) else "abuffersink"
    graph.link_nodes(source, *(graph.add(name, args) for name, args in filters), graph.add(sink))
    graph.configure()
    return graph


def drain(graph, check):
    while True:
        check()
        try:
            yield graph.pull()
        except (av.error.BlockingIOError, av.error.EOFError):
            return


def normalize_timestamp(frame, origin, index, rate=None):
    if frame.pts is None:
        if not rate:
            raise ValueError("Source has no usable frame timestamps or frame rate.")
        frame.time_base = 1 / Fraction(rate)
        frame.pts = index
        frame.duration = 1
    elif origin is not None:
        frame.pts -= round(origin / frame.time_base)


def probe(path, check=None):
    check = check or guard(45)
    with open_media(path) as container:
        stream = video_stream(container)
        rate = stream.average_rate or stream.guessed_rate
        seconds = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration or 0) / av.time_base
        if not math.isfinite(seconds) or seconds > MAX_SECONDS:
            raise ValueError(f"Source duration exceeds the {MAX_SECONDS}-second media limit.")
        frames = decoded(container, stream, check)
        first = next(frames, None)
        if first is None:
            raise ValueError("The selected file has no decodable video stream.")
        rotation_filters(first)
        if seconds <= 0:
            rate = stream.guessed_rate or stream.average_rate
            # Duration-less elementary streams: bounded scan, one frame at a time.
            start, end, count = None, None, 0
            for frame in _prepend(first, frames):
                check()
                if count >= MAX_SECONDS * 240:
                    raise ValueError("Could not determine a bounded source duration.")
                stamp = float(frame.pts * frame.time_base) if frame.pts is not None else count / float(rate) if rate else None
                if stamp is None:
                    raise ValueError("Could not determine the source video's duration.")
                period = float(frame.duration * frame.time_base) if frame.duration else 1 / float(rate) if rate else 0
                start = stamp if start is None else min(start, stamp)
                end = max(end or stamp, stamp + period)
                count += 1
                if end - start > MAX_SECONDS:
                    raise ValueError(f"Source duration exceeds the {MAX_SECONDS}-second media limit.")
            seconds = end - start
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Could not determine the source video's duration.")
        return {"width": stream.width, "height": stream.height, "fps": float(rate or 0), "seconds": seconds,
                "video_stream": stream.index, "has_audio": bool(container.streams.audio), "rotation": float(first.rotation)}


def _prepend(first, rest):
    yield first
    yield from rest


def write_spool(output, data):
    # Recheck while writing: a bad duration header or other jobs may exhaust
    # space after the preflight estimate. Preserve the same 64 MiB reserve.
    if shutil.disk_usage(Path(output.name).parent).free < len(data) + 64 * 1024**2:
        raise ValueError("Insufficient temporary disk space while decoding source media.")
    output.write(data)


def write_video(path, info, width, height, destination, check):
    check_dimensions(width, height)
    count = 0
    with open_media(path) as container, Path(destination).open("wb") as output:
        stream = video_stream(container, info["video_stream"])
        graph = None
        first_frame = None
        source_frames = 0
        def write_ready():
            nonlocal count
            for frame in drain(graph, check):
                # The former rawvideo muxer filled a positive initial PTS with
                # copies of the first *filtered* frame. fps:start_time=0 would
                # choose an earlier decoded frame and change the source prefix.
                if frame.pts < 0:
                    continue
                copies = 1 + (frame.pts if count == 0 else 0)
                if count + copies > MAX_SECONDS * 24:
                    raise ValueError(f"Decoded video exceeds the {MAX_SECONDS}-second media limit.")
                data = frame.to_ndarray(format="rgb24").tobytes()
                for _ in range(copies):
                    check()
                    write_spool(output, data)
                    count += 1
        for i, frame in enumerate(decoded(container, stream, check)):
            normalize_timestamp(frame, Fraction(container.start_time or 0, av.time_base), i, stream.guessed_rate or stream.average_rate)
            source_frames += 1
            first_frame = frame if source_frames == 1 else None
            if graph is None:
                filters = rotation_filters(frame) + [("fps", "24"), ("scale", f"{width}:{height}:force_original_aspect_ratio=decrease"),
                    ("pad", f"{width}:{height}:(ow-iw)/2:(oh-ih)/2"), ("setsar", "1"), ("format", "rgb24")]
                graph = graph_for(frame, filters)
            graph.push(frame)
            write_ready()
        if graph is None:
            raise ValueError("The source video did not decode to complete RGB frames.")
        graph.push(None)
        write_ready()
        if not count and source_frames == 1:
            # A single still frame with no video packet duration can be dropped
            # entirely by fps. Keep it upright and cover the container timeline
            # (which may be longer because of an accompanying audio stream).
            still = graph_for(first_frame, rotation_filters(first_frame) + [
                ("scale", f"{width}:{height}:force_original_aspect_ratio=decrease"),
                ("pad", f"{width}:{height}:(ow-iw)/2:(oh-ih)/2"),
                ("setsar", "1"), ("format", "rgb24")])
            still.push(first_frame)
            still.push(None)
            image = next(drain(still, check), None)
            if image is not None:
                copies = max(1, round(info["seconds"] * 24))
                if copies > MAX_SECONDS * 24:
                    raise ValueError(f"Decoded video exceeds the {MAX_SECONDS}-second media limit.")
                data = image.to_ndarray(format="rgb24").tobytes()
                for _ in range(copies):
                    check()
                    write_spool(output, data)
                count = copies
    if not count:
        raise ValueError("The source video did not decode to complete RGB frames.")
    return count


def write_audio(path, leading_samples, samples, destination, check):
    written = 0
    with open_media(path) as container, Path(destination).open("wb") as output:
        if not container.streams.audio:
            raise ValueError("Source audio stream disappeared; select the video again.")
        stream = container.streams.audio[0]
        stream.codec_context.thread_count = 2
        graph = None
        def write_ready():
            nonlocal written
            for frame in drain(graph, check):
                array = frame.to_ndarray().reshape(-1)
                count = min(frame.samples, samples - written)
                if count:
                    write_spool(output, np.asarray(array[:count * 2], dtype="<f4").tobytes())
                    written += count
                if written == samples:
                    return
        for i, frame in enumerate(decoded(container, stream, check)):
            if frame.pts is None:
                raise ValueError("Source audio has no presentation timestamps; normalize this source first.")
            normalize_timestamp(frame, Fraction(container.start_time or 0, av.time_base), i)
            if graph is None:
                graph = graph_for(frame, [("aresample", "32000:async=1:first_pts=0"), ("adelay", f"{leading_samples}S:all=1"),
                    ("apad", None), ("atrim", f"end_sample={samples}"), ("aformat", "sample_fmts=flt:sample_rates=32000:channel_layouts=stereo")])
            graph.push(frame)
            write_ready()
            if written == samples:
                break
        if graph is not None and written < samples:
            graph.push(None)
            write_ready()
        if written != samples:
            raise ValueError("Could not align source audio with its video.")


def tail_frames(path):
    """At most four upright 640px frames; short clips still provide one image."""
    check = guard(45)
    info = probe(path, check)
    start = max(0, info["seconds"] - 2)
    with open_media(path) as container:
        stream = video_stream(container, info["video_stream"])
        origin = stream.start_time * stream.time_base if stream.start_time is not None else Fraction(container.start_time or 0, av.time_base)
        if start:
            container.seek(int((Fraction(start) + origin) / stream.time_base), stream=stream, backward=True)
        graph, count, fallback = None, 0, None
        for i, frame in enumerate(decoded(container, stream, check)):
            normalize_timestamp(frame, origin, i, stream.guessed_rate or stream.average_rate)
            if graph is None:
                graph = graph_for(frame, rotation_filters(frame) + [("trim", f"start={start}"), ("setpts", f"PTS-{start}/TB"),
                    ("fps", "2"), ("scale", "640:640:force_original_aspect_ratio=decrease"), ("format", "rgb24")])
            fallback = frame
            graph.push(frame)
            for image in drain(graph, check):
                yield image.to_image()
                count += 1
                if count == 4:
                    return
        if graph is not None:
            graph.push(None)
            for image in drain(graph, check):
                yield image.to_image()
                count += 1
                if count == 4:
                    return
        if not count and fallback is not None:
            still = graph_for(fallback, rotation_filters(fallback) + [("scale", "640:640:force_original_aspect_ratio=decrease"), ("format", "rgb24")])
            still.push(fallback)
            yield still.pull().to_image()
