# H3 continuity: PyAV migration

This patch replaces H3 continuity's external media commands with PyAV's linked
libavformat, libavcodec and libavfilter libraries. PyAV still uses FFmpeg
libraries; the removed dependency is on separate executables on PATH. The
existing `av>=18.0` requirement is sufficient.

## Scope and preserved behavior

- `video_source.py`: probe and streaming RGB/PCM conversion; import cache v2.
- `media.py`: bounded, upright, chronological Forge JPEGs; cache signature v2.
- `pyav_media.py`: shared local-file decoder, filter graphs and resource guards.
- `routes.py`: remove the obsolete subprocess exception path.
- `nodes.py`: local-only export inspection and finite positive duration checks.
- `inspection.py`: update the media error terminology.

The Director UI, Duration behavior, workflow connections, Prompt Forge policy,
native continuation implementation and Enhanced Video Combine encoder are
unchanged. Completed checkpoints are immutable and remain readable. Selecting
an old uploaded source causes one fresh import instead of reusing a cache from
the former normalizer.

Normalization retains 24 fps, aspect-preserving scale/pad, square pixels,
17k+5 leading video padding, matching leading silence, stereo 32 kHz audio,
and the globally rounded 40 Hz audio-token boundary. The video decoder writes
frames incrementally to temporary disk. Public video VAE calls remain bounded
to 124 frames; audio encoding and cumulative export still scale with length.

Two compatibility details matter: a positive initial video timestamp must fill
the gap with the first **filtered** frame, and tail seeking must use the video
stream start rather than an AAC-primed container start. Duration-less elementary
streams use a bounded scan and the guessed frame rate where available. Very
short clips that cannot supply a 2 fps tail now yield one upright still.

## Bounds and failure behavior

Local validated input/output paths are opened as file objects. External data
references, nested protocols and playlist demuxers are disabled. The allowlist
is `mov,matroska,webm,avi,mpegts,mpeg,m4v,ogg,flv,asf,h264,hevc,ivf,nut,mjpeg,wav,mp3,flac,aac`;
an allowed container still needs a decodable video stream. Other formats must
be converted before import. Standard eight right-angle/reflection orientations
are supported; arbitrary-angle rotation is rejected with an actionable error.
Audio without presentation timestamps is also rejected rather than guessed.

The limit is 3,600 seconds, 33,554,432 pixels per frame and 32,768 per edge.
Probe/tail extraction has a 45-second cooperative deadline; conversion shares
a 30-minute deadline. Queue cancellation is checked between packets, decoded
frames, filter pulls and writes. These checks cannot forcibly interrupt a
single native library call. Actual spool writes preserve 64 MiB free space
after the existing preflight estimate. Temporary files and partial JPEG writes
are cleaned up on errors. These guards are not a hard RAM/VRAM quota or a
guarantee that hour-long sources are practical.

## Validation recorded for this handoff

Base: `40322936acd07dfc926bdcc0afa0ba1c994163c8` (package 0.4.61).
ComfyUI: `79be670e2d9be63e238785af307369d2b9039ed1` (0.37.0).
Python 3.12.14; CPU Torch 2.14.0; PyAV 18.1.0 and lower-bound 18.0.0.

- 116 CPU/media/workflow regression tests pass with PyAV 18.1.0.
- All 30 migration tests also pass with PyAV 18.0.0.
- Fifteen media fixtures compare the original CLI path with the replacement:
  CFR, VFR, silent, delayed stereo, audio gaps, nonzero start, three-frame clips,
  and all eight right-angle/reflection orientations. Recorded normalized video
  and audio errors are zero with these builds; frame counts and padding match.
- Five tail-image comparisons cover chronology, rotation and offsets. JPEG
  bytes differ because Pillow encodes them; decoded images are compared with
  a tolerance. No tail tiles were added to the UI.
- Sixteen actual ComfyUI queued jobs (eight sources, two backends) run through
  import, Append & Stage, the unmodified Enhanced Video Combine, and Publish.
  The resulting files are decoded and their video/audio hashes compared.
  All eight pairs match; the PyAV server runs with an empty PATH.

The queue harness uses deterministic CPU stand-in VAEs and native H3 temporal
scheduling. It establishes the media, queue, checkpoint and exporter contract,
not neural reconstruction quality. No H3 model weights, GPU sampling, browser
interaction or LLM generation were exercised in this migration run. Before a
release, run a real H3 video+audio VAE import and continuation on a GPU, inspect
the seam and duration, and exercise Forge vision on that source. Other codecs,
platforms and libav builds may have different numerical results.

Development CLI baselines and fixture generators belong in the separate
validation bundle, not the Registry archive. Do not install the development
queue node into a user's normal ComfyUI instance. Publication and any Registry
manual review remain maintainer actions; this patch does not claim Registry
approval.
