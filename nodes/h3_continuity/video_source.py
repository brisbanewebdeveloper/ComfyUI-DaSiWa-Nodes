"""Import ordinary videos through the connected H3 VAEs during queue execution.

HTTP preparation only probes and hashes media. No model is loaded there.
The source is content-addressed; replacing an uploaded file cannot change a queued
take. Native VAE calls preserve ComfyUI's model loading/offloading behaviour.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from comfy.nested_tensor import NestedTensor

from .core import ClipStore, atomic_json, mode_family, safe_id
from .vendor.continuation_nodes import validate_h3_av_latent

IMPORT_VERSION = 2  # New media normalization backend; keep old checkpoints immutable.


def input_video(filename, input_root=None):
    if input_root is None:
        import folder_paths
        input_root = folder_paths.get_input_directory()
    root = Path(input_root).resolve()
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise ValueError("Select a video uploaded to ComfyUI input.")
    relative = Path(filename.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Video must be inside ComfyUI input.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source video is missing. Select it again in the Director.")
    return path


def file_digest(path, interrupt=False):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if interrupt:
                from comfy.model_management import throw_exception_if_processing_interrupted
                throw_exception_if_processing_interrupted()
            digest.update(block)
    return digest.hexdigest()


def probe_video(path):
    from .pyav_media import probe
    return probe(path)


def manifest_dir(store, source_id):
    return store.clip_dir("_imports", safe_id(source_id))


def read_manifest(store, source_id):
    data = json.loads(store.member("_imports", source_id, "source.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("clip_id") != source_id:
        raise ValueError("Imported video identity does not match its manifest.")
    return data


def prepare_video(filename, store=None, input_root=None):
    store = store or ClipStore()
    path = input_video(filename, input_root)
    digest = file_digest(path)
    # Include the upload name so two identical uploads never share a mutable path.
    source_id = hashlib.sha256((filename + "\0" + digest).encode()).hexdigest()
    directory = manifest_dir(store, source_id)
    if (directory / "source.json").exists():
        return read_manifest(store, source_id)
    info = probe_video(path)
    directory.mkdir(parents=True, exist_ok=True)
    # Tail images are only needed on an explicit vision Forge request.
    if file_digest(path) != digest:
        raise ValueError("The source changed during preparation. Select it again.")
    data = {**info, "clip_id": source_id, "filename": filename, "sha256": digest,
            "kind": "video", "prompt": "", "thumbnails": [], "preview_warning": ""}
    atomic_json(directory / "source.json", data)
    return data


def encode_video_chunks(frames, leading, total, vae):
    """124-frame public VAE calls, advancing by 119 frames (7 native chunks).

    Native H3 encodes independent 17-frame chunks, then drops three tokens only
    at the global end. Five look-ahead frames make each call valid (17k+5);
    remove their two tokens from every nonfinal call. This avoids loading the
    entire decoded movie as a floating point tensor.
    """
    pieces = []
    for start in range(0, total, 119):
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        stop = min(start + 124, total)
        indices = np.clip(np.arange(start, stop) - leading, 0, len(frames) - 1)
        pixels = torch.from_numpy(np.array(frames[indices], copy=True)).float().div_(255)
        latent = vae.encode(pixels)
        final = stop == total
        expected = ((stop - start - 5) // 17) * 5 + 2
        if latent.ndim != 5 or latent.shape[:2] != (1, 24) or latent.shape[2] != expected:
            raise ValueError("Connect the native MiniMax H3 video VAE: unexpected encoded shape.")
        pieces.append((latent if final else latent[:, :, :-2]).detach().cpu().clone())
        if final:
            break
    return torch.cat(pieces, dim=2)


def encode_source(path, info, width, height, vae, audio_vae):
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 source canvas must have both edges divisible by 32.")
    if audio_vae is None:
        raise ValueError("Connect the MiniMax H3 audio VAE to import video, including silent video.")
    if int(getattr(audio_vae, "audio_sample_rate", 32000)) != 32000:
        raise ValueError("Source import requires the MiniMax H3 audio VAE (32000 Hz).")
    from .pyav_media import guard, write_video, write_audio
    check = guard(interrupt=True)
    with tempfile.TemporaryDirectory(prefix="dasiwa-h3-import-") as work:
        work = Path(work)
        from .inspection import check_import_space
        check_import_space(info["seconds"], width, height, work)
        raw_video = work / "video.rgb"
        write_video(path, info, width, height, raw_video, check)
        size = raw_video.stat().st_size
        frame_bytes = width * height * 3
        if not size or size % frame_bytes:
            raise ValueError("The source video did not decode to complete RGB frames.")
        actual = size // frame_bytes
        total = max(5, actual)
        total += (5 - total) % 17
        leading = total - actual
        # Preserve the real ending. Grid padding is at the beginning, never the tail.
        frames = np.memmap(raw_video, dtype=np.uint8, mode="r", shape=(actual, height, width, 3))
        try:
            video = encode_video_chunks(frames, leading, total, vae)
        finally:
            frames._mmap.close()
        audio_tokens = round(total / 24 * 40)
        audio_samples = audio_tokens * 800
        audio_path = work / "audio.f32"
        if info["has_audio"]:
            leading_samples = round(leading / 24 * 32000)
            write_audio(path, leading_samples, audio_samples, audio_path, check)
            waveform = np.fromfile(audio_path, dtype="<f4")
            if waveform.size != audio_samples * 2:
                raise ValueError("Could not align source audio with its video.")
            waveform = torch.from_numpy(waveform.reshape(1, audio_samples, 2))
        else:
            # A zero latent is not encoded silence. Use the actual audio encoder.
            waveform = torch.zeros((1, audio_samples, 2), dtype=torch.float32)
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        audio = audio_vae.encode(waveform).detach().cpu()
        latent = {"samples": NestedTensor((video, audio))}
        _, _, verified_frames = validate_h3_av_latent(latent, name="imported video")
        if verified_frames != total:
            raise ValueError("The connected H3 VAEs returned an incompatible AV duration.")
        return latent, {"decoded_frames": actual, "leading_pad_frames": leading,
                        "fit": "contain", "audio": "encoded" if info["has_audio"] else "encoded silence"}


def import_cache_id(manifest, guide):
    key = [IMPORT_VERSION, manifest["clip_id"], guide["width"], guide["height"], mode_family(guide["mode"])]
    return "import_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:48]


def import_checkpoint(settings, guide, vae, audio_vae, store=None, input_root=None):
    store = store or ClipStore()
    manifest = read_manifest(store, settings["source_video_id"])
    path = input_video(manifest["filename"], input_root)
    if file_digest(path, interrupt=True) != manifest["sha256"]:
        raise ValueError("The selected source video changed. Select it again before continuing.")
    clip_id = import_cache_id(manifest, guide)
    directory = store.clip_dir(settings["session"], clip_id)
    if (directory / "clip.json").exists():
        # An immutable source checkpoint is reused on rerolls of the same upload/canvas.
        return (*store.load(settings["session"], clip_id), clip_id)
    latent, normalization = encode_source(path, manifest, guide["width"], guide["height"], vae, audio_vae)
    if file_digest(path, interrupt=True) != manifest["sha256"]:
        raise ValueError("Source video changed during encoding. Select it again.")
    context = {"session": settings["session"], "run_id": clip_id, "mode": guide["mode"],
               "resolved_prompt": "", "provenance": {"kind": "imported_video",
               "source_video_id": manifest["clip_id"], "filename": manifest["filename"],
               "sha256": manifest["sha256"], "normalization": normalization}}
    # Stage under an execution-owned temporary ID, publish, then atomically expose
    # the deterministic directory. A failed encode never poisons the cache key.
    import uuid
    staging_id = "importing_" + uuid.uuid4().hex
    context["run_id"] = staging_id
    ticket = store.stage(latent, context)
    staged = store.clip_dir(settings["session"], staging_id)
    try:
        for name in manifest["thumbnails"]:
            shutil.copyfile(manifest_dir(store, manifest["clip_id"]) / name, staged / name)
        data = store.metadata(settings["session"], staging_id, ready=False)
        data.update(clip_id=clip_id, status="ready", output_path=str(path),
                    completed_ns=data["created_ns"], thumbnails=manifest["thumbnails"])
        atomic_json(staged / "clip.json", data)
        staged.rename(directory)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return latent, data, clip_id
