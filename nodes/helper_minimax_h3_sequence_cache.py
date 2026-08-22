"""Safe local disk cache for DaSiWa MiniMax H3 sequence segments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import folder_paths
import torch
from comfy.nested_tensor import NestedTensor

from .helper_minimax_h3_director import normalize_guide
from .helper_minimax_h3_sequence import audio_to_cpu, latent_to_cpu

log = logging.getLogger("ComfyUI-DaSiWa-Nodes.minimax_h3_sequence_cache")
CACHE_VERSION = 1


def segment_fingerprint(
    segment: dict,
    index: int,
    *,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    continuity: bool,
    context_frames: int,
) -> str:
    """Return stable identity; cache_key must change when attached media changes."""
    state = normalize_guide(segment["guide"])
    data = {
        "version": CACHE_VERSION,
        "index": int(index),
        "cache_key": segment.get("cache_key", ""),
        "mode": state.mode,
        "prompt": state.resolved_prompt,
        "width": state.width,
        "height": state.height,
        "frames": state.length,
        "seed": int(seed),
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler": str(sampler),
        "scheduler": str(scheduler),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "continuity": bool(continuity),
        "context_frames": int(context_frames) if continuity else 0,
    }
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_path(node_id: str, cache_key: str, index: int) -> Path | None:
    if not cache_key:
        return None
    node_hash = hashlib.sha256(str(node_id).encode("utf-8")).hexdigest()[:20]
    key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    try:
        root = Path(folder_paths.get_output_directory()) / "dasiwa_h3_sequence_cache" / node_hash
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("MiniMax H3 sequence cache is unavailable: %s", exc)
        return None
    return root / f"seg_{int(index):04d}_{key_hash}.pt"


def _latent_streams(latent: dict | None):
    latent = latent_to_cpu(latent)
    if latent is None:
        return None
    return tuple(latent["samples"].unbind())


def save_segment(
    node_id: str,
    cache_key: str,
    index: int,
    fingerprint: str,
    images: torch.Tensor,
    audio: dict,
    latent: dict | None,
) -> bool:
    """Atomically cache one CPU segment; failures do not abort generation."""
    path = _cache_path(node_id, cache_key, index)
    if path is None:
        return False
    payload = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "images": images.detach().cpu().contiguous(),
        "audio": audio_to_cpu(audio),
        "latent_streams": _latent_streams(latent),
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temp_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("MiniMax H3 segment %d cache write skipped: %s", index + 1, exc)
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_segment(
    node_id: str,
    cache_key: str,
    index: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Load a matching cache created by this node; reject stale or malformed data."""
    path = _cache_path(node_id, cache_key, index)
    if path is None or not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        log.warning("MiniMax H3 segment %d cache read skipped: %s", index + 1, exc)
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    images = payload.get("images")
    audio = payload.get("audio")
    if not torch.is_tensor(images) or not isinstance(audio, dict):
        return None
    streams = payload.get("latent_streams")
    latent = None
    if isinstance(streams, (tuple, list)) and len(streams) == 2 and all(torch.is_tensor(item) for item in streams):
        latent = {"samples": NestedTensor(tuple(streams))}
    return {"images": images, "audio": audio_to_cpu(audio), "latent": latent}
