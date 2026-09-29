"""Small, chronological tail thumbnails for visual prompt assistance."""
from pathlib import Path


def make_tail_thumbnails(video, directory):
    from .pyav_media import tail_frames
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    try:
        for index, image in enumerate(tail_frames(video), 1):
            name = f"tail-{index:02d}.jpg"
            names.append(name)
            image.save(directory / name, format="JPEG", quality=90)
    except (OSError, ValueError) as exc:
        for name in names:
            (directory / name).unlink(missing_ok=True)
        raise RuntimeError(f"Could not extract tail previews: {exc}") from exc
    if not names:
        raise RuntimeError("The exported file provided no decodable tail frames.")
    return names


def ensure_tail_thumbnails(metadata, directory):
    """Lazy visual evidence; no extraction during upload, capture or text-only Forge.

    Keep this cache separate from the immutable checkpoint/manifest metadata.
    Source identity is rechecked even when JPEGs already exist.
    """
    from .core import atomic_json
    from .video_source import input_video, file_digest
    import folder_paths
    import json
    directory = Path(directory).resolve()
    if metadata.get("kind") == "video":
        path = input_video(metadata["filename"])
        if file_digest(path) != metadata["sha256"]:
            raise ValueError("Source video changed; select it again before analysis.")
    else:
        path = Path(metadata.get("output_path") or "").resolve()
        roots = [Path(folder_paths.get_output_directory()).resolve(), Path(folder_paths.get_temp_directory()).resolve(), Path(folder_paths.get_input_directory()).resolve()]
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
            raise ValueError("Source video is unavailable for visual analysis.")
        provenance = metadata.get("provenance") or {}
        if provenance.get("kind") == "imported_video" and file_digest(path) != provenance.get("sha256"):
            raise ValueError("Source video changed; select it again before analysis.")
    signature = ["pyav-v2", str(path), path.stat().st_size, path.stat().st_mtime_ns]
    cache = directory / "tail-cache.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text())
            names = data.get("thumbnails", []) if isinstance(data, dict) else []
            if isinstance(data, dict) and data.get("signature") == signature and isinstance(names, list) and names and all(isinstance(n, str) and (directory / n).resolve().parent == directory and n.endswith(".jpg") and (directory / n).is_file() for n in names):
                return names
        except (OSError, ValueError, TypeError):
            pass  # Optional preview cache: regenerate after a partial or invalid write.
    directory.mkdir(parents=True, exist_ok=True)
    for image in directory.glob("tail-*.jpg"):
        image.unlink()
    names = make_tail_thumbnails(path, directory)
    atomic_json(cache, {"signature": signature, "thumbnails": names})
    return names
