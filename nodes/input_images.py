"""Small ComfyUI input-folder listing endpoint for Director UIs.

Core ComfyUI does not expose an image-listing route, but DaSiWa UIs (e.g. the
MiniMax H3 Director watermark picker) need to feed a <select> from the input
folder. This registers a tiny read-only route that returns image paths relative
to the input directory (subfolders preserved), matching the path shape that
``folder_paths.get_annotated_filepath`` and the ComfyUI ``/view`` endpoint
understand.

Mirrors the route-registration pattern already used by ``nodes_system_monitor``.
"""
import os

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".avif",
}

try:
    import folder_paths
    from aiohttp import web
    from server import PromptServer
except ImportError:
    folder_paths = None
    web = None
    PromptServer = None


def list_input_images():
    """Return a sorted list of image paths (relative to the input folder).

    Subfolders are preserved with a ``subfolder/name`` form so the result maps
    directly onto ``folder_paths.get_annotated_filepath``. Returns ``[]`` when
    the input folder is unavailable (never raises into the UI).
    """
    if folder_paths is None:
        return []
    try:
        base = folder_paths.get_input_directory()
    except Exception:
        return []
    if not base or not os.path.isdir(base):
        return []
    found = []
    for root, _dirs, files in os.walk(base):
        for name in files:
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            rel = os.path.relpath(os.path.join(root, name), base)
            found.append(rel.replace(os.sep, "/"))
    return sorted(found)


def register_routes():
    """Register the listing route on the running ComfyUI server, if present."""
    if PromptServer is None or web is None:
        return

    @PromptServer.instance.routes.get("/dasiwa/input-images")
    async def dasiwa_input_images(request):
        return web.json_response({"images": list_input_images()})


register_routes()
