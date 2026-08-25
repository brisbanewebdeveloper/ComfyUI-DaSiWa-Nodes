"""Director 2.0 built-in live step preview.

Native WrappersMP.OUTER_SAMPLE wrapper attached to the execution model before
sampling. Each denoising step the x0 latent is decoded to preview frames
(preview_tiny_vae > preview_vae > core previewer > Latent2RGB) and shipped to
the Director's own Preview & Output panel via
PromptServer.send_sync(PREVIEW_EVENT, payload, client_id).

No KJNodes dependency: _AsyncPreviewEncoder, _encode_mp4_nvenc,
_encode_animated_webp, _decode_video_frames_l2rgb, _tiny_vae_decode_to_pil are
ports of ComfyUI-KJNodes/nodes/preview_override_node.py (GPL-3.0; the
verbatim-copied nodes/tiny_vae.py carries its own attribution header).
"""
import base64
import io as pyio
import logging
import queue
import threading

import numpy as np
import torch
from PIL import Image, ImageOps

# ComfyUI core modules are only importable inside a running ComfyUI install.
# Import them eagerly when available and fall back to None so this module stays
# import-safe in the repo's dev test env; every use site below checks the
# None-sentinel before touching the module (and tests monkeypatch the names).
try:
    import comfy.patcher_extension
except ImportError:
    comfy = None

try:
    import latent_preview
except ImportError:
    latent_preview = None

from .tiny_vae import load_tiny_vae_decoder

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

PREVIEW_EVENT = "dasiwa_director_v2_preview"


def prompt_server():
    """The running PromptServer instance, or None outside a ComfyUI server."""
    return getattr(PromptServer, "instance", None) if PromptServer is not None else None


# ---- KJ ports (GPL-3.0, ComfyUI-KJNodes) ----------------------------------
# _AsyncPreviewEncoder:       preview_override_node.py:37-69 (bounded FIFO, drop-on-full)
# _probe_nvenc/_NVENC_AVAILABLE/_NVENC_MIN_W/_NVENC_MIN_H/_nvenc_warned: :112-127
# _encode_mp4_nvenc:         preview_override_node.py:130-184 (frag MP4, WebP fallback)
# _encode_animated_webp:     preview_override_node.py:187-214
# _decode_video_frames_l2rgb: preview_override_node.py:82-109 (bulk GPU->CPU copy)
# Each block is ported verbatim with only the log tag changed to the Director's.


class _AsyncPreviewEncoder:
    """Off-thread encoder. Bounded FIFO drops-on-full so the sampler never blocks on us."""
    # Source: ComfyUI-KJNodes (GPL-3.0)

    _STOP = object()

    def __init__(self, max_in_flight=2):
        self.q = queue.Queue(maxsize=max_in_flight)
        self.thread = threading.Thread(target=self._run, name="dasiwa_director_preview_encoder", daemon=True)
        self.thread.start()

    def submit(self, fn):
        try:
            self.q.put_nowait(fn)
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._STOP:
                return
            try:
                item()
            except Exception:
                logging.exception("[dasiwa director preview] async encoder error")

    def shutdown(self, drain_timeout=5.0):
        try:
            self.q.put(self._STOP, timeout=drain_timeout)
        except queue.Full:
            pass
        self.thread.join(timeout=drain_timeout)


def _get_core_previewer(load_device, latent_format):
    # Walk past custom-node hooks on get_previewer to reach the unwrapped core function.
    # Source: ComfyUI-KJNodes (GPL-3.0)
    if latent_preview is None:
        return None
    fn = latent_preview.get_previewer
    seen = set()
    while hasattr(fn, "__wrapped__") and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__
    return fn(load_device, latent_format)


def _decode_video_frames_l2rgb(x0, latent_format, max_frames, stride=1):
    # Bulk-blocking GPU->CPU copy (not per-frame non_blocking) avoids torn frames at high res.
    # Source: ComfyUI-KJNodes (GPL-3.0)
    if x0.ndim != 5:
        return []
    rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
    if rgb_factors is None:
        return []
    try:
        reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
        if reshape is not None:
            x0 = reshape(x0)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        factors = torch.tensor(rgb_factors, device=x0.device, dtype=x0.dtype).transpose(0, 1)
        bias_t = torch.tensor(bias, device=x0.device, dtype=x0.dtype) if bias is not None else None
        x = x0[0]
        if stride > 1:
            x = x[:, ::stride]
        t_total = x.shape[1]
        if max_frames > 0 and max_frames < t_total:
            indices = np.linspace(0, t_total - 1, max_frames).round().astype(int).tolist()
            x = x[:, indices]
        x = x.movedim(0, -1)
        rgb = torch.nn.functional.linear(x, factors, bias=bias_t)
        rgb.add_(1.0).mul_(127.5).clamp_(0, 255)
        rgb_cpu = rgb.to(torch.uint8).cpu().numpy()
        return [Image.fromarray(rgb_cpu[i]) for i in range(rgb_cpu.shape[0])]
    except Exception:
        return []


# PyPI PyAV wheels typically lack NVENC; probe once at import.
def _probe_nvenc():
    # Source: ComfyUI-KJNodes (GPL-3.0)
    try:
        import av  # noqa
        av.Codec("h264_nvenc", "w")
        return True
    except Exception:
        return False


_NVENC_AVAILABLE = _probe_nvenc()

# NVENC H.264 rejects sub-145x49 inputs at avcodec_open2 — fall back to WebP for small frames.
_NVENC_MIN_W = 145
_NVENC_MIN_H = 49

_nvenc_warned = False


def _encode_mp4_nvenc(frames, fps, max_res):
    # Fragmented MP4 so the browser can decode mid-download. Returns (None, 0, 0) on failure
    # (including too-small-for-NVENC), so caller falls through to WebP.
    # Source: ComfyUI-KJNodes (GPL-3.0)
    global _nvenc_warned
    if not frames:
        return None, 0, 0
    try:
        import av
    except Exception:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    # yuv420p requires even dimensions.
    w0, h0 = pil_frames[0].width, pil_frames[0].height
    out_w, out_h = w0 & ~1, h0 & ~1
    if (out_w, out_h) != (w0, h0):
        pil_frames = [pf.resize((out_w, out_h), Image.LANCZOS) for pf in pil_frames]
    if out_w < _NVENC_MIN_W or out_h < _NVENC_MIN_H:
        return None, 0, 0
    # Driver/GPU varies what option combos are accepted; bare preset always works.
    option_candidates = [
        {"preset": "p1", "rc": "vbr", "cq": "23"},
        {"preset": "p1"},
    ]
    last_err = None
    for opts in option_candidates:
        buf = pyio.BytesIO()
        try:
            container = av.open(
                buf, mode="w", format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            stream = container.add_stream("h264_nvenc", rate=int(max(1, fps)))
            stream.width = out_w
            stream.height = out_h
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            for pf in pil_frames:
                for pkt in stream.encode(av.VideoFrame.from_image(pf)):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            return base64.b64encode(buf.getvalue()).decode("ascii"), out_w, out_h
        except Exception as e:
            last_err = e
            continue
    if not _nvenc_warned:
        _nvenc_warned = True
        logging.warning(f"[dasiwa director preview] NVENC MP4 encode failed, using WebP fallback: {last_err}")
    return None, 0, 0


def _encode_animated_webp(frames, fps, quality, max_res):
    # Source: ComfyUI-KJNodes (GPL-3.0)
    if not frames:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f
        if pf.mode != "RGB":
            pf = pf.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    buf = pyio.BytesIO()
    try:
        pil_frames[0].save(
            buf,
            format="WEBP",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=4,
        )
    except Exception as e:
        logging.warning(f"[dasiwa director preview] Animated WebP encode failed: {e}")
        return None, 0, 0
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil_frames[0].width, pil_frames[0].height


def _tiny_vae_decode_to_pil(decoder, x0, max_frames=None, stride=1):
    """KJ :306-319. taeh3 output is not [0,1]-guaranteed -> clamp before uint8."""
    # Source: ComfyUI-KJNodes (GPL-3.0)
    if x0.ndim == 4:
        rgb = decoder.decode(x0[:1])[0].movedim(0, -1).unsqueeze(0)
    elif x0.ndim == 5:
        indices = list(range(0, x0.shape[2], max(1, stride)))
        if max_frames is not None and 0 < max_frames < len(indices):
            picks = np.linspace(0, len(indices) - 1, max_frames).round().astype(int).tolist()
            indices = [indices[i] for i in picks]
        rgb = decoder.decode_video(x0[:1], frame_indices=indices)
    else:
        return []
    u8 = rgb.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


def _full_vae_decode_to_pil(vae, x0, max_frames=None):
    """KJ :282-303, minus LTX keyframe trimming. Assumes vae.decode() in [0,1]
    (H3's MiniMaxH3VideoVAE._finalize_pixels outputs [0,1], identity process_output)."""
    if vae is None or x0.ndim != 5:
        return []
    try:
        images = vae.decode(x0)
    except Exception as e:
        logging.warning(f"[dasiwa director preview] full VAE decode failed: {e}")
        return []
    if images.ndim == 5:
        images = images[0]
    if images.ndim != 4:
        return []
    t_total = images.shape[0]
    if max_frames is not None and 0 < max_frames < t_total:
        images = images[np.linspace(0, t_total - 1, max_frames).round().astype(int).tolist()]
    u8 = (images.float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


def _normalize_packed_x0(x0, latent_shapes):
    """KJ :347-358 (keyframe part dropped — H3 has none): restore the video
    sub-latent [B,24,T,H,W] from the packed AV latent."""
    # Source: ComfyUI-KJNodes (GPL-3.0)
    if latent_shapes and len(latent_shapes) > 0:
        target = latent_shapes[0]
        if x0.ndim == 3 and len(target) >= 3:
            cut = 1
            for d in target[1:]:
                cut *= int(d)
            x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(target)[1:])
    return x0


class _DirectorStepPreview:
    """OUTER_SAMPLE wrapper: per-step x0 -> frames -> PromptServer.send_sync.

    Settings come from the execution dict's 'save' block (the Director's
    Preview & Output options): live_step_preview (default True),
    preview_max_resolution (default 1024, 0 = full), preview_frames (default 1),
    preview_fps (default 12). Decoders: preview_tiny_vae name > preview_vae >
    core previewer > Latent2RGB. Never rebinds x0 (sampler reuses it)."""

    def __init__(self, settings, preview_tiny_vae, preview_vae, unique_id, client_id):
        save = dict((settings or {}).get("save") or {})
        self.max_resolution = int(save.get("preview_max_resolution", 1024) or 0)
        self.preview_frames = max(1, int(save.get("preview_frames", 1) or 1))
        self.preview_fps = max(1, int(save.get("preview_fps", 12) or 12))
        self.tiny_vae_name = preview_tiny_vae or "none"
        self.preview_vae = preview_vae
        self.unique_id = unique_id
        self.client_id = client_id

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback,
                 disable_pbar, seed, latent_shapes=None):
        guider = executor.class_obj
        model_patcher = guider.model_patcher
        latent_format = model_patcher.model.latent_format

        tiny_vae = None
        if self.tiny_vae_name not in ("", "none"):
            tiny_vae = load_tiny_vae_decoder(self.tiny_vae_name)
            if tiny_vae is not None and latent_shapes:
                channels = int(latent_shapes[0][1])
                if channels != tiny_vae.latent_channels:
                    logging.warning(
                        f"[dasiwa director preview] '{self.tiny_vae_name}' decodes "
                        f"{tiny_vae.latent_channels}-channel latents but the model's are "
                        f"{channels}-channel; ignoring it.")
                    tiny_vae = None
        # A connected full VAE only matters when the tiny VAE is unavailable.
        full_vae = self.preview_vae
        previewer = _get_core_previewer(model_patcher.load_device, latent_format)
        fallback = None
        rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
        if rgb_factors is not None and latent_preview is not None:
            fallback = latent_preview.Latent2RGBPreviewer(
                rgb_factors,
                getattr(latent_format, "latent_rgb_factors_bias", None),
                getattr(latent_format, "latent_rgb_factors_reshape", None),
            )

        original_callback = callback
        node_id = str(self.unique_id) if self.unique_id else None
        max_res = self.max_resolution
        animate = self.preview_frames > 1
        max_pil = self.preview_frames if animate else 1
        encoder = _AsyncPreviewEncoder()

        def _encode_and_send(pil_frames, sent_step, total_steps):
            if len(pil_frames) > 1:
                b64, w_, h_, mime = None, 0, 0, None
                if _NVENC_AVAILABLE:
                    b64, w_, h_ = _encode_mp4_nvenc(pil_frames, self.preview_fps, max_res)
                    if b64:
                        mime = "video/mp4"
                if not b64:
                    b64, w_, h_ = _encode_animated_webp(pil_frames, self.preview_fps, 80, max_res)
                    mime = "image/webp"
            else:
                pil = pil_frames[0].convert("RGB")
                if max_res and max_res > 0 and (pil.width > max_res or pil.height > max_res):
                    pil = ImageOps.contain(pil, (max_res, max_res), Image.LANCZOS)
                buf = pyio.BytesIO()
                pil.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                w_, h_ = pil.width, pil.height
                mime = "image/jpeg"
            if not b64:
                return
            server = prompt_server()
            if node_id and self.client_id and server is not None:
                server.send_sync(PREVIEW_EVENT, {
                    "node_id": node_id,
                    "image": b64,
                    "mime": mime,
                    "w": w_, "h": h_,
                    "step": sent_step, "total": total_steps,
                    "fps": self.preview_fps if mime in ("video/mp4", "image/webp") else None,
                }, self.client_id)

        def new_callback(step, x0, x, total):
            nonlocal tiny_vae
            try:
                x0_view = x0
                if latent_shapes:
                    x0_view = _normalize_packed_x0(x0_view, latent_shapes)
                pil_frames = []
                if tiny_vae is not None:
                    try:
                        pil_frames = _tiny_vae_decode_to_pil(tiny_vae, x0_view, max_frames=max_pil)
                    except Exception as e:
                        # OOM at 16x upscale is the likely cause — drop to the cheaper paths for good.
                        logging.warning(f"[dasiwa director preview] tiny VAE decode failed, falling back: {e}")
                        tiny_vae = None
                if not pil_frames and full_vae is not None and x0_view.ndim == 5:
                    pil_frames = _full_vae_decode_to_pil(full_vae, x0_view, max_frames=max_pil)
                if not pil_frames and animate and x0_view.ndim == 5:
                    pil_frames = _decode_video_frames_l2rgb(x0_view, latent_format, self.preview_frames)
                if not pil_frames:
                    for prev in (previewer, fallback):
                        if prev is None:
                            continue
                        try:
                            out = prev.decode_latent_to_preview(x0_view)
                        except Exception:
                            continue
                        if isinstance(out, Image.Image):
                            pil_frames = [out]
                            break
                if pil_frames:
                    encoder.submit(lambda f=pil_frames, s=step: _encode_and_send(f, s + 1, total))
            except Exception as e:
                logging.warning(f"[dasiwa director preview] frame send failed: {e}")
            if original_callback is not None:
                original_callback(step, x0, x, total)

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback,
                           disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.shutdown(drain_timeout=5.0)


def attach_step_preview(model, settings, preview_tiny_vae=None, preview_vae=None,
                       unique_id=None, client_id=None):
    """Attach the built-in step-preview wrapper. No-op for headless/API runs
    (client_id None) or when Preview & Output options disabled it
    (save.live_step_preview False)."""
    if model is None or client_id is None:
        return
    save = dict((settings or {}).get("save") or {})
    if not save.get("live_step_preview", True):
        return
    if comfy is None:
        # ComfyUI core (patcher_extension) is not importable in this environment —
        # nothing to attach. Keep sampling working; the preview simply is unavailable.
        return
    wrapper = _DirectorStepPreview(settings, preview_tiny_vae, preview_vae, unique_id, client_id)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                               "dasiwa_director_v2", wrapper)
