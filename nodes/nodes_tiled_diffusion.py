"""Generic per-step tiled diffusion for 4-D image latents (MODEL patch).

Written from this pack's own H3TiledDiffusion pattern (nodes/h3_tiled_sampling.py)
and ComfyUI's model_function_wrapper contract: every sampler step the denoiser
runs once per overlapping latent tile and the x0 predictions are blended with
positive weights. Nothing is retained on the wrapper between forwards.
"""
import torch
import comfy.model_management


def tile_starts(total, tile, overlap):
    """Tile origins; the last tile is shifted back so it ends at the edge."""
    if tile >= total:
        return [0]
    starts = list(range(0, total - tile + 1, tile - overlap))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def tile_weight(th, tw, device):
    # Separable fp32 tent (pyramid), strictly > 0 so the summed weight buffer
    # never hits zero at edges/corners and fp16 never underflows the merge.
    yy = 1 - (torch.arange(th, device=device).float() - (th - 1) / 2).abs() / ((th + 1) / 2)
    xx = 1 - (torch.arange(tw, device=device).float() - (tw - 1) / 2).abs() / ((tw + 1) / 2)
    return (yy[:, None] * xx[None, :])[None, None]


class TiledDiffusion:
    """model_function_wrapper: tiles the denoiser call, blends x0 predictions."""

    def __init__(self, tile_width_px, tile_height_px, overlap_px, downscale):
        if min(tile_width_px, tile_height_px) <= 0 or overlap_px < 0 or downscale <= 0:
            raise ValueError("Tile sizes must be positive and overlap nonnegative")
        if any(v % 8 or v % downscale for v in (tile_width_px, tile_height_px, overlap_px)):
            raise ValueError(f"Tile sizes and overlap must be multiples of 8 and of the model's latent scale ({downscale}px)")
        if overlap_px >= min(tile_width_px, tile_height_px):
            raise ValueError("Overlap must be smaller than both tile sizes")
        self.tile_w, self.tile_h, self.overlap = (v // downscale for v in (tile_width_px, tile_height_px, overlap_px))

    def __call__(self, model_function, args):
        x, c = args["input"], args["c"]
        if x.ndim != 4:
            raise ValueError("Tiled Diffusion (Image) only supports 4-D image latents (B,C,H,W)")
        if c.get("control") is not None:
            raise ValueError("Tiled Diffusion (Image) does not support ControlNet; remove the tiled MODEL patch or the ControlNet")
        h, w = x.shape[-2:]
        if h <= self.tile_h and w <= self.tile_w:
            return model_function(x, args["timestep"], **c)
        # shortcut: only denoise_mask / c_concat are spatial in core conds, when a new spatial cond appears add it here.
        spatial = {k: c[k] for k in ("denoise_mask", "c_concat")
                   if isinstance(c.get(k), torch.Tensor) and tuple(c[k].shape[-2:]) == (h, w)}
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        weights = torch.zeros((1, 1, h, w), dtype=torch.float32, device=x.device)
        for y in tile_starts(h, self.tile_h, self.overlap):
            for xs in tile_starts(w, self.tile_w, self.overlap):
                comfy.model_management.throw_exception_if_processing_interrupted()
                y1, x1 = min(h, y + self.tile_h), min(w, xs + self.tile_w)
                ct = dict(c)
                ct["transformer_options"] = dict(c.get("transformer_options", {}))
                for k, v in spatial.items():
                    ct[k] = v[..., y:y1, xs:x1]
                pred = model_function(x[..., y:y1, xs:x1], args["timestep"], **ct)
                weight = tile_weight(y1 - y, x1 - xs, x.device)
                out[..., y:y1, xs:x1] += pred.float() * weight
                weights[..., y:y1, xs:x1] += weight
        return (out / weights).to(x.dtype)


class DaSiWaTiledDiffusion:
    DESCRIPTION = "Patches a MODEL so every sampling step denoises overlapping image tiles and blends them, letting any image model sample canvases far larger than its native size."

    @classmethod
    def INPUT_TYPES(cls):
        size = {"default": 1024, "min": 64, "max": 8192, "step": 8}
        return {"required": {
            "model": ("MODEL", {"tooltip": "Image diffusion model to patch; use the native checkpoint resolution as tile size."}),
            "tile_width": ("INT", {**size, "tooltip": "Tile width in pixels (multiple of the model's latent scale, usually 8)."}),
            "tile_height": ("INT", {**size, "tooltip": "Tile height in pixels (multiple of the model's latent scale, usually 8)."}),
            "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8, "tooltip": "Overlap between neighbouring tiles in pixels; 64 or more hides seams."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "DaSiWa/Image"

    def patch(self, model, tile_width, tile_height, overlap):
        downscale = model.model.latent_format.spacial_downscale_ratio
        patched = model.clone()
        patched.set_model_unet_function_wrapper(TiledDiffusion(tile_width, tile_height, overlap, downscale))
        return (patched,)
