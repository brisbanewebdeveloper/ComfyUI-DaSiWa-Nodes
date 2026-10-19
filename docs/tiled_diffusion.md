# Tiled Diffusion (Image)

**Tiled Diffusion (Image)** is a MODEL patch for ordinary image checkpoints (SD 1.x/SDXL/Flux-family and other 4-D latent models). Every sampler step, the patched model denoises the latent in overlapping tiles and blends the per-tile predictions with positive pyramid weights, so a model can sample a canvas far larger than its native resolution (hires-fix / upscale passes) without tile seams and without the full-canvas attention cost.

Registered node type `DaSiWaTiledDiffusion`, category **DaSiWa/Image**. Written from this pack's own MiniMaxH3 per-step tiled sampler; no code from other tiled-diffusion packs.

## Wiring

`Load Checkpoint → Tiled Diffusion (Image) → KSampler (model)`. Everything else (conditioning, latent, VAE) stays as it is. Any sampler and scheduler works, because tiling happens inside the model call.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `model` | — | Image diffusion model to patch. The clone is patched; the original stays untouched. |
| `tile_width` / `tile_height` | 1024 | Tile size in pixels. Use the model's native resolution (SD 1.5: 512, SDXL/Flux: 1024). Must be a multiple of 8 and of the model's latent scale. |
| `overlap` | 128 | Overlap of neighbouring tiles in pixels; **64 px or more** blends seams away. Must be smaller than both tile sizes. |

Latents no larger than one tile are passed through with a single model call, so the patch costs nothing on small images.

## Limits

- **ControlNet is not supported**: the wrapper raises a clear error. Remove either the ControlNet or the tiled MODEL patch.
- `denoise_mask` (inpaint/mask sampling) and inpaint-model `c_concat` latents are sliced per tile; all other conditioning passes through unchanged.
- Image latents only (B,C,H,W). Video models raise an error; use **MiniMaxH3 Enhanced Upscale** for H3.
- Per-step tiling multiplies the number of model forwards by the tile count; VRAM stays near one tile's cost.
