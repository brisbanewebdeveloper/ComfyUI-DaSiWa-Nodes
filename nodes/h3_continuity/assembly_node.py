"""Queue-only file montage: no model, latent, sampler or VAE dependencies."""
from .assembly import assemble


class DaSiWaH3ClipAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "assembly_data": ("STRING", {"default": '{"version":1,"rows":[],"repair_transitions":false,"audio_policy":"preserve"}',
                                          "multiline": True,
                                          "tooltip": "Ordered uploaded media or checkpoint chain JSON. No AI repair. Matching 24-fps SDR canvases only; video/audio are re-encoded."}),
            "filename_prefix": ("STRING", {"default": "H3_Joined",
                                            "tooltip": "New unique MP4 in ComfyUI output; 1–80 letters, numbers, underscores or hyphens. Existing files are never replaced."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/MiniMax H3/Continuity"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Uploaded files and checkpoint exports are external mutable state.
        return float("nan")

    def execute(self, assembly_data, filename_prefix):
        result = assemble(assembly_data, filename_prefix)
        return {"ui": {"h3_clip_assembly": [result], "gifs": [result["preview"]]},
                "result": (result["filename"],)}
