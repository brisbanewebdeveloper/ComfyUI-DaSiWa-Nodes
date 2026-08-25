"""Contract tests for Director-owned Image Inpaint execution."""
from nodes.helper_minimax_h3_director_execute_v2 import (
    DEFAULT_POSTPROCESS_RECIPE,
    normalize_postprocess_recipe,
)


def test_recipe_defaults_are_disabled_and_stably_ordered():
    recipe = normalize_postprocess_recipe(None)

    assert recipe == DEFAULT_POSTPROCESS_RECIPE
    assert [stage["id"] for stage in recipe] == [
        "frame_interpolation", "torch_resize", "model_upscale", "rtx_refiner", "watermark",
    ]
    assert not any(stage["enabled"] for stage in recipe)


def test_recipe_normalization_keeps_known_overrides_and_ignores_unknown_stages():
    recipe = normalize_postprocess_recipe([
        {"id": "torch_resize", "enabled": True, "scale_multiplier": 3},
        {"id": "unknown", "enabled": True},
    ])

    resize = next(stage for stage in recipe if stage["id"] == "torch_resize")
    assert resize["enabled"] is True
    assert resize["scale_multiplier"] == 3
    assert len(recipe) == len(DEFAULT_POSTPROCESS_RECIPE)


def test_rtx_refiner_recipe_exposes_full_node_options():
    recipe = normalize_postprocess_recipe(None)
    rtx = next(stage for stage in recipe if stage["id"] == "rtx_refiner")

    # The burger modal offers every option the real RTX node exposes.
    for key in (
        "denoise", "denoise_quality", "deblur", "deblur_quality",
        "upscale", "upscale_quality", "resize_type", "scale", "megapixels",
        "width", "height", "divisible_by", "ratio_preset", "resize_method",
        "device_id", "empty_cache", "use_mmap", "auto_unload_models",
    ):
        assert key in rtx, f"missing rtx_refiner option: {key}"
    # Defaults match the real node (VSR Ultra 2x, 8-aligned).
    assert rtx["upscale"] == "VSR"
    assert rtx["upscale_quality"] == "Ultra"
    assert rtx["resize_type"] == "Scale"
    assert rtx["scale"] == 2.0
    assert rtx["divisible_by"] == "8"
    assert rtx["use_mmap"] is False
    assert rtx["auto_unload_models"] is True
