import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
import nodes_rtx_upscaler_refiner as rtx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX effect requires CUDA")
def test_regular_node_keeps_single_image_batch_with_default_chunking(monkeypatch):
    class FakeEffect:
        def run(self, frame):
            return type("Result", (), {"image": frame.clone()})()

    monkeypatch.setattr(rtx, "_import_vfx", lambda: (object(), object()))
    monkeypatch.setattr(rtx, "_maybe_vfx_effect", lambda *args, **kwargs: __import__("contextlib").nullcontext(FakeEffect() if args[1] else None))
    monkeypatch.setattr(rtx, "_cleanup_cuda", lambda **kwargs: None)
    monkeypatch.setattr(rtx, "force_gc_and_cleanup", lambda *args: None)
    options = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["optional"]
    assert options["chunking"][1]["default"] is True
    assert options["lossless_fp16"][1]["default"] is True
    images = torch.full((5, 8, 8, 3), 0.1, dtype=torch.float32)
    kwargs = dict(images=images, denoise=True, denoise_quality="High", deblur=False,
                  deblur_quality="High", upscale="Off", upscale_quality="High",
                  resize_type="Same Size", scale=1.0, megapixels=1.0,
                  width=8, height=8, divisible_by="8", ratio_preset="1:1",
                  resize_method="Center Crop (Fill)", device_id=0, chunk_frames=2)
    chunked, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**kwargs)
    unchunked, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**kwargs, chunking=False)
    assert chunked.shape == images.shape
    assert chunked.dtype == torch.float32
    assert torch.equal(chunked, unchunked)


def test_regular_batch_lossless_fp16_only_when_exact():
    exact = torch.full((5, 8, 8, 3), 0.25, dtype=torch.float32)
    result = rtx._lossless_fp16_image_batch(exact, 2)
    assert result.dtype == torch.float16 and torch.equal(result.float(), exact)
    exact[4, 0, 0, 0] = 0.1
    result = rtx._lossless_fp16_image_batch(exact, 2)
    assert result is exact and result.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX effect requires CUDA")
def test_direct_sdk_copy_keeps_pixels_after_sdk_reuses_buffer():
    class ReusingEffect:
        def __init__(self):
            self.buffer = torch.empty((3, 8, 8), device="cuda")

        def run(self, frame):
            self.buffer.copy_(frame)
            return type("Result", (), {"image": self.buffer})()

    effect = ReusingEffect()
    row = torch.empty((8, 8, 3), dtype=torch.float32)
    frame = torch.full((3, 8, 8), 0.25, device="cuda")
    rtx._run_vfx_effect_into(effect, frame, row, torch.device("cuda:0"))
    effect.run(torch.ones_like(frame))
    assert torch.equal(row, torch.full_like(row, 0.25))

# All feature tests below use concrete SDK signatures and CPU tensors. They are
# contract tests, NOT NVIDIA SDK inference/quality validation.
from contextlib import nullcontext
from types import SimpleNamespace
from enum import IntEnum
import inspect


class Encoding(IntEnum):
    RGB8 = 0
    RGB10A2 = 1


class Levels(IntEnum):
    HIGH = 3
    ULTRA = 4
    DENOISE_HIGH = 10
    DENOISE_ULTRA = 11
    DEBLUR_HIGH = 14
    DEBLUR_ULTRA = 15
    HIGHBITRATE_HIGH = 18
    STREAMING_MEDIUM = 21
    STREAMING_ULTRA = 23


class ModernVSR:
    calls = []
    def __init__(self, quality=Levels.HIGH, strength=1.0, device=0, image_encoding=Encoding.RGB8):
        self.quality, self.strength, self.device, self.image_encoding = quality, strength, device, image_encoding
        self.calls.append((quality, strength, device, image_encoding))
        self.buffer = None
        self.closed = False
        self.output_height = self.output_width = 8
    def load(self):
        pass
    def close(self):
        self.closed = True
    def run(self, frame):
        rgb = rtx._dasiwa_rtx_unpack_rgb10a2(frame) if self.image_encoding == Encoding.RGB10A2 else frame
        rgb = torch.nn.functional.interpolate(rgb[None], size=(self.output_height, self.output_width),
            mode="bilinear", align_corners=False)[0]
        result = rtx._dasiwa_rtx_pack_rgb10a2(rgb) if self.image_encoding == Encoding.RGB10A2 else rgb
        if self.buffer is None or self.buffer.shape != result.shape:
            self.buffer = torch.empty_like(result)
        self.buffer.copy_(result)
        return SimpleNamespace(image=self.buffer)


class ModernHDR:
    calls = []
    def __init__(self, *, contrast=100, saturation=100, middle_gray=50, luminance=650, debanding_off=0, device=0):
        self.calls.append((contrast, saturation, middle_gray, luminance, debanding_off, device))
        self.closed = False
        self.buffer = None
    def close(self):
        self.closed = True
    def run(self, frame):
        # Deliberately distinct normalized signal, not simulated HDR science.
        result = rtx._dasiwa_rtx_pack_rgb10a2(frame * 0.5)
        if self.buffer is None or self.buffer.shape != result.shape:
            self.buffer = torch.empty_like(result)
        self.buffer.copy_(result)
        return SimpleNamespace(image=self.buffer)


class LegacyLevels(IntEnum):
    HIGH = 3
    DENOISE_HIGH = 10
    DEBLUR_HIGH = 14


class LegacyVSR:
    QualityLevel = LegacyLevels
    def __init__(self, quality=Levels.HIGH, device=0):
        raise AssertionError("schema must not instantiate any effect")


@pytest.fixture
def modern_sdk(monkeypatch):
    sdk = SimpleNamespace(VideoSuperRes=ModernVSR, TrueHDR=ModernHDR,
        ImageEncoding=Encoding, effects=SimpleNamespace(QualityLevel=Levels))
    monkeypatch.setitem(sys.modules, "nvvfx", sdk)
    ModernVSR.calls.clear()
    ModernHDR.calls.clear()
    return sdk


@pytest.fixture
def cpu_execution(monkeypatch, modern_sdk):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None))
    original_to = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
            kwargs["device"] = torch.device("cpu")
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(rtx, "_allocate_output_tensor", lambda shape, dtype, device, **kw: (torch.empty(shape, dtype=dtype), None))
    monkeypatch.setattr(rtx, "force_gc_and_cleanup", lambda *a: None)
    monkeypatch.setattr(rtx, "_model_management", None)


def options(images, **extra):
    values = dict(images=images, denoise=False, denoise_quality="High", deblur=False,
        deblur_quality="High", upscale="Off", upscale_quality="High", resize_type="Same Size",
        scale=1., megapixels=1., width=8, height=8, divisible_by="8", ratio_preset="1:1",
        resize_method="Center Crop (Fill)", device_id=0, lossless_fp16=False, auto_unload_models=False)
    values.update(extra)
    return values


def test_workflow_abi(modern_sdk):
    schema = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()
    required = ["images", "denoise", "denoise_quality", "deblur", "deblur_quality", "upscale",
        "upscale_quality", "resize_type", "scale", "megapixels", "width", "height", "divisible_by",
        "ratio_preset", "resize_method", "device_id"]
    old_optional = ["empty_cache", "use_mmap", "auto_unload_models", "chunking", "lossless_fp16", "chunk_frames"]
    assert list(schema["required"]) == required
    assert list(schema["optional"])[:6] == old_optional
    assert list(inspect.signature(rtx.DaSiWa_RTX_UpscalerRefiner.execute).parameters)[1:23] == required + old_optional
    assert rtx.DaSiWa_RTX_UpscalerRefiner.RETURN_TYPES == ("IMAGE",)
    assert rtx.DaSiWa_RTX_UpscalerRefiner.RETURN_NAMES == ("images",)
    assert rtx.DaSiWa_RTX_UpscalerRefiner.FUNCTION == "execute"
    assert set(rtx.NODE_CLASS_MAPPINGS) == {"DaSiWa_RTX_UpscalerRefiner"}


@pytest.mark.parametrize("kind", ["modern", "legacy", "missing"])
def test_capability_probe_no_allocation(monkeypatch, modern_sdk, kind):
    if kind == "legacy":
        monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=LegacyVSR))
    elif kind == "missing":
        monkeypatch.setitem(sys.modules, "nvvfx", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("GPU probe")))
    capabilities = rtx._dasiwa_rtx_capabilities()
    schema = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()
    assert capabilities["hdr"] == (kind == "modern")
    assert capabilities["encoding"] == (kind == "modern")
    assert "hdr_enabled" in schema["optional"]
    assert "image_encoding" in schema["optional"]
    assert schema["optional"]["hdr_enabled"][1]["dasiwa_unavailable"] == (kind != "modern")
    assert schema["optional"]["image_encoding"][1]["dasiwa_unavailable"] == (kind != "modern")
    assert schema["required"]["upscale"][0][:3] == ["Off", "VSR", "High Bitrate"]


def test_old_sdk_strength_is_detected_by_signature(monkeypatch):
    class OldStrength:
        def __init__(self, quality=3, strength=1.0, device=0):
            raise AssertionError("schema allocation")
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=OldStrength))
    assert rtx._dasiwa_rtx_capabilities()["strength"]
    assert "denoise_strength" in rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["optional"]


@pytest.mark.parametrize("extra, message", [
    ({"image_encoding": "10-bit RGB (RGB10A2)"}, "RGB10A2"),
    ({"hdr_enabled": True}, "TrueHDR"),
    ({"upscale": "Streaming Ultra"}, "Streaming Ultra"),
    ({"denoise_strength": .5}, "strength")])
def test_explicit_unsupported_requests_fail(monkeypatch, extra, message):
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=LegacyVSR))
    with pytest.raises(RuntimeError, match=message):
        rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(torch.zeros(1, 8, 8, 3), **extra))


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_rgb10a2_roundtrip(noncontiguous):
    rgb = torch.stack([torch.linspace(0, 1, 35).reshape(5, 7), torch.ones(5, 7), torch.zeros(5, 7)])
    if noncontiguous:
        rgb = rgb.transpose(1, 2)
    packed = rtx._dasiwa_rtx_pack_rgb10a2(rgb)
    assert packed.dtype == torch.uint32 and packed.is_contiguous()
    assert torch.equal(rtx._dasiwa_rtx_unpack_rgb10a2(packed), (rgb * 1023).round() / 1023)
    for channel in range(3):
        color = torch.zeros(3, 1, 1)
        color[channel] = 1
        assert int(rtx._dasiwa_rtx_pack_rgb10a2(color).item()) == (1023 << (channel * 10)) | (3 << 30)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("passes", [(True, False, False), (False, True, False), (False, False, True), (True, True, True)])
def test_pipeline_and_reused_buffers(cpu_execution, packed, passes):
    images = torch.stack([torch.full((8, 8, 3), .25), torch.full((8, 8, 3), .75)])
    denoise, deblur, upscale = passes
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, denoise=denoise, deblur=deblur,
        upscale="VSR" if upscale else "Off", image_encoding="10-bit RGB (RGB10A2)" if packed else "8-bit RGB",
        denoise_strength=.2, deblur_strength=.4, upscale_strength=.6))
    expected = (images * 1023).round() / 1023 if packed else images
    assert result.dtype == torch.float32 and torch.equal(result, expected)
    expected_strengths = [s for active, s in zip(passes, [.2, .4, .6]) if active]
    assert [call[1] for call in ModernVSR.calls] == expected_strengths
    assert all(call[3] == (Encoding.RGB10A2 if packed else Encoding.RGB8) for call in ModernVSR.calls)


@pytest.mark.parametrize("mode, level", [("Streaming Medium", Levels.STREAMING_MEDIUM), ("Streaming Ultra", Levels.STREAMING_ULTRA)])
def test_streaming_exact_levels(cpu_execution, mode, level):
    rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(torch.ones(1, 8, 8, 3), upscale=mode, upscale_quality="Low"))
    assert ModernVSR.calls[0][0] == level


@pytest.mark.parametrize("multipass", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
def test_hdr_only_and_final_pass(cpu_execution, multipass, alpha):
    images = torch.stack([torch.full((8, 8, 4), .25), torch.full((8, 8, 4), .75)])
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, hdr_enabled=True,
        denoise=multipass, deblur=multipass, upscale="VSR" if multipass else "Off",
        preserve_alpha=alpha, channel_integrity_check=True, hdr_contrast=120, hdr_saturation=110,
        hdr_middle_gray=60, hdr_luminance=1000, hdr_debanding=False))
    assert result.dtype == torch.float32
    assert result.shape[-1] == (4 if alpha else 3)
    assert torch.equal(result[..., :3], (images[..., :3] * .5 * 1023).round() / 1023)
    if alpha:
        assert torch.equal(result[..., 3], images[..., 3])
    assert ModernHDR.calls == [(120, 110, 60, 1000, 1, 0)]


@pytest.mark.parametrize("method", ["Center Crop (Fill)", "Letterbox (Fit)"])
def test_alpha_spatial_alignment(cpu_execution, method):
    alpha = torch.linspace(0, 1, 128).reshape(8, 16, 1)
    images = torch.cat([torch.full((8, 16, 3), .5), alpha], dim=-1)[None]
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, upscale="VSR", resize_type="Manual",
        width=16, height=16, preserve_alpha=True, resize_method=method, image_encoding="10-bit RGB (RGB10A2)"))
    aligned = rtx._fit_frame_to_target_aspect(alpha.permute(2, 0, 1), 16, 16, method)
    expected = torch.nn.functional.interpolate(aligned[None], size=(16, 16), mode="bilinear", align_corners=False)[0, 0]
    assert result.shape == (1, 16, 16, 4)
    assert torch.equal(result[0, :, :, 3], expected)


def test_legacy_no_effect_call_and_alpha_default(cpu_execution):
    images = torch.full((1, 8, 8, 4), .25)
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images))
    assert result.shape == (1, 8, 8, 3)
    assert torch.equal(result, images[..., :3])
    assert not ModernHDR.calls and not ModernVSR.calls
    assert rtx._projected_output_bytes(2, 8, 8, torch.float32, 4) == 2048


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_integrity_nonfinite(value):
    frame = torch.ones(3, 8, 8)
    result = frame.clone()
    result[0, 0, 0] = value
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        rtx._dasiwa_rtx_integrity(frame, result, "VSR")


def test_integrity_collapse_and_legitimate_frames():
    frame = torch.ones(3, 8, 8)
    result = frame.clone()
    result[0] = 0
    with pytest.raises(RuntimeError, match="suspected channel collapse"):
        rtx._dasiwa_rtx_integrity(frame, result, "VSR")
    for legitimate in (torch.zeros_like(frame), result, torch.full_like(frame, .001)):
        rtx._dasiwa_rtx_integrity(legitimate, legitimate, "VSR")


def test_integrity_preclamp_and_hdr_not_sdr(cpu_execution, monkeypatch):
    original_run = ModernVSR.run
    def run(self, frame):
        result = original_run(self, frame)
        result.image[0, 0, 0] = float("inf")
        return result
    monkeypatch.setattr(ModernVSR, "run", run)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(torch.ones(1, 8, 8, 3), denoise=True, channel_integrity_check=True))


def test_new_settings_constructor_defaults_match_installed_contract(modern_sdk):
    instance = rtx._create_vfx_effect(ModernVSR, Levels.ULTRA, 2)
    assert instance.strength == 1.0 and instance.device == 2 and instance.image_encoding == Encoding.RGB8


def test_legacy_sdk_execution_with_old_prompt(cpu_execution, monkeypatch):
    class WorkingLegacy(ModernVSR):
        QualityLevel = LegacyLevels
        def __init__(self, quality=LegacyLevels.HIGH, device=0):
            super().__init__(quality=quality, device=device)
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=WorkingLegacy))
    images = torch.full((2, 8, 8, 3), .25)
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, denoise=True))
    assert torch.equal(result, images)
    assert ModernVSR.calls == [(LegacyLevels.DENOISE_HIGH, 1.0, 0, Encoding.RGB8)]


def test_fp16_hdr_input_keeps_pq_precision(cpu_execution):
    images = torch.full((1, 8, 8, 3), .25, dtype=torch.float16)
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, hdr_enabled=True, lossless_fp16=True))
    expected = torch.full((1, 8, 8, 3), 128 / 1023)
    assert torch.equal(result, expected) and result.dtype == torch.float32


@pytest.mark.parametrize("extra, message", [
    ({"image_encoding": "10-bit RGB (RGB10A2)"}, "RGB10A2"),
    ({"hdr_enabled": True}, "TrueHDR"),
    ({"denoise_strength": .5}, "strength")])
def test_core_input_preparation_preserves_unsupported_requests(monkeypatch, extra, message):
    import execution
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=LegacyVSR))
    prepared, *_ = execution.get_input_data(extra, rtx.DaSiWa_RTX_UpscalerRefiner, "rtx-test")
    assert {name: value[0] for name, value in prepared.items()} == extra
    with pytest.raises(RuntimeError, match=message):
        rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(
            torch.zeros(1, 8, 8, 3), **{name: value[0] for name, value in prepared.items()}))
