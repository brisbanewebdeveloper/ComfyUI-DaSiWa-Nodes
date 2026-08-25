import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

# The repo's nodes/ directory is a namespace package. When a live ComfyUI
# checkout is on sys.path (required to run the comfy-gated tests), ComfyUI's
# own top-level nodes.py shadows it, so `from nodes import ...` resolves to the
# wrong module. To keep this one test file runnable in BOTH the repo dev venv
# (no ComfyUI) and a live ComfyUI checkout (comfy importable), we load the
# repo modules by absolute file path under a synthetic package name, so no
# `import nodes.X` ever happens. Relative imports inside the loaded modules
# (.helper_logging, .tiny_vae, ...) resolve through the synthetic package's
# __path__, which points at the real nodes/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_NODES_DIR = _REPO_ROOT / "nodes"
_PKG = "dasiwa_director_v2_nodes_pkg"


def _load_repo_module(name):
    """Load nodes/<name>.py under the synthetic package (idempotent)."""
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_NODES_DIR)]
        sys.modules[_PKG] = pkg
    module_name = f"{_PKG}.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _NODES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_repo_module("helper_minimax_h3_director_execute_v2")
director_v2 = _load_repo_module("nodes_minimax_h3_director_v2")
preview = _load_repo_module("helper_minimax_h3_director_preview_v2")
PREVIEW_EVENT = preview.PREVIEW_EVENT
_DirectorStepPreview = preview._DirectorStepPreview
attach_step_preview = preview.attach_step_preview
MiniMaxH3DirectorV2 = director_v2.MiniMaxH3DirectorV2

OBJ = object()

# Step-preview tests need a live ComfyUI checkout (comfy.patcher_extension,
# comfy_extras samplers, the real VAEDecode). They skip in the repo dev venv.
COMFY_AVAILABLE = importlib.util.find_spec("comfy") is not None
comfy_patch = None
if COMFY_AVAILABLE:
    import comfy.patcher_extension as comfy_patch

try:
    import folder_paths  # noqa: F401
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


def _tl(internal_execution=None):
    return json.dumps({"internal_execution": internal_execution or {}})


def test_optional_sampling_sockets_present():
    opt = MiniMaxH3DirectorV2.INPUT_TYPES()["optional"]
    for key in ("external_sampler", "external_scheduler", "external_steps",
                "external_shift_video", "external_shift_audio"):
        assert key in opt, f"missing optional socket {key}"
    assert opt["external_sampler"][0] == "STRING"
    assert opt["external_sampler"][1].get("forceInput") is True
    assert opt["external_scheduler"][0] == "STRING"
    assert opt["external_scheduler"][1].get("forceInput") is True
    assert opt["external_steps"][0] == "INT"
    assert opt["external_steps"][1].get("forceInput") is True
    assert opt["external_shift_video"][0] == "FLOAT"
    assert opt["external_shift_video"][1].get("forceInput") is True
    assert opt["external_shift_audio"][0] == "FLOAT"
    assert opt["external_shift_audio"][1].get("forceInput") is True


def test_preview_sockets_present():
    opt = MiniMaxH3DirectorV2.INPUT_TYPES()["optional"]
    # The tiny-VAE selector is a bare-list COMBO spec (the same shape as
    # core VAELoader.vae_name) — rendered like a model selector. The
    # hollow input ring is stripped client-side: the Director's JS nulls the
    # socket's shape (VAELoader.vae_name is required -> no ring; an optional
    # socket would draw a HollowCircle). No `socketless` option dict key is
    # relied on: in frontend v1.49.x addComboWidget only copies
    # {values,advanced,hidden} onto widget.options, so a legacy
    # options-dict `socketless` flag never reaches the gate and does nothing.
    assert isinstance(opt["preview_tiny_vae"][0], list)
    assert opt["preview_tiny_vae"][1].get("forceInput") is not True
    assert opt["preview_tiny_vae"][1].get("default") == "none"
    # The full-quality VAE path remains an optional input socket.
    assert opt["preview_vae"][0] == "VAE"
    # INPUT_TYPES() is evaluated once at node load: the options are the
    # list itself (element 0 of the tuple), not a callable or a dict key.
    names = list(opt["preview_tiny_vae"][0])
    assert "none" in names
    if HAS_FOLDER_PATHS:
        # Options come from the models/vae_approx folder listing.
        assert any(str(name).startswith("taeh3") for name in names)


def test_hidden_unique_id_present():
    hidden = MiniMaxH3DirectorV2.INPUT_TYPES()["hidden"]
    assert hidden["unique_id"] == "UNIQUE_ID"


def _capture_settings(monkeypatch):
    captured = {}

    def fake_execute_h3(*args, **kwargs):
        captured["settings"] = kwargs.get("settings") or args[6]
        return (None, None)

    monkeypatch.setattr(director_v2, "execute_h3", fake_execute_h3)
    monkeypatch.setattr(director_v2, "publish_media_output", lambda *args, **kwargs: {"ui": {}})
    return captured


def test_external_sampling_wins_over_internal(monkeypatch):
    captured = _capture_settings(monkeypatch)
    node = MiniMaxH3DirectorV2()
    node.execute(
        mode="T2VA", prompt="", width=16, height=16, duration=1, ref_image_size="match",
        timeline_data=_tl({"sampler": "res_multistep", "scheduler": "simple", "steps": 25,
                           "shift_video": 11, "shift_audio": 4}),
        fl2va_model=OBJ, clip=OBJ, vae=OBJ,
        external_sampler="uni_pc", external_scheduler="karras", external_steps=40,
        external_shift_video=2.0, external_shift_audio=1.0,
    )
    settings = captured["settings"]
    assert settings["sampler"] == "uni_pc"
    assert settings["scheduler"] == "karras"
    assert settings["steps"] == 40
    assert settings["shift_video"] == 2.0
    assert settings["shift_audio"] == 1.0


def test_external_empty_falls_back_to_internal(monkeypatch):
    captured = _capture_settings(monkeypatch)
    node = MiniMaxH3DirectorV2()
    node.execute(
        mode="T2VA", prompt="", width=16, height=16, duration=1, ref_image_size="match",
        timeline_data=_tl({"sampler": "res_multistep", "scheduler": "karras", "steps": 30,
                           "shift_video": 7.0, "shift_audio": 2.0}),
        fl2va_model=OBJ, clip=OBJ, vae=OBJ,
        external_sampler="", external_scheduler="", external_steps=0,
        external_shift_video=0.0, external_shift_audio=0.0,
    )
    settings = captured["settings"]
    assert settings["sampler"] == "res_multistep"
    assert settings["scheduler"] == "karras"
    assert settings["steps"] == 30
    assert settings["shift_video"] == 7.0
    assert settings["shift_audio"] == 2.0


def test_both_absent_uses_helper_defaults(monkeypatch):
    # Neither external sockets nor internal_execution carry sampling keys, so
    # nothing is written into the execution dict: the helper applies its
    # documented defaults at consumption time via settings.get(...).
    captured = _capture_settings(monkeypatch)
    node = MiniMaxH3DirectorV2()
    node.execute(
        mode="T2VA", prompt="", width=16, height=16, duration=1, ref_image_size="match",
        timeline_data=_tl({}),
        fl2va_model=OBJ, clip=OBJ, vae=OBJ,
    )
    settings = captured["settings"]
    for key in ("sampler", "scheduler", "steps", "shift_video", "shift_audio"):
        assert key not in settings, f"{key} must not be force-set when nothing is configured"
    # The effective values the helper computes from this dict are the defaults.
    assert settings.get("sampler", "res_multistep") == "res_multistep"
    assert settings.get("scheduler", "simple") == "simple"
    assert int(settings.get("steps", 25)) == 25
    assert float(settings.get("shift_video", 11.0)) == 11.0
    assert float(settings.get("shift_audio", 4.0)) == 4.0


def test_execute_threads_preview_and_identity_settings(monkeypatch):
    captured = _capture_settings(monkeypatch)
    node = MiniMaxH3DirectorV2()
    node.execute(
        mode="T2VA", prompt="", width=16, height=16, duration=1, ref_image_size="match",
        timeline_data=_tl({}),
        fl2va_model=OBJ, clip=OBJ, vae=OBJ,
        preview_tiny_vae="taeh3.safetensors", preview_vae=OBJ, unique_id="node-7",
    )
    settings = captured["settings"]
    assert settings["preview_tiny_vae"] == "taeh3.safetensors"
    assert settings["preview_vae"] is OBJ
    assert settings["unique_id"] == "node-7"


def test_director_v2_is_a_distinct_internal_execution_node():
    schema = MiniMaxH3DirectorV2.INPUT_TYPES()

    assert "clip" in schema["optional"]
    assert "vae" in schema["optional"]
    assert "fl2va_clip" not in schema["optional"]
    assert "ref2va_clip" not in schema["optional"]
    assert "patched_model" not in schema["optional"]
    assert "patched_clip" not in schema["optional"]
    assert schema["optional"]["fl2va_model"][0] == "MODEL"
    assert schema["optional"]["ref2va_model"][0] == "MODEL"
    assert MiniMaxH3DirectorV2.OUTPUT_NODE is True
    assert MiniMaxH3DirectorV2.RETURN_TYPES == ("FLOAT", "INT", "IMAGE")
    assert MiniMaxH3DirectorV2.RETURN_NAMES == ("frame_rate", "duration", "images")


def test_director_v2_prefers_the_returned_lora_model():
    assert MiniMaxH3DirectorV2.select_execution_model("REF2VA", "fl2", "ref") == "ref"
    assert MiniMaxH3DirectorV2.select_execution_model("I2VA", "fl2", "ref") == "fl2"


if COMFY_AVAILABLE:
    # nodes (ComfyUI's top-level nodes.py) hosts VAEDecode/LoraLoader — the
    # live env only. The repo modules are already loaded via the synthetic
    # package above; reference them directly so no `nodes.*` submodule import
    # is attempted (ComfyUI's nodes.py is a file, not a package).
    import nodes as nodes_module

    preview_module = preview
    tiny_vae_module = _load_repo_module("tiny_vae")

    def _fake_video_tensor():
        return torch.zeros(1, 24, 4, 2, 2)

    def _fake_prompt_server():
        fake = type("FakePromptServer", (), {
            "instance": type("FakeInstance", (), {
                "client_id": "client-1",
                "sent": [],
                "send_sync": lambda self, event, data, sid=None: self.sent.append((event, data, sid)),
            })(),
        })()
        return fake

    def _fake_model_patcher(monkeypatch):
        class Model:
            latent_rgb_factors = None
            latent_format = None

        class Patcher:
            def __init__(self):
                self.model = Model()
                self.load_device = torch.device("cpu")
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.wrappers = {}

            def add_wrapper_with_key(self, wrapper_type, key, wrapper):
                self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

            def get_wrappers(self, wrapper_type, key):
                return self.wrappers.get(wrapper_type, {}).get(key, [])

        patcher = Patcher()
        fake_server = _fake_prompt_server()
        fake_previewer = SimpleNamespace(decode_latent_to_preview=lambda self, x0: None)
        # The per-step send path lives in the preview module (its module-level
        # prompt_server() helper), so patch that — not the execute helper.
        monkeypatch.setattr(preview_module, "prompt_server", lambda: fake_server.instance)
        monkeypatch.setattr(preview_module, "latent_preview", SimpleNamespace(
            get_previewer=lambda device, fmt: fake_previewer,
            Latent2RGBPreviewer=lambda *a, **k: None,
        ))
        return patcher, fake_server.instance

    def _fake_executor(patcher):
        class Executor:
            class_obj = type("Guider", (), {"model_patcher": patcher})()

            # The wrapper advances the executor via __call__
            # (executor(noise, latent, sampler, sigmas, denoise_mask, callback,
            # disable_pbar, seed, latent_shapes=...)), so self is the executor.
            def __call__(self, noise, latent_image, sampler, sigmas, denoise_mask, callback,
                         disable_pbar, seed, latent_shapes=None):
                if callback:
                    callback(0, latent_image, latent_image, 1)
                return "sampled"

        return Executor()

    def _fake_tiny_vae(monkeypatch):
        def _decode_video(latent, frame_indices=None):
            # Honour frame_indices (the wrapper samples only the requested
            # frames) and emit a real N x H x W x 3 RGB "decoded video" so the
            # JPEG/animated encoders and PIL all see a valid 3-channel array.
            idx = list(range(4)) if frame_indices is None else list(frame_indices)
            return torch.zeros(len(idx), 8, 8, 3, dtype=torch.float32)

        decoder = SimpleNamespace(
            latent_channels=24,
            upscale_ratio=16,
            decode_video=_decode_video,
        )
        monkeypatch.setattr(tiny_vae_module, "load_tiny_vae_decoder", lambda name, device=None, dtype=None: decoder)
        monkeypatch.setattr(preview_module, "load_tiny_vae_decoder", lambda name, device=None, dtype=None: decoder)
        return decoder

    def test_step_preview_wrapper_attached(monkeypatch):
        patcher, sent = _fake_model_patcher(monkeypatch)
        _fake_tiny_vae(monkeypatch)
        settings = {
            "save": {"live_step_preview": True, "preview_max_resolution": 1024, "preview_frames": 1, "preview_fps": 12},
            "preview_tiny_vae": "taeh3.safetensors",
            "preview_vae": None,
            "unique_id": "abc",
            "_client_id": "cid",
        }
        attach_step_preview(patcher, settings, preview_tiny_vae="taeh3.safetensors",
                           preview_vae=None, unique_id="abc", client_id="cid")
        wrappers = patcher.get_wrappers(comfy_patch.WrappersMP.OUTER_SAMPLE, "dasiwa_director_v2")
        assert len(wrappers) == 1

        result = wrappers[0](_fake_executor(patcher), None, _fake_video_tensor(), None, None, None, None,
                            False, None, latent_shapes=[(1, 24, 4, 2, 2)])
        assert result == "sampled"
        assert len(sent.sent) >= 1
        event, payload, sid = sent.sent[0]
        assert event == PREVIEW_EVENT
        assert payload["node_id"] == "abc"
        assert payload["mime"] == "image/jpeg"
        assert payload["image"]
        assert payload["step"] == 1 and payload["total"] == 1
        assert sid == "cid"

    def test_step_preview_disabled_flag(monkeypatch):
        patcher, _ = _fake_model_patcher(monkeypatch)
        attach_step_preview(patcher, {"save": {"live_step_preview": False}}, unique_id="abc", client_id="cid")
        assert patcher.get_wrappers(comfy_patch.WrappersMP.OUTER_SAMPLE, "dasiwa_director_v2") == []

    def test_client_id_none_skips_attach(monkeypatch):
        patcher, _ = _fake_model_patcher(monkeypatch)
        attach_step_preview(patcher, {"save": {"live_step_preview": True}}, unique_id="abc", client_id=None)
        assert patcher.get_wrappers(comfy_patch.WrappersMP.OUTER_SAMPLE, "dasiwa_director_v2") == []

    def test_preview_vae_input_precedence(monkeypatch):
        # Full-VAE path: preview_vae decodes instead of the tiny VAE.
        patcher, sent = _fake_model_patcher(monkeypatch)
        monkeypatch.setattr(preview_module, "load_tiny_vae_decoder", lambda name, device=None, dtype=None: None)
        decode_calls = []
        def _full_decode(latent):
            decode_calls.append(1)
            # _full_vae_decode_to_pil expects a (T,H,W,C) tensor in [0,1].
            return torch.zeros(1, 8, 8, 3, dtype=torch.float32)

        full_vae = SimpleNamespace(decode=_full_decode)
        settings = {"save": {"live_step_preview": True, "preview_max_resolution": 0,
                             "preview_frames": 1, "preview_fps": 12},
                    "preview_vae": full_vae, "unique_id": "abc", "_client_id": "cid"}
        attach_step_preview(patcher, settings, preview_tiny_vae="none", preview_vae=full_vae,
                           unique_id="abc", client_id="cid")
        wrappers = patcher.get_wrappers(comfy_patch.WrappersMP.OUTER_SAMPLE, "dasiwa_director_v2")
        wrappers[0](_fake_executor(patcher), None, _fake_video_tensor(), None, None, None, None,
                    False, None, latent_shapes=[(1, 24, 4, 2, 2)])
        assert decode_calls, "connected preview_vae must run the full-quality decode path"
        assert sent.sent and sent.sent[0][0] == PREVIEW_EVENT

        # Latent2RGB fallback: neither VAE, rgb_factors present on the latent format.
        patcher2, sent2 = _fake_model_patcher(monkeypatch)

        class RGBFormat:
            latent_rgb_factors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            latent_rgb_factors_bias = None
            latent_rgb_factors_reshape = None

        patcher2.model.latent_format = RGBFormat()
        monkeypatch.setattr(preview_module, "latent_preview", SimpleNamespace(
            get_previewer=lambda device, fmt: SimpleNamespace(
                decode_latent_to_preview=lambda x0: Image.new("RGB", (8, 8)),
            ),
            Latent2RGBPreviewer=lambda *a, **k: None,
        ))
        settings2 = {"save": {"live_step_preview": True, "preview_max_resolution": 0,
                              "preview_frames": 1, "preview_fps": 12},
                     "preview_vae": None, "unique_id": "abc", "_client_id": "cid"}
        attach_step_preview(patcher2, settings2, preview_tiny_vae="none", preview_vae=None,
                           unique_id="abc", client_id="cid")
        sent2.sent.clear()
        wrappers2 = patcher2.get_wrappers(comfy_patch.WrappersMP.OUTER_SAMPLE, "dasiwa_director_v2")
        wrappers2[0](_fake_executor(patcher2), None, _fake_video_tensor(), None, None, None, None,
                     False, None, latent_shapes=[(1, 24, 4, 2, 2)])
        assert sent2.sent and sent2.sent[0][0] == PREVIEW_EVENT

    def test_step_preview_attached_in_execute_h3(monkeypatch):
        # execute_h3 resolves its sampler/VAE names via function-local imports,
        # so the fakes are patched on the source modules.
        from comfy_extras import nodes_custom_sampler as custom_sampler_module
        from comfy_extras import nodes_minimax_h3 as minimax_module
        patcher, _ = _fake_model_patcher(monkeypatch)

        class FakeImageToVideo:
            @staticmethod
            def execute(clip, vae, prompt, width, height, length, first_frame, last_frame):
                return ["cond"], {"samples": _fake_video_tensor()}

        class FakeSigmaShift:
            # The real MiniMaxH3SigmaShift clones the model patcher; the fake
            # passes the (fake) patcher straight through so attach_step_preview
            # receives the same object the test then inspects.
            @staticmethod
            def execute(model, shift_video, shift_audio):
                return [model]

        class FakeGuiderNode:
            @staticmethod
            def execute(model, positive):
                return [type("G", (), {"model_patcher": patcher})()]

        class FakeSamplerNode:
            @staticmethod
            def execute(name):
                return ["sampler"]

        class FakeSchedulerNode:
            @staticmethod
            def execute(model, scheduler, steps, denoise):
                assert scheduler == "karras" and steps == 12
                return ["sigmas"]

        class FakeNoise:
            @staticmethod
            def execute(seed):
                return [type("N", (), {"seed": seed, "generate_noise": lambda self, latent: latent})()]

        class FakeSamplerCustom:
            @staticmethod
            def execute(noise, guider, sampler, sigmas, latent):
                return [{"samples": _fake_video_tensor()}]

        class FakeVAEDecode:
            def __init__(self):
                pass

            def decode(self, vae, latent):
                return [torch.zeros(1, 1, 2, 2, 3)]

        monkeypatch.setattr(minimax_module, "MiniMaxH3ImageToVideo", FakeImageToVideo)
        monkeypatch.setattr(minimax_module, "MiniMaxH3SigmaShift", FakeSigmaShift)
        monkeypatch.setattr(custom_sampler_module, "BasicGuider", FakeGuiderNode)
        monkeypatch.setattr(custom_sampler_module, "KSamplerSelect", FakeSamplerNode)
        monkeypatch.setattr(custom_sampler_module, "BasicScheduler", FakeSchedulerNode)
        monkeypatch.setattr(custom_sampler_module, "RandomNoise", FakeNoise)
        monkeypatch.setattr(custom_sampler_module, "SamplerCustomAdvanced", FakeSamplerCustom)
        monkeypatch.setattr(nodes_module, "VAEDecode", FakeVAEDecode)

        guide = {"mode": "T2VA", "width": 16, "height": 16, "length": 5, "resolved_prompt": "p",
                 "first_frame": None, "last_frame": None}
        settings = {"sampler": "res_multistep", "scheduler": "karras", "steps": 12,
                    "preview_tiny_vae": "taeh3.safetensors", "unique_id": "abc", "_client_id": "cid"}
        helper.execute_h3(guide, patcher, None, None, None, 7, settings)
        wrappers = patcher.get_wrappers(comfy_patch.WrappersMP.OUTER_SAMPLE, "dasiwa_director_v2")
        assert len(wrappers) == 1
