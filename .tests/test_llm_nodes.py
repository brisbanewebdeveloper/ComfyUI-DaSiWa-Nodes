import importlib.util
import sys
import types
from pathlib import Path

import pytest


folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = "/tmp/comfy-models"
folder_paths.add_model_folder_path = lambda *_: None
folder_paths.get_folder_paths = lambda _: [folder_paths.models_dir]
folder_paths.get_full_path = lambda *_: None
sys.modules["folder_paths"] = folder_paths

HELPER_PATH = Path(__file__).parents[1] / "nodes" / "helper_logging.py"
helper_spec = importlib.util.spec_from_file_location("helper_logging", HELPER_PATH)
assert helper_spec is not None and helper_spec.loader is not None
helper_logging = importlib.util.module_from_spec(helper_spec)
sys.modules["helper_logging"] = helper_logging
helper_spec.loader.exec_module(helper_logging)
MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_llm.py"
spec = importlib.util.spec_from_file_location("nodes_llm", MODULE_PATH)
assert spec is not None and spec.loader is not None
llm_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_nodes)
# Keep the stub only in the module under test; it must not leak into sibling tests.
sys.modules.pop("folder_paths", None)


def test_selector_exposes_kv_cache_strategy_and_llama_cpp_controls(monkeypatch):
    monkeypatch.setattr(llm_nodes, "_resolve_model_path", lambda *args, **kwargs: "/models/chat.gguf")

    config = llm_nodes.DaSiWa_LLMModelSelector().select(
        "None", "", "", "main", False, "llama_cpp", "text", "cuda", "auto", "none",
        "unload_after_run", False, "auto", "quantized", "quanto", 4, 128,
        8192, -1, 0, "", "", "http://127.0.0.1:11434", 300, False,
    )[0]

    assert config["backend"] == "llama_cpp"
    assert config["kv_cache_implementation"] == "quantized"
    assert config["kv_cache_quant_backend"] == "quanto"
    assert config["llama_n_ctx"] == 8192
    assert config["llama_n_gpu_layers"] == -1
    inputs = llm_nodes.DaSiWa_LLMModelSelector.INPUT_TYPES()
    assert list(inputs["required"])[-2:] == ["ollama_url", "ollama_timeout"]
    assert inputs["optional"]["allow_remote_ollama"][1]["default"] is False
    assert config["allow_remote_ollama"] is False


def test_transformers_generation_kwargs_support_quantized_kv_cache():
    config = {
        "kv_cache_implementation": "quantized",
        "kv_cache_quant_backend": "quanto",
        "kv_cache_nbits": 4,
        "kv_cache_residual_length": 128,
    }

    kwargs = llm_nodes._generation_cache_kwargs(config, True)

    assert kwargs["use_cache"] is True
    assert kwargs["cache_implementation"] == "quantized"
    assert kwargs["cache_config"] == {
        "backend": "quanto",
        "nbits": 4,
        "residual_length": 128,
    }


def test_unload_releases_custom_llm_and_comfy_managed_models(monkeypatch):
    calls = []

    class Model:
        def to(self, device):
            calls.append(("model_to", device))

    llm_nodes._LLM_CACHE[("/models/chat",)] = llm_nodes._LoadedLLM(Model(), None, None, False)
    monkeypatch.setattr(llm_nodes, "_cleanup_cuda", lambda: calls.append(("cleanup_cuda",)))
    monkeypatch.setitem(sys.modules, "comfy.model_management", types.SimpleNamespace(
        unload_all_models=lambda: calls.append(("unload_all_models",)),
        soft_empty_cache=lambda: calls.append(("soft_empty_cache",)),
    ))
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(
        model_management=sys.modules["comfy.model_management"]
    ))

    llm_nodes._release_all_model_memory()

    assert llm_nodes._LLM_CACHE == {}
    assert ("model_to", "cpu") in calls
    assert ("unload_all_models",) in calls
    assert ("soft_empty_cache",) in calls
    assert ("cleanup_cuda",) in calls


def test_ollama_unload_request_sets_keep_alive_zero(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"message": {"content": "done"}}'

    def fake_urlopen(request, timeout):
        captured["payload"] = request.data
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(llm_nodes.urlrequest, "urlopen", fake_urlopen)
    config = {
        "model_path": "qwen3:8b",
        "ollama_url": "http://ollama:11434",
        "allow_remote_ollama": True,
        "ollama_timeout": 12,
    }

    response, image_count = llm_nodes._run_ollama_generation(
        config, "", "hello", 32, 0.7, 0.9, 1.0, -1, [], unload_after_request=True,
    )

    assert response == "done"
    assert image_count == 0
    assert captured["timeout"] == 12
    assert llm_nodes.json.loads(captured["payload"])["keep_alive"] == 0


@pytest.mark.parametrize("allow_remote", (False, "true", 1))
def test_ollama_remote_url_requires_explicit_boolean_opt_in(monkeypatch, allow_remote):
    monkeypatch.setattr(
        llm_nodes.urlrequest,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("Blocked remote request reached the network boundary."),
    )
    config = {
        "model_path": "qwen3:8b",
        "ollama_url": "https://ollama.example.com",
        "ollama_timeout": 12,
        "allow_remote_ollama": allow_remote,
    }

    with pytest.raises(ValueError, match="allow_remote_ollama=true"):
        llm_nodes._run_ollama_generation(
            config, "", "hello", 32, 0.7, 0.9, 1.0, -1, [], unload_after_request=False,
        )


def test_ollama_loopback_url_does_not_require_remote_opt_in(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"message": {"content": "local"}}'

    monkeypatch.setattr(llm_nodes.urlrequest, "urlopen", lambda *_args, **_kwargs: Response())
    config = {
        "model_path": "qwen3:8b",
        "ollama_url": "http://[::1]:11434",
        "ollama_timeout": 12,
    }

    response, _ = llm_nodes._run_ollama_generation(
        config, "", "hello", 32, 0.7, 0.9, 1.0, -1, [], unload_after_request=False,
    )

    assert response == "local"
