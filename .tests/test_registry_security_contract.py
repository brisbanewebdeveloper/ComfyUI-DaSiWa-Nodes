import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
MODULE_PATH = REPO_ROOT / "nodes" / "nodes_llm.py"
HELPER_PATH = REPO_ROOT / "nodes" / "helper_logging.py"


def _module(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = "/tmp/comfy-models"
    folder_paths.add_model_folder_path = lambda *_: None
    folder_paths.get_folder_paths = lambda _: [folder_paths.models_dir]
    folder_paths.get_full_path = lambda *_: None
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    helper_spec = importlib.util.spec_from_file_location("helper_logging", HELPER_PATH)
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    monkeypatch.setitem(sys.modules, "helper_logging", helper)
    helper_spec.loader.exec_module(helper)

    spec = importlib.util.spec_from_file_location("registry_security_llm_nodes", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformers_load_uses_remote_code_disabled(monkeypatch):
    module = _module(monkeypatch)
    calls = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            calls.append(kwargs)
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            calls.append(kwargs)
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = Tokenizer
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForCausalLM = Model
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    module._load_transformers_model({
        "model_path": "/models/local",
        "task": "text",
        "device": "cpu",
        "dtype": "auto",
        "quantization": "none",
        "cache_mode": "unload_after_run",
        "attention_implementation": "auto",
    }, need_vision=False)

    assert calls
    assert all(call["trust_remote_code"] is False for call in calls)
