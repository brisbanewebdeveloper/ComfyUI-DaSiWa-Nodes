import importlib.util
import json
import sys
import types
from pathlib import Path

# Load the repo module by absolute file path under a synthetic package: a live
# ComfyUI checkout puts its own top-level `nodes` on sys.path and would shadow
# `nodes/nodes_*.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_NODES_DIR = _REPO_ROOT / "nodes"
_PKG = "dasiwa_seed_control_nodes_pkg"


def _load_repo_module(name):
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


def _module():
    return _load_repo_module("nodes_seed_control")


def _package_source():
    return (_REPO_ROOT / "__init__.py").read_text(encoding="utf-8")


def test_schema_exposes_hidden_local_seed_and_state_widgets():
    module = _module()
    required = module.DaSiWa_SeedControl.INPUT_TYPES()["required"]

    assert required["seed_value"][0] == "INT"
    assert required["seed_value"][1]["min"] == 0
    assert required["seed_value"][1]["max"] == 0xFFFFFFFFFFFFFFFF
    assert required["seed_value"][1]["hidden"] is True
    assert required["seed_control_state"][0] == "STRING"
    assert required["seed_control_state"][1]["hidden"] is True
    assert module.DaSiWa_SeedControl.CATEGORY == "DaSiWa"
    assert module.DaSiWa_SeedControl.RETURN_TYPES == ("INT", "NOISE")
    assert module.DaSiWa_SeedControl.RETURN_NAMES == ("seed", "noise")


def test_external_seed_is_a_force_input_int_in_the_full_64_bit_range():
    module = _module()
    optional = module.DaSiWa_SeedControl.INPUT_TYPES()["optional"]

    assert optional["seed"][0] == "INT"
    assert optional["seed"][1]["forceInput"] is True
    assert optional["seed"][1]["min"] == 0
    assert optional["seed"][1]["max"] == 0xFFFFFFFFFFFFFFFF


def test_external_seed_passes_through_when_linked():
    module = _module()
    node = module.DaSiWa_SeedControl()

    seed, noise = node.execute(0, json.dumps(module.DEFAULT_STATE), seed=12345)

    assert seed == 12345
    assert noise.seed == 12345


def test_local_seed_is_emitted_when_no_external_seed_is_linked():
    module = _module()
    node = module.DaSiWa_SeedControl()

    seed, _ = node.execute(987654, json.dumps({"mode": "fixed", "last_seed": "42", "recent": ["42", "1"]}))

    assert seed == 987654


def test_random_mode_without_local_value_rolls_a_fresh_seed_on_execute():
    module = _module()
    node = module.DaSiWa_SeedControl()

    first, _ = node.execute(0, json.dumps(module.DEFAULT_STATE))
    second, _ = node.execute(0, json.dumps(module.DEFAULT_STATE))

    assert 0 <= first <= 0xFFFFFFFFFFFFFFFF
    assert 0 <= second <= 0xFFFFFFFFFFFFFFFF
    assert first != second


def test_fixed_mode_without_local_value_keeps_zero_instead_of_rolling():
    module = _module()
    node = module.DaSiWa_SeedControl()

    seed, noise = node.execute(0, json.dumps({"mode": "fixed", "last_seed": None, "recent": []}))

    assert seed == 0
    assert noise.seed == 0


def test_random_mode_with_a_local_value_keeps_that_value():
    module = _module()
    node = module.DaSiWa_SeedControl()

    seed, _ = node.execute(777, json.dumps({"mode": "random", "last_seed": "777", "recent": ["777"]}))

    assert seed == 777


def test_noise_output_is_compatible_and_wraps_the_effective_seed():
    module = _module()
    node = module.DaSiWa_SeedControl()

    # External link wins; the NOISE object must wrap that same int.
    _, noise = node.execute(0, json.dumps(module.DEFAULT_STATE), seed=4242)
    assert noise.seed == 4242
    assert callable(getattr(noise, "generate_noise", None))

    # Local value is wrapped when no external seed is linked.
    _, noise_local = node.execute(999, json.dumps(module.DEFAULT_STATE))
    assert noise_local.seed == 999


def test_out_of_range_seeds_raise():
    module = _module()
    node = module.DaSiWa_SeedControl()
    import pytest

    with pytest.raises(ValueError):
        node.execute(-1, json.dumps(module.DEFAULT_STATE))
    with pytest.raises(ValueError):
        node.execute(0xFFFFFFFFFFFFFFFF + 1, json.dumps(module.DEFAULT_STATE))


def test_state_normalization_cleans_garbage_payloads():
    module = _module()

    state = module._normalize_state(json.dumps({"mode": "bogus", "last_seed": "abc", "recent": ["1", "2", 3, "x", None] + ["4"] * 20}))

    assert state["mode"] == "random"
    assert state["last_seed"] is None
    assert state["recent"] == ["1", "2", "3", "4", "4", "4", "4", "4", "4", "4"]  # capped at 10
    # Invalid JSON and non-dict payloads fall back to the defaults.
    assert module._normalize_state("not json")["mode"] == "random"
    assert module._normalize_state({"mode": "fixed", "last_seed": 42, "recent": []})["last_seed"] == "42"
    assert module._normalize_state(None) == module.DEFAULT_STATE


def test_is_changed_is_stable_outside_random_zero_mode():
    module = _module()
    cls = module.DaSiWa_SeedControl

    assert cls.IS_CHANGED(0, json.dumps(module.DEFAULT_STATE), seed=1) is False
    assert cls.IS_CHANGED(42, json.dumps(module.DEFAULT_STATE)) is False
    assert cls.IS_CHANGED(0, json.dumps({"mode": "fixed", "last_seed": None, "recent": []})) is False
    # Random mode without a local value must re-execute every queue.
    assert cls.IS_CHANGED(0, json.dumps(module.DEFAULT_STATE)) is not False


def test_package_registers_the_node_and_display_name():
    source = _package_source()

    assert "from .nodes.nodes_seed_control import DaSiWa_SeedControl" in source
    assert '"DaSiWa_SeedControl": DaSiWa_SeedControl' in source
    assert '"DaSiWa_SeedControl": "Seed Control"' in source
