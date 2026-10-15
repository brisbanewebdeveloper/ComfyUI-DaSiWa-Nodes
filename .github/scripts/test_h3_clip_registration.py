"""Additive registration and guarded assembly HTTP boundary contracts."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def mapping_keys(name):
    tree = ast.parse((ROOT / '__init__.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets) and isinstance(node.value, ast.Dict) and node.value.keys:
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError(name)


def test_new_segment_and_assembly_nodes_registered_additively():
    required = {'DaSiWaH3ContinuitySegmentPrepare', 'DaSiWaH3ContinuitySegmentDecode', 'DaSiWaH3ClipAssemble'}
    legacy = {'MiniMaxH3Director', 'MiniMaxH3DirectorGuide', 'DaSiWaH3ContinuityAppend', 'DaSiWaH3ContinuityPublish', 'DaSiWaH3TiledUpscale'}
    assert required | legacy <= mapping_keys('NODE_CLASS_MAPPINGS')
    assert required | legacy <= mapping_keys('NODE_DISPLAY_NAME_MAPPINGS')


def test_preflight_is_bounded_and_does_not_run_assembly():
    tree = ast.parse((ROOT / 'nodes/h3_continuity/routes.py').read_text())
    route = next((n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'assembly_preflight'), None)
    assert route is not None
    strings = {n.value for n in ast.walk(route) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert '/df_h3_continuity/assembly/preflight' in strings
    calls = [n for n in ast.walk(route) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == 'read' for n in calls)
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == 'to_thread' for n in calls)
    assert not any(isinstance(n.func, ast.Name) and n.func.id == 'assemble' for n in calls)
