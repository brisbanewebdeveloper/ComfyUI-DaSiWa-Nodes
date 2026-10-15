"""Saved-workflow ABI from the clean pre-segment revision, without GPU imports."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / 'tests/fixtures/h3_legacy_node_contract.json').read_text())
KEYS = {'RETURN_TYPES', 'RETURN_NAMES', 'FUNCTION', 'CATEGORY', 'OUTPUT_NODE', 'OUTPUT_IS_LIST'}


def canonical(node):
    if isinstance(node, ast.AST):
        return {"_type": type(node).__name__, **{
            name: canonical(value) for name, value in ast.iter_fields(node)
            if value is not None and value != []
        }}
    if isinstance(node, list):
        return [canonical(value) for value in node]
    return node


def test_existing_h3_node_contracts_are_unchanged():
    for filename, expected_classes in CONTRACT.items():
        classes = {}
        for cls in ast.parse((ROOT / filename).read_text()).body:
            if not isinstance(cls, ast.ClassDef) or cls.name not in expected_classes:
                continue
            contract = {}
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'INPUT_TYPES':
                    contract[node.name] = canonical(node)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in KEYS:
                            contract[target.id] = canonical(node.value)
            classes[cls.name] = contract
        assert classes == expected_classes, filename
