"""Exercise the real route handler with bounded chunked requests; no Core/GPU."""
import ast
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]


class Content:
    def __init__(self, payload):
        self.payload = payload

    async def read(self, count):
        chunk = self.payload[:min(count, 7)]
        self.payload = self.payload[len(chunk):]
        return chunk


def call_handler(payload, content_length=None, preflight=None):
    module_name = '_h3_assembly_http_test.assembly'
    fake = ModuleType(module_name)
    calls = []
    def check(data):
        calls.append(data)
        return preflight(data) if preflight else {'ok': True, 'seconds': 1}
    fake.preflight = check
    old = sys.modules.get(module_name)
    sys.modules[module_name] = fake
    try:
        tree = ast.parse((ROOT / 'nodes/h3_continuity/routes.py').read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'assembly_preflight')
        namespace = {'__package__': '_h3_assembly_http_test', 'web': web, 'asyncio': asyncio,
                     'routes': SimpleNamespace(post=lambda _: lambda fn: fn)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / 'nodes/h3_continuity/routes.py'), 'exec'), namespace)
        request = SimpleNamespace(content_length=content_length, content=Content(payload))
        response = asyncio.run(namespace['assembly_preflight'](request))
        return response.status, json.loads(response.body), calls
    finally:
        if old is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old


def test_valid_chunked_request_calls_read_only_preflight_once():
    status, result, calls = call_handler(b'{"version":1,"rows":[]}')
    assert status == 200 and result['ok'] is True
    assert calls == [{'version': 1, 'rows': []}]


@pytest.mark.parametrize('payload,length', [(b'x' * 65537, None), (b'{}', 65537)])
def test_oversized_requests_never_call_media_probe(payload, length):
    status, _, calls = call_handler(payload, length)
    assert status == 413 and not calls


@pytest.mark.parametrize('payload', [b'not-json', b'[]', b'null', b'\xff'])
def test_malformed_requests_are_client_errors(payload):
    status, result, calls = call_handler(payload)
    assert status == 400 and result['ok'] is False and not calls


def test_preflight_rejection_is_not_http_success():
    status, result, _ = call_handler(b'{}', preflight=lambda _: {'ok': False, 'issues': ['incompatible clips']})
    assert status == 400 and result['issues'] == ['incompatible clips']
