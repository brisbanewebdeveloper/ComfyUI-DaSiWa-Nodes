"""The build route resolves wildcard rules exactly like the node, outside the graph."""
import asyncio
import importlib
import sys
import types

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def test_build_route_matches_node(monkeypatch):
    routes = web.RouteTableDef()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))))
    module = importlib.reload(importlib.import_module("nodes.nodes_wildcard_preset_prompt_builder"))
    library = {"Art": [{"subject": "Light", "booru_wildcards": ["{red|blue} light"]}],
               "Negative Prompt": [{"subject": "Bad", "booru_presets": ["blurry"]}]}
    monkeypatch.setattr(module, "_load_library", lambda: library)
    rules = {"style": "Booru", "seed": 7, "reroll": 3, "token_budget": 200,
             "selection_state": {"Art/Light/wildcards": {"enabled": True, "weight": 1.2},
                                 "Negative Prompt/Bad/presets": {"enabled": True, "weight": 1}}}
    expected = module.build_prompts(library, "Booru", 7, 3, 200, rules["selection_state"])

    async def exercise():
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/dasiwa/wildcard-preset-prompt-builder/build", json=rules)
            assert response.status == 200
            payload = await response.json()
            assert (payload["positive"], payload["negative"]) == expected[:2]
            assert payload["positive"] in ("(red light:1.2)", "(blue light:1.2)")
            assert payload["negative"] == "blurry"
            bad = await client.post("/dasiwa/wildcard-preset-prompt-builder/build", json=[])
            assert bad.status == 400

    asyncio.run(exercise())
