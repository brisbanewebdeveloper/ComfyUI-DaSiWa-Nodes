"""Verify the three Nodes 2.0 UI fixes against an isolated ComfyUI instance.

uv run --no-project --with playwright python tools/verify_nodes2_ui.py \
    --url http://127.0.0.1:8199
Uses a fresh browser context; clears its graph but never queues model execution.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", required=True, help="Isolated ComfyUI serving this checkout")
parser.add_argument("--screenshots", type=Path, help="Optional screenshot directory")
args = parser.parse_args()
if args.screenshots:
    args.screenshots.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1800, "height": 1200})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until="domcontentloaded")
    page.wait_for_function("Boolean(window.comfyAPI?.app?.app?.graph && "
                           "window.LiteGraph?.registered_node_types?.DaSiWa_LLMPromptWriter)", timeout=90000)
    page.evaluate("window.app=window.comfyAPI.app.app")
    page.wait_for_function("Boolean(app.graph._nodes.length)")
    for close in page.get_by_role("button", name="Close dialog", exact=True).all():
        close.click()

    def settle():
        page.evaluate("()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")

    def create(node_type):
        page.evaluate("""type=>{app.graph.clear();window.n=LiteGraph.createNode(type);
            n.pos=[100,100];app.graph.add(n)}""", node_type)
        settle()

    def reload():
        saved = page.evaluate("app.graph.serialize()")
        # Exercise legacy indexed restoration, not only modern named values.
        saved["nodes"][0].pop("widgets_values_named", None)
        page.evaluate("async saved=>{await app.loadGraphData(saved);window.n=app.graph._nodes[0]}", saved)
        settle()

    def inputs():
        prompt = page.evaluate("async()=>await app.graphToPrompt()")
        return prompt["output"][str(page.evaluate("n.id"))]["inputs"]

    def screenshot(name):
        if args.screenshots:
            page.screenshot(path=str(args.screenshots / f"{name}.png"))

    for modern in [False, True, False]:
        mode = "modern" if modern else "classic"
        page.evaluate("mode=>app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',mode)", modern)
        page.wait_for_function("mode=>LiteGraph.vueNodesMode===mode", arg=modern)

        create("DaSiWa_WildcardPresetPromptBuilder")
        root = page.locator(".dasiwa-wildcard-builder")
        page.wait_for_selector(".dasiwa-wildcard-selections")
        bounds = root.bounding_box()
        assert bounds["width"] > 400 and 350 < bounds["height"] < 500, bounds
        selections = page.locator(".dasiwa-wildcard-selections")
        assert selections.evaluate("e=>e.scrollHeight>e.clientHeight")
        selections.click(position={"x": 2, "y": 2})
        selections.hover()
        scale = page.evaluate("app.canvas.ds.scale")
        page.mouse.wheel(0, 400)
        page.wait_for_function("document.querySelector('.dasiwa-wildcard-selections').scrollTop>0")
        assert page.evaluate("app.canvas.ds.scale") == scale
        assert abs(root.bounding_box()["height"] - bounds["height"]) < 1
        if modern:
            page.keyboard.down("Control")
            page.mouse.wheel(0, -100)
            page.keyboard.up("Control")
            page.wait_for_function("scale=>app.canvas.ds.scale!==scale", arg=scale)
            page.evaluate("scale=>{app.canvas.ds.scale=scale;app.graph.setDirtyCanvas(true,true)}", scale)
            settle()
        selections.evaluate("e=>e.scrollTop=0")
        root.locator("details").first.locator("summary").click()
        root.locator(".dasiwa-wildcard-subject input[type=checkbox]").first.check()
        expected = page.evaluate("n.widgets.find(w=>w.name==='selection_state').value")
        assert any(row.get("enabled") for row in json.loads(expected).values())
        reload()
        page.wait_for_selector(".dasiwa-wildcard-selections")
        assert inputs()["selection_state"] == expected
        if modern:
            page.evaluate("n.setSize([760,950])")
            settle()
            assert abs(root.evaluate("e=>e.clientHeight") - 490) < 1
        screenshot(f"wildcard-{mode}")
        print(f"PASS {mode}: wildcard bounds, real wheel scrolling, selection and array-only reload")

        create("DaSiWa_LLMPromptWriter")
        areas = page.locator("textarea:visible")
        expect(areas).to_have_count(2)
        initial = [areas.nth(i).bounding_box() for i in range(2)]
        assert initial[1]["height"] > initial[0]["height"] * 1.5, initial
        areas.nth(0).fill("quality tags, trigger word")
        areas.nth(1).fill("Line one\nLine two")
        areas.nth(1).press("Tab")
        settle()
        assert inputs()["start_with"] == "quality tags, trigger word"
        assert inputs()["idea"] == "Line one\nLine two"
        reload()
        areas = page.locator("textarea:visible")
        assert areas.nth(0).input_value() == "quality tags, trigger word"
        assert areas.nth(1).input_value() == "Line one\nLine two"
        before = [areas.nth(i).bounding_box()["height"] for i in range(2)]
        page.evaluate("n.setSize([500,n.size[1]+240]);n.setDirtyCanvas(true,true)")
        settle()
        after = [areas.nth(i).bounding_box()["height"] for i in range(2)]
        assert abs(after[0] - before[0]) < 2, (before, after)
        assert after[1] > before[1] + 100, (before, after)
        screenshot(f"writer-{mode}")
        print(f"PASS {mode}: writer geometry, typing, API export, array-only reload and resize")

        create("DaSiWa_EnhancedVideoCombine")
        help_button = page.locator(".dasiwa-enhanced-video-preview").get_by_role("button", name="Help", exact=True)
        help_button.click()
        dialog = page.locator("dialog[open]")
        assert "Enhanced Video Combine Help" in dialog.inner_text()
        assert "audio" in dialog.inner_text() and "preview" in dialog.inner_text()
        screenshot(f"help-{mode}")
        dialog.get_by_title("Close", exact=True).click()
        page.wait_for_function("!document.querySelector('dialog[open]')")
        reload()
        page.locator(".dasiwa-enhanced-video-preview").get_by_role("button", name="Help", exact=True).click()
        page.locator("dialog[open]").get_by_title("Close", exact=True).click()
        values = inputs()
        assert isinstance(values["save_first_frame"], bool)
        assert isinstance(values["save_last_frame"], bool)
        print(f"PASS {mode}: help open/close and reload; existing frame-export inputs preserved")

    page.evaluate("app.graph.clear()")
    settle()
    assert not errors, errors
    print("PASS renderer switching and removal; no page errors")
    browser.close()
