"""Browser regression check against an isolated ComfyUI serving this checkout.

uv run --no-project --with playwright python tools/verify_advanced_lora_frontend.py \
    --url http://127.0.0.1:8199
No models are loaded and no prompts are queued.
"""
import argparse
import json
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", required=True, help="Isolated ComfyUI URL; graph is cleared")
args = parser.parse_args()

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until="domcontentloaded")
    page.wait_for_function("Boolean(window.comfyAPI?.app?.app?.graph && "
                           "window.LiteGraph?.registered_node_types?.DaSiWa_LTX2LoraLoader)", timeout=90000)
    page.evaluate("window.app=window.comfyAPI.app.app")
    page.wait_for_load_state("networkidle")
    for close in page.get_by_role("button", name="Close dialog", exact=True).all():
        close.click()
    page.evaluate("""()=>{app.graph.clear(); window.n=LiteGraph.createNode('DaSiWa_LTX2LoraLoader');
        n.pos=[100,100]; app.graph.add(n);} """)
    page.wait_for_selector(".dasiwa-lora-surface")

    def settle():
        page.evaluate("()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")

    def click(x, y):
        settle()
        rect = page.locator(".dasiwa-lora-surface").bounding_box()
        dims = page.evaluate("({width:n.size[0],height:parseFloat(n._loraSurface.style.height)})")
        page.mouse.click(rect["x"] + x * rect["width"] / dims["width"],
                         rect["y"] + y * rect["height"] / dims["height"])
        settle()

    def state():
        return page.evaluate("({rows:JSON.parse(n.properties.stack_data),cache:n.properties.use_cache,"
                             "mode:n.properties.model_type,theme:n.properties.theme})")

    for modern in [False, True, False]:
        page.evaluate("mode=>app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',mode)", modern)
        page.wait_for_function("mode=>LiteGraph.vueNodesMode===mode", arg=modern)
        page.wait_for_selector(".dasiwa-lora-surface")
        settle()
        original_zoom = page.evaluate("app.canvas.ds.scale")
        for zoom in [0.75, 1, 1.5, 2]:
            page.evaluate("z=>{app.canvas.ds.scale=z;app.canvas.setDirty(true,true)}", zoom)
            settle()
            assert page.evaluate("n._loraSurface.tagName") == "DIV"
            assert page.locator(".dasiwa-lora-surface canvas").count() == 0
            assert page.locator(".dasiwa-lora-surface button").count() > 0
        page.evaluate("z=>{app.canvas.ds.scale=z;app.canvas.setDirty(true,true)}", original_zoom)
        settle()
        click(225, 48)
        assert state()["cache"] is True
        click(225, 48)
        assert state()["cache"] is False
        click(423, 48)
        assert len(state()["rows"]) == 2
        click(25, 73)
        assert state()["rows"][0]["on"] is False
        click(449, 73)
        assert state()["rows"][0]["str"] == 1.05
        click(420, 73)
        page.locator("input[type=number]").last.fill("-0.25")
        page.locator("input[type=number]").last.press("Enter")
        assert state()["rows"][0]["str"] == -0.25
        click(360, 48)
        assert state()["theme"] != "a"
        click(360, 28)
        page.locator(".litemenu-entry").filter(has_text="LTX-2.3").click()
        assert state()["mode"] == "LTX-2.3"
        click(535, 73)
        page.locator("input[type=number]").last.fill("0")
        page.locator("input[type=number]").last.press("Enter")
        assert state()["rows"][0]["as"] == 0
        click(120, 73)
        page.get_by_placeholder("Search LoRAs...").fill("None")
        page.locator(".litemenu-entry").filter(has_text="None").first.click()
        saved = page.evaluate("app.graph.serialize()")
        assert len(saved["nodes"][0]["widgets_values"]) == 3
        saved["nodes"][0].pop("widgets_values_named", None)
        page.evaluate("async saved=>{await app.loadGraphData(saved);window.n=app.graph._nodes[0]}", saved)
        page.wait_for_selector(".dasiwa-lora-surface")
        assert state()["rows"][0]["str"] == -0.25 and state()["rows"][0]["as"] == 0
        assert state()["rows"][0]["on"] is False and state()["cache"] is False
        exported = page.evaluate("async()=>await app.graphToPrompt()")
        inputs = exported["output"][str(saved["nodes"][0]["id"])]["inputs"]
        assert json.loads(inputs["stack_data"]) == state()["rows"]
        assert inputs["model_type"] == "LTX-2.3" and inputs["use_cache"] is False
        click(485, 48)
        assert all(row["lora"] == "None" and row["str"] == 1 and row["as"] == 1
                   for row in state()["rows"])
        click(440, 48)
        assert len(state()["rows"]) == 1
        page.evaluate("n.properties.theme='a';n.setDirtyCanvas(true)")
        print(f"PASS {'Nodes 2.0' if modern else 'classic'}: pointer controls, picker, reload, API export")

    # A workflow predating the optional cache widget still gets the opt-out default.
    saved = page.evaluate("app.graph.serialize()")
    saved["nodes"][0]["widgets_values"] = saved["nodes"][0]["widgets_values"][:2]
    saved["nodes"][0].pop("widgets_values_named", None)
    saved["nodes"][0]["properties"].pop("use_cache", None)
    page.evaluate("async saved=>{await app.loadGraphData(saved);window.n=app.graph._nodes[0]}", saved)
    assert state()["cache"] is False
    page.evaluate("n.setSize([900,240]);n.setDirtyCanvas(true)")
    page.wait_for_function("n._loraSurface.querySelectorAll('button').length>0")
    page.evaluate("app.graph.remove(n)")
    page.wait_for_function("!document.querySelector('.dasiwa-lora-surface')")
    page.evaluate("app.graph.add(n)")
    page.wait_for_selector(".dasiwa-lora-surface")
    assert page.evaluate("Boolean(n._loraSurface?.isConnected)")
    page.evaluate("app.graph.remove(n)")
    assert not errors, errors
    print("PASS old workflow, resize, removal/re-add; no page errors")
    browser.close()
