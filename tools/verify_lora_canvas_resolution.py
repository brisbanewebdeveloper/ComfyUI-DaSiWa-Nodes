"""Check the real LoRA DOM surface in an isolated browser, without ComfyUI/models.

uv run --no-project --with playwright python tools/verify_lora_canvas_resolution.py
Only the ComfyUI node host is stubbed; DOM, layout and pointer events are real.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "js/advanced_lora_loader_ui.js").read_text().split("\n", 1)[1]
HOST = """
window.app = {registerExtension(e) {window.extension = e}, canvas: {}};
class TestNode {
  constructor() {this.widgets = []; this.properties = {}; this.flags = {}; this.size = [720, 89];}
  addWidget(type, name, value, callback, options) {
    const w = {type, name, value, callback, options}; this.widgets.push(w); return w;
  }
  addDOMWidget(name, type, root, options) {
    this.root = root; document.querySelector('#host').append(root);
    const w = {name, type, options}; this.widgets.push(w); return w;
  }
  setDirtyCanvas() {}
}
window.TestNode = TestNode;
"""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for dpr in [1, 1.25, 2]:
        page = browser.new_page(viewport={"width": 1800, "height": 800}, device_scale_factor=dpr)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content('<style>#host{width:720px;transform-origin:0 0}</style><div id="host"></div>')
        page.add_script_tag(content=HOST)
        page.add_script_tag(content=SOURCE)
        page.evaluate("""async () => {
          await extension.beforeRegisterNodeDef(TestNode, {name:'DaSiWa_LTX2LoraLoader'}, app);
          window.n = new TestNode(); n.onNodeCreated();
          window.draws = 0;
          const draw = n.drawLoraStack;
          n.drawLoraStack = function(ctx) {draws++; return draw.call(this, ctx)};
        }""")

        def settle():
            page.evaluate("()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")

        def click(x, y):
            rect = page.locator(".dasiwa-lora-surface").bounding_box()
            page.mouse.click(rect["x"] + x * rect["width"] / 720,
                             rect["y"] + y * rect["height"] / 89)
            settle()

        for zoom in [1, 1.5, 2, 0.75]:
            # DOM text and SVG remain native at every CSS zoom, without repaint polling.
            page.evaluate("z=>document.querySelector('#host').style.transform=`scale(${z})`", zoom)
            settle()
            assert page.evaluate("n._loraSurface.tagName") == "DIV"
            assert page.locator(".dasiwa-lora-surface canvas").count() == 0
            assert page.locator(".dasiwa-lora-surface svg").count() == 2
            assert page.get_by_role("button", name="MODEL: Basic ▼", exact=True).is_visible()
            before = page.evaluate("draws")
            settle()
            assert page.evaluate("draws") == before, "Unchanged surface should not redraw"
            click(25, 73)
            assert page.evaluate("JSON.parse(n.properties.stack_data)[0].on") is False
            click(25, 73)
            click(420, 73)
            rect = page.locator(".dasiwa-lora-surface").bounding_box()
            editor = page.locator("input[type=number]").evaluate("e=>e.parentElement.parentElement.getBoundingClientRect().toJSON()")
            assert abs(editor["x"] - (rect["x"] + (628 * 0.72 + 6) * zoom)) < 1
            assert abs(editor["y"] - (rect["y"] + 61 * zoom)) < 1
            page.locator("input[type=number]").fill("-0.25")
            page.locator("input[type=number]").press("Enter")
            assert page.evaluate("JSON.parse(n.properties.stack_data)[0].str") == -0.25

        page.get_by_role("button", name="ALL✓", exact=True).press("Enter")
        settle()
        assert page.evaluate("JSON.parse(n.properties.stack_data)[0].on") is False
        assert page.evaluate("n._loraSurface.contains(document.activeElement)")
        page.evaluate("document.querySelector('#host').style.width='640px'")
        settle()
        assert page.locator('[data-action=trash]').last.evaluate("e=>e.getBoundingClientRect().right<=n._loraSurface.getBoundingClientRect().right")
        page.evaluate("document.querySelector('#host').style.width='720px'")
        settle()
        page.evaluate("n.properties.stack_data=JSON.stringify([{on:true,lora:'<img src=x onerror=alert(1)>.safetensors',str:1,vs:1,as:1}]);n.setDirtyCanvas(true)")
        settle()
        assert page.locator(".dasiwa-lora-surface img").count() == 0
        assert '<img src=x' in page.locator(".dasiwa-lora-surface").inner_text()
        cdp = page.context.new_cdp_session(page)
        for scale_factor in [2.5, dpr]:
            cdp.send("Emulation.setDeviceMetricsOverride", {
                "width": 1800, "height": 800, "deviceScaleFactor": scale_factor, "mobile": False,
            })
            settle()
            assert page.evaluate("n._loraSurface.getBoundingClientRect().width") == 720 * 0.75
        page.evaluate("n.size[0]=900;document.querySelector('#host').style.width='900px';n.setDirtyCanvas(true)")
        settle()
        assert page.evaluate("n._loraSurface.getBoundingClientRect().width") == 900 * 0.75
        page.evaluate("n.onRemoved();n.root.remove()")
        settle()
        before = page.evaluate("draws")
        settle()
        assert page.evaluate("draws") == before
        page.evaluate("document.querySelector('#host').append(n.root);n.onAdded()")
        settle()
        assert page.evaluate("draws") > before
        assert not errors, errors
        print(f"PASS DPR {dpr}: zoom, no idle redraw, hit tests, editor, resize, removal/re-add")
        page.close()
    browser.close()
