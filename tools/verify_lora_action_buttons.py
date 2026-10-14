"""Real-pointer checks for rounded LoRA actions; isolated ComfyUI only, no queue.

uv run --no-project --with playwright python tools/verify_lora_action_buttons.py --url http://127.0.0.1:8199
Metadata responses are mocked; this tests UI behavior, not Civitai integration.
"""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', required=True)
parser.add_argument('--screenshot-dir', type=Path)
args = parser.parse_args()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 1000}, device_scale_factor=2)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    requests = []

    def info(route):
        requests.append(route.request.url)
        route.fulfill(json={'file': 'probe.safetensors', 'civitaiFound': False,
                            'civitaiError': 'UI test', 'trainedWords': [], 'images': []})

    page.route('**/dasiwa/ltx2/lorainfo?*', info)
    page.route('**/dasiwa/ltx2/keycounts?*', lambda r: r.fulfill(json={'v': 1, 'a': 1}))
    page.goto(args.url, wait_until='domcontentloaded')
    page.wait_for_function("Boolean(window.comfyAPI?.app?.app?.graph && window.LiteGraph?.registered_node_types?.DaSiWa_LTX2LoraLoader)", timeout=90000)
    page.evaluate('window.app=window.comfyAPI.app.app')
    page.wait_for_load_state('networkidle')
    for close in page.get_by_role('button', name='Close dialog', exact=True).all():
        close.click()
    page.evaluate("app.graph.clear();window.n=LiteGraph.createNode('DaSiWa_LTX2LoraLoader');n.pos=[100,100];app.graph.add(n)")

    def settle():
        page.evaluate('()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))')

    def move(fraction, row=0):
        settle()
        rect = page.locator('.dasiwa-lora-surface').bounding_box()
        height = page.evaluate('61 + JSON.parse(n.properties.stack_data).length * 28')
        page.mouse.move(rect['x'] + fraction * rect['width'],
                        rect['y'] + (73 + row * 28) * rect['height'] / height)
        settle()

    def pixels():
        return page.locator('.dasiwa-lora-surface').screenshot(animations='disabled')

    for modern in [False, True, False]:
        page.evaluate("v=>app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',v)", modern)
        page.wait_for_function('v=>LiteGraph.vueNodesMode===v', arg=modern)
        page.wait_for_selector('.dasiwa-lora-surface')
        for width in [400, 720, 1000]:
            page.evaluate('w=>{n.setSize([w,180]);n.setDirtyCanvas(true)}', width)
            for theme in ['a', 'b', 'c', 'd', 'e', 'f']:
                page.evaluate("t=>{n.properties.theme=t;n.properties.model_type='Basic';n.properties.stack_data=JSON.stringify([{on:true,lora:'probe.safetensors',str:0.5,vs:0.75,as:0.25},{on:true,lora:'None',str:1,vs:1,as:1}]);n.setDirtyCanvas(true)}", theme)
                move(.80)
                normal = pixels()
                for fraction, action, tooltip in [(.947, 'info', 'Show LoRA info'), (.979, 'trash', 'does not delete')]:
                    move(fraction)
                    assert page.evaluate('n._loraHoveredAction') == '0:' + action
                    assert tooltip in page.locator('.dasiwa-lora-surface').get_attribute('title')
                    assert page.locator('.dasiwa-lora-surface').evaluate('(e)=>e.style.cursor') == 'pointer'
                    assert pixels() != normal, 'Hover must visibly change the canvas'
                    move(.80)
                    assert page.evaluate('n._loraHoveredAction') is None
                    assert pixels() == normal, 'Hover must reset on moving away'
                move(.947)
                page.mouse.down()
                page.mouse.up()
                page.wait_for_selector('[data-action="close"]')
                page.locator('[data-action="close"]').click()
                move(.979)
                page.mouse.down()
                page.mouse.up()
                row = page.evaluate('JSON.parse(n.properties.stack_data)[0]')
                assert row == {'on': True, 'lora': 'None', 'str': 0.5, 'vs': 0.75, 'as': 0.25}
                for fraction in [.947, .979]:
                    move(fraction, 1)
                    assert page.evaluate('n._loraHoveredAction') is None
                    assert 'Select a LoRA' in page.locator('.dasiwa-lora-surface').get_attribute('title')
                    assert page.locator('.dasiwa-lora-surface').evaluate('(e)=>e.style.cursor') == 'not-allowed'
                    before = len(requests)
                    page.mouse.down()
                    page.mouse.up()
                    settle()
                    assert len(requests) == before
                move(.947, 1)
                page.mouse.move(20, 600)
                settle()
                assert page.evaluate('n._loraHoveredAction') is None
                assert page.locator('.dasiwa-lora-surface').get_attribute('title') == ''
            print(f'PASS modern={modern}, width={width}: six themes, hover pixels/reset, tooltips, info/clear clicks, disabled slots')
        if args.screenshot_dir:
            args.screenshot_dir.mkdir(parents=True, exist_ok=True)
            page.evaluate("n.setSize([1000,180]);n.properties.theme='a';n.properties.stack_data=JSON.stringify([{on:true,lora:'Example LoRA.safetensors',str:1,vs:1,as:1},{on:true,lora:'None',str:1,vs:1,as:1}]);n.setDirtyCanvas(true)")
            move(.947)
            page.locator('.dasiwa-lora-surface').screenshot(path=str(args.screenshot_dir / f'buttons-{modern}.png'))
    assert not errors, errors
    print('PASS no page errors; no prompts queued')
    browser.close()
