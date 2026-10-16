"""Real Director dialog/queue smoke against an isolated CPU ComfyUI only."""
import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('url')
parser.add_argument('--input', type=Path, nargs=2, required=True)
parser.add_argument('--screenshots', type=Path)
args = parser.parse_args()
url = args.url.rstrip('/')
info = json.load(urlopen(url + '/object_info/DaSiWaH3ClipAssemble'))
assert info['DaSiWaH3ClipAssemble']['python_module'] == 'custom_nodes.dasiwa_segment_verify', 'Use the isolated verification instance.'
if args.screenshots:
    args.screenshots.mkdir(parents=True, exist_ok=True)
with sync_playwright() as playwright:
    options = dict(headless=True, args=['--no-sandbox'])
    if os.environ.get('H3_TEST_CHROME'):
        options['executable_path'] = os.environ['H3_TEST_CHROME']
    browser = playwright.chromium.launch(**options)
    page = browser.new_page(viewport=dict(width=1800, height=1400))
    errors, queued = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: queued.append(request.post_data_json) if request.method == 'POST' and request.url.split('?')[0].endswith('/prompt') else None)
    page.goto(url, wait_until='networkidle')
    page.wait_for_function("Boolean(window.LiteGraph?.registered_node_types.MiniMaxH3Director)")
    page.evaluate("async()=>{window.testApp=(await import('/scripts/app.js')).app}")
    results = []
    for modern in (False, True):
        page.evaluate("async modern=>{await testApp.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',modern);testApp.graph.clear();window.n=LiteGraph.createNode('MiniMaxH3Director');n.pos=[50,50];testApp.graph.add(n)}", modern)
        button = page.get_by_role('button', name='Join clips…', exact=True)
        button.wait_for(state='visible')
        button.evaluate('(b)=>b.click()')
        dialog = page.get_by_role('dialog', name='Join clips', exact=True)
        expect(dialog).to_be_visible()
        assert len(queued) == len(results), 'Opening a dialog must not queue work.'
        dialog.get_by_role('tab', name='Video files', exact=True).click()
        dialog.locator('input[type=file]').set_input_files([str(p.resolve()) for p in args.input])
        expect(dialog.locator('.clip-row')).to_have_count(2, timeout=15000)
        expect(dialog.locator('.clip-time').first).to_contain_text('Source duration: 1.000 s', timeout=15000)
        expect(dialog.locator('.clip-time').last).to_contain_text('Source duration: 1.000 s', timeout=15000)
        expect(dialog.locator('.clip-head')).to_contain_text('Start (s)')
        expect(dialog.locator('.clip-head')).to_contain_text('End (s)')
        expect(dialog.get_by_text('Start (s)', exact=True)).to_have_count(1)
        end_input=dialog.get_by_role('spinbutton', name='End (s)', exact=False).first
        expect(end_input).to_have_value('1.000')  # prefilled full length; saved value stays "file end"
        box=[dialog.get_by_role('spinbutton', name='Start (s)', exact=False).nth(i).bounding_box()['x'] for i in range(2)]
        assert abs(box[0]-box[1])<1, box
        start_input=dialog.get_by_role('spinbutton', name='Start (s)', exact=False).first
        start_input.fill('0.25');end_input.focus()
        expect(dialog.locator('.clip-time').first).to_contain_text('Selected: 0.750 s')
        start_input.fill('0');end_input.focus()
        dialog.get_by_role('button', name='Remove ', exact=False).last.click()
        expect(dialog.locator('.clip-row')).to_have_count(1)
        dialog.locator('input[type=file]').set_input_files(str(args.input[1].resolve()))
        expect(dialog.locator('.clip-row')).to_have_count(2, timeout=15000)
        expect(dialog.locator('.clip-time').last).to_contain_text('Source duration: 1.000 s', timeout=15000)
        saved = page.evaluate("()=>{const w=n.widgets.find(w=>w.name==='timeline_data');return {state:JSON.parse(w.value),graph:testApp.graph.serialize()}}")
        assert len(saved['state']['clip_assembly']['rows']) == 2
        assert saved['state']['continuity']['output_mode'] == 'full'
        assert saved['state']['items'] == [], 'Uploads must not become REF2VA references.'
        if args.screenshots and not modern:
            dialog.screenshot(path=str(args.screenshots / 'DaSiWa-H3-Join-Clips.png'))
        dialog.get_by_role('button', name='Join only — no AI', exact=True).click()
        confirm = dialog.get_by_role('button', name='Re-encode and join', exact=True)
        expect(confirm).to_be_visible(timeout=15000)
        assert len(queued) == len(results), 'Preflight must not submit the job.'
        confirm.click()
        expect(dialog.locator('.clip-status')).to_contain_text('H3_Joined', timeout=90000)
        assert len(queued) == len(results) + 1
        prompt = queued[-1]['prompt']
        assert len(prompt) == 1 and next(iter(prompt.values()))['class_type'] == 'DaSiWaH3ClipAssemble'
        dialog.get_by_role('button', name='Close', exact=True).click()
        expect(dialog).to_have_count(0)
        page.evaluate("async graph=>{await testApp.loadGraphData(graph);window.n=testApp.graph._nodes.find(n=>n.comfyClass==='MiniMaxH3Director')}", saved['graph'])
        state = page.evaluate("JSON.parse(n.widgets.find(w=>w.name==='timeline_data').value)")
        assert state['clip_assembly'] == saved['state']['clip_assembly']
        assert state['continuity']['source_id'] == saved['state']['continuity']['source_id']
        results.append(dict(modern=modern, roundtrip='passed', isolated_model_free_queue='passed'))
    page.evaluate("""async()=>{
        const a=testApp;a.graph.clear();
        const types=['MiniMaxH3Director','MiniMaxH3DirectorGuide','DaSiWaH3ContinuitySegmentPrepare','DaSiWaH3TiledUpscale','DaSiWaH3ContinuitySegmentDecode','DaSiWa_EnhancedVideoCombine','DaSiWaH3ContinuityPublish'];
        const [d,g,p,u,dec,exp,pub]=types.map(t=>{const n=LiteGraph.createNode(t);a.graph.add(n);return n});
        const wire=(from,name,to,input)=>from.connect(from.outputs.findIndex(x=>x.name===name),to,to.inputs.findIndex(x=>x.name===input));
        wire(d,'guide',g,'guide');wire(g,'continuity_context',p,'context');wire(p,'render_latent',u,'latent');wire(p,'segment_context',u,'continuity_context');wire(u,'latent',dec,'latent');wire(p,'segment_context',dec,'context');wire(dec,'images',exp,'images');wire(exp,'filename',pub,'filename');wire(p,'ticket',pub,'ticket');
        d.__dasiwaH3State().continuity.output_mode='segment';
        const timeline=d.widgets.find(w=>w.name==='timeline_data');
        timeline.serializeValue();u.mode=4;timeline.serializeValue();
    }""")
    assert errors == [], errors
    print(json.dumps(dict(cases=results, bypassed_optional_upscale='passed', uncaught_browser_errors=errors), indent=2))
    browser.close()
