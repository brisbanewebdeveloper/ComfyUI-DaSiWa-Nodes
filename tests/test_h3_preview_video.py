"""Real-video regression test for Director preview.

Run: uv run --no-project --with playwright python tests/test_h3_preview_video.py [screenshot.png]
Requires ffmpeg for the generated, silent VP8/WebM fixture.
"""
import base64
from pathlib import Path
import subprocess
import sys
import tempfile


def run_video_tests():
    from playwright.sync_api import sync_playwright

    root = Path(__file__).resolve().parents[1]
    source = (root / 'js/minimax_h3_director.js').read_text()
    preview = source[source.index('  let previewOverlay = null;'):source.index('  const syncAttachedPrompts =')]
    styles = source[source.index('function installStyles()'):source.index('function parseState(')]
    with tempfile.TemporaryDirectory(prefix='director-video-') as temp:
        fixture = Path(temp) / 'preview.webm'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', 'testsrc2=size=320x180:rate=24:duration=8', '-an',
                        '-c:v', 'libvpx', '-deadline', 'realtime', '-b:v', '250k', str(fixture)], check=True)
        video_bytes = fixture.read_bytes()
    video_url = 'data:video/webm;base64,' + base64.b64encode(video_bytes).decode()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 900, 'height': 800})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_content('<html><body></body></html>')
        page.evaluate("code => window.eval('let cssInstalled = false;\\n' + code + '\\ninstallStyles();')", styles)
        page.evaluate("""([code,url]) => {
            window.viewUrl = () => url;
            window.isLockedSlot = () => false; window.setStatus = () => {}; window.mutate = () => {};
            window.waveformFetches = 0; window.realFetch = fetch;
            window.fetch = (...args) => { waveformFetches++; return realFetch(...args); };
            window.eval(code + '\\nwindow.openPreview = openPreview; window.closePreview = closePreview;');
            openPreview({type:'video', value:'preview.webm', source_duration:8, trim_start:3, trim_end:5});
            window.media = document.querySelector('video');
            window.playingTimes = [];
            media.addEventListener('playing', () => playingTimes.push(media.currentTime));
            document.querySelector('.ds-h3-play-crop-btn').click();
        }""", [preview, video_url])
        assert page.locator('.ds-h3-preview-playhead').count() == 1, 'Video preview has no playback cursor'
        page.wait_for_function('playingTimes.length > 0')
        assert abs(page.evaluate('playingTimes[0]') - 3) < 0.15
        page.wait_for_function('!media.paused && media.currentTime > 3.3')
        cursor = page.locator('.ds-h3-preview-playhead').evaluate('el => parseFloat(el.style.left)')
        assert 37.5 < cursor < 62.5
        page.wait_for_function('media.paused && media.currentTime >= 5', timeout=6000)
        assert abs(page.evaluate('media.currentTime') - 5) < 0.15
        page.wait_for_function("Math.abs(parseFloat(document.querySelector('.ds-h3-preview-playhead').style.left) - 62.5) < 1")
        assert page.evaluate('media.videoWidth === 320 && media.videoHeight === 180')
        assert page.evaluate('waveformFetches') == 0, 'Silent video must not trigger audio-waveform decoding'
        print('PASS real video crop 3s–5s, live cursor, decoded frames and no audio decoding')
        grip = page.locator('.ds-h3-preview-playhead-grip')
        track = grip.locator('../..')
        rect = track.bounding_box()
        grip.hover(); page.mouse.down()
        page.mouse.move(rect['x'] + rect['width'] * 0.5, rect['y'] + 7, steps=5)
        page.mouse.up()
        page.wait_for_function('Math.abs(media.currentTime - 4) < 0.1')
        assert page.locator('.ds-h3-ts-input').input_value() == '3.00'
        assert page.locator('.ds-h3-te-input').input_value() == '5.00'
        page.evaluate('window.selectedStart = null; media.addEventListener("playing", () => selectedStart = media.currentTime, {once:true}); media.play()')
        page.wait_for_function('selectedStart !== null')
        assert abs(page.evaluate('selectedStart') - 4) < 0.15
        page.evaluate('media.pause()')
        grip.focus(); page.keyboard.press('ArrowLeft')
        page.wait_for_function('Math.abs(media.currentTime - 3) < 0.1')
        print('PASS video cursor drag/keyboard, unchanged crop and playback from chosen point')
        # Drag crop independently, below the cursor grip.
        page.mouse.move(rect['x'] + rect['width'] * 0.5, rect['y'] + 28)
        page.mouse.down()
        page.mouse.move(rect['x'] + rect['width'] * 0.625, rect['y'] + 28, steps=5)
        page.mouse.up()
        assert abs(float(page.locator('.ds-h3-ts-input').input_value()) - 4) < 0.05
        assert abs(float(page.locator('.ds-h3-te-input').input_value()) - 6) < 0.05
        if len(sys.argv) > 1:
            page.locator('.ds-h3-preview-panel').screenshot(path=sys.argv[1])
        page.click('.ds-h3-play-crop-btn')
        page.wait_for_function('!media.paused && media.currentTime > 4')
        page.evaluate('closePreview(true)')
        assert page.evaluate('media.paused')
        print('PASS independent video crop dragging and stop on close')
        for ranges in (True, False):
            url = f'https://video.test/{ranges}.webm'
            def serve(route):
                headers = {'Access-Control-Allow-Origin': '*'}
                byte_range = route.request.headers.get('range', '')
                if ranges:
                    headers['Accept-Ranges'] = 'bytes'
                if ranges and byte_range.startswith('bytes='):
                    first, last = byte_range[6:].split('-')
                    start = int(first or 0); end = int(last) if last else len(video_bytes) - 1
                    headers['Content-Range'] = f'bytes {start}-{end}/{len(video_bytes)}'
                    route.fulfill(status=206, body=video_bytes[start:end+1], content_type='video/webm', headers=headers)
                else:
                    route.fulfill(body=video_bytes, content_type='video/webm', headers=headers)
            page.route(url, serve)
            page.evaluate("""url => {
                window.viewUrl = () => url;
                openPreview({type:'video', value:'preview.webm', source_duration:8, trim_start:3, trim_end:5});
                window.media = document.querySelector('video'); window.playingTimes = [];
                media.addEventListener('playing', () => playingTimes.push(media.currentTime));
                document.querySelector('.ds-h3-play-crop-btn').click();
            }""", url)
            page.wait_for_function('playingTimes.length > 0')
            start = page.evaluate('playingTimes[0]')
            assert abs(start - 3) < 0.15, f'HTTP video (ranges={ranges}) started at {start}s'
            page.wait_for_function('media.paused && media.currentTime >= 5', timeout=6000)
            assert abs(page.evaluate('media.currentTime') - 5) < 0.15
            assert page.evaluate('media.videoWidth === 320')
            blob_url = page.evaluate('media.currentSrc')
            if ranges:
                assert not blob_url.startswith('blob:'), 'Range-enabled video should retain native seeking'
            page.evaluate('window.revokedUrls = []; window.realRevoke = URL.revokeObjectURL; URL.revokeObjectURL = url => {revokedUrls.push(url); realRevoke(url)}')
            page.evaluate('closePreview(true)')
            if blob_url.startswith('blob:'):
                assert blob_url in page.evaluate('revokedUrls')
            page.evaluate('URL.revokeObjectURL = realRevoke')
            print(f'PASS HTTP video crop with byte ranges={ranges}, Blob fallback={blob_url.startswith("blob:")}')
        page.wait_for_timeout(100)
        assert not errors, errors
        browser.close()


if __name__ == '__main__':
    run_video_tests()
