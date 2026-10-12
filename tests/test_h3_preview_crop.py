"""Exercise Director's actual preview code with real browser audio.

Run: uv run --no-project --with playwright python tests/test_h3_preview_crop.py
"""
import base64
import io
import math
import sys
from pathlib import Path
import struct
import wave

def run_preview_tests():
    from playwright.sync_api import sync_playwright

    ROOT = Path(__file__).resolve().parents[1]
    source = (ROOT / "js/minimax_h3_director.js").read_text()
    preview = source[source.index("  let previewOverlay = null;"):source.index("  const syncAttachedPrompts =")]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        # Left channel is silent; the right channel has a rising envelope.
        wav.writeframes(b"".join(struct.pack("<hh", 0, int((2000 + i / 64000 * 22000) * math.sin(i * 0.2))) for i in range(64000)))
    audio_url = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_content("<html><body></body></html>")
        styles = source[source.index('function installStyles()'):source.index('function parseState(')]
        page.evaluate("code => window.eval('let cssInstalled = false;\\n' + code + '\\ninstallStyles();')", styles)
        page.evaluate("""([code, url]) => {
            window.viewUrl = () => url;
            window.isLockedSlot = () => false;
            window.setStatus = () => {};
            window.mutate = () => {};
            window.eval(code + '\\nwindow.openPreview = openPreview; window.closePreview = closePreview; window.readAudioPeaks = readAudioPeaks;');
        }""", [preview, audio_url])
        # Click synchronously before the browser has loaded metadata. The crop
        # must not lose its start when play() initiates the initial media load.
        page.evaluate("""() => {
            openPreview({type:'audio', value:'test.wav', source_duration:8, trim_start:3, trim_end:5});
            window.media = document.querySelector('audio');
            window.playingTimes = [];
            media.addEventListener('playing', () => playingTimes.push(media.currentTime));
            document.querySelector('.ds-h3-play-crop-btn').click();
        }""")
        page.wait_for_function("playingTimes.length > 0")
        start = page.evaluate("playingTimes[0]")
        assert abs(start - 3) < 0.15, f"Crop began at {start}s instead of 3s"
        page.wait_for_function("media.paused && media.currentTime >= 5", timeout=6000)
        end = page.evaluate("media.currentTime")
        assert abs(end - 5) < 0.15, f"Crop ended at {end}s instead of 5s"
        assert page.locator('.ds-h3-play-crop-btn').inner_text() == '▶ Play crop'
        print(f"PASS cold audio: {start:.3f}s–{end:.3f}s")
        # Native pause events are queued. Restarting the crop before that event
        # arrives must not let the old pause cancel the newly playing crop.
        page.evaluate("""async () => {
            await media.play();
            media.pause();
            document.querySelector('.ds-h3-play-crop-btn').click();
        }""")
        page.wait_for_function("!media.paused && media.currentTime >= 3.2")
        page.wait_for_function("media.paused && media.currentTime >= 5", timeout=6000)
        end = page.evaluate('media.currentTime')
        assert abs(end - 5) < 0.15, f'Restarted crop ran through to {end}s'
        print(f'PASS restart after queued native pause: stopped at {end:.3f}s')
        page.click('.ds-h3-play-crop-btn')
        page.wait_for_function('!media.paused && media.currentTime >= 3.2')
        page.evaluate('media.pause()')
        page.wait_for_function("document.querySelector('.ds-h3-play-crop-btn').textContent === '▶ Play crop'")
        paused_at = page.evaluate('media.currentTime')
        page.wait_for_timeout(350)
        assert page.evaluate('media.paused && media.currentTime') == paused_at
        print('PASS native pause still stops crop playback')
        # The native media player intentionally remains a full-file preview.
        page.evaluate('media.currentTime = 5.5; media.play()')
        page.wait_for_function('!media.paused && media.currentTime > 5.7')
        page.evaluate('media.pause()')
        print('PASS native full-file playback remains available')
        page.wait_for_function("document.querySelector('.ds-h3-preview-waveform')?.dataset.ready === 'true'", timeout=3000)
        waveform = page.locator('.ds-h3-preview-waveform')
        assert waveform.is_visible()
        # Verify real amplitude data reached the canvas, not decorative bars.
        assert waveform.evaluate("c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data].some((v, i) => i % 4 === 3 && v > 0)")
        page.evaluate('media.currentTime = 4')
        page.wait_for_function("Math.abs(parseFloat(document.querySelector('.ds-h3-preview-playhead').style.left) - 50) < 1")
        assert '4.00' in page.locator('.ds-h3-preview-position').inner_text()
        page.click('.ds-h3-play-crop-btn')
        page.wait_for_function('!media.paused && media.currentTime > 3.4')
        cursor = page.locator('.ds-h3-preview-playhead').evaluate('el => parseFloat(el.style.left)')
        assert 37.5 < cursor < 62.5
        page.wait_for_function('media.paused && media.currentTime >= 5', timeout=6000)
        page.wait_for_function("Math.abs(parseFloat(document.querySelector('.ds-h3-preview-playhead').style.left) - 62.5) < 1")
        # Existing drag interaction moves the selection on top of the waveform.
        track = waveform.locator('..')
        rect = track.bounding_box()
        page.mouse.move(rect['x'] + rect['width'] * 0.5, rect['y'] + 30)
        page.mouse.down()
        page.mouse.move(rect['x'] + rect['width'] * 0.625, rect['y'] + 30, steps=5)
        page.mouse.up()
        assert abs(float(page.locator('.ds-h3-ts-input').input_value()) - 4) < 0.05
        assert abs(float(page.locator('.ds-h3-te-input').input_value()) - 6) < 0.05
        print('PASS full waveform, live cursor, crop stop and crop dragging')
        page.click('.ds-h3-play-crop-btn')
        page.wait_for_function('!media.paused')
        page.evaluate('closePreview(true)')
        assert page.evaluate('media.paused')
        assert page.locator('.ds-h3-preview-waveform').count() == 0
        print('PASS preview close stops audio and removes visualization')
        decoded = page.evaluate('async () => await readAudioPeaks("test.wav")')
        assert len(decoded) == 1024
        assert 0.05 < decoded[0] < 0.1 and 0.65 < decoded[-1] < 0.8
        print('PASS waveform covers full source and non-silent right channel')
        page.evaluate("""peaks => {
            window.realFetch = window.fetch;
            window.waveformFetches = 0;
            window.fetch = (...args) => { waveformFetches++; return realFetch(...args); };
            openPreview({type:'audio', value:'test.wav', source_duration:8, trim_start:2, trim_end:6, waveform_peaks:peaks});
            window.media = document.querySelector('audio');
            media.currentTime = 4;
        }""", decoded)
        page.wait_for_function("document.querySelector('.ds-h3-preview-waveform')?.dataset.ready === 'true'")
        assert page.evaluate('waveformFetches') == 0
        waveform = page.locator('.ds-h3-preview-waveform')
        page.wait_for_function('document.querySelector(".ds-h3-preview-waveform").width > 500')
        original_width = waveform.evaluate('c => c.width')
        page.set_viewport_size({'width': 400, 'height': 700})
        page.wait_for_function('document.querySelector(".ds-h3-preview-waveform").width < ' + str(original_width))
        assert page.locator('.ds-h3-preview-position').is_visible()
        assert page.locator('.ds-h3-save-btn').is_visible()
        page.set_viewport_size({'width': 900, 'height': 700})
        page.wait_for_function('document.querySelector(".ds-h3-preview-waveform").width > 500')
        if len(sys.argv) > 1:
            page.locator('.ds-h3-preview-panel').screenshot(path=sys.argv[1])
        print('PASS cached waveform and responsive redraw')
        # Drag the cursor's dedicated grip without changing either crop edge.
        rect = waveform.locator('..').bounding_box()
        grip = page.locator('.ds-h3-preview-playhead-grip')
        assert grip.count() == 1, 'Playback cursor has no draggable grip'
        grip.hover()
        page.mouse.down()
        page.mouse.move(rect['x'] + rect['width'] * 0.625, rect['y'] + 7, steps=5)
        page.mouse.up()
        page.wait_for_function('Math.abs(media.currentTime - 5) < 0.1')
        assert page.locator('.ds-h3-ts-input').input_value() == '2.00'
        assert page.locator('.ds-h3-te-input').input_value() == '6.00'
        page.evaluate('window.scrubStart = null; media.addEventListener("playing", () => scrubStart = media.currentTime, {once:true}); media.play()')
        page.wait_for_function('scrubStart !== null')
        assert abs(page.evaluate('scrubStart') - 5) < 0.15
        page.evaluate('media.pause()')
        grip.focus()
        page.keyboard.press('ArrowLeft')
        page.wait_for_function('Math.abs(media.currentTime - 4) < 0.1')
        print('PASS draggable/keyboard playback cursor and play from chosen position')
        page.evaluate("""() => {
            closePreview(true);
            window.fetch = () => Promise.reject(new Error('Test waveform fetch failure'));
            openPreview({type:'audio', value:'test.wav', source_duration:8, trim_start:3, trim_end:5});
            window.media = document.querySelector('audio');
        }""")
        page.wait_for_function("document.querySelector('.ds-h3-preview-position').textContent.includes('Waveform unavailable')")
        page.click('.ds-h3-play-crop-btn')
        page.wait_for_function('!media.paused && media.currentTime > 3.2')
        page.wait_for_function('media.paused && media.currentTime >= 5', timeout=6000)
        assert abs(page.evaluate('media.currentTime') - 5) < 0.15
        print('PASS waveform failure preserves real audio crop playback')
        page.evaluate("""() => {
            closePreview(true);
            window.fetchAborted = false;
            window.fetch = (_, {signal}) => new Promise((resolve, reject) => {
                signal.addEventListener('abort', () => {fetchAborted = true; reject(new DOMException('Aborted', 'AbortError'));});
            });
            openPreview({type:'audio', value:'test.wav', source_duration:8, trim_start:3, trim_end:5});
            closePreview(true);
        }""")
        assert page.evaluate('fetchAborted')
        assert page.locator('.ds-h3-preview-overlay').count() == 0
        print('PASS closing preview aborts pending waveform fetch')
        def serve_ranges(route):
            data = buffer.getvalue()
            byte_range = route.request.headers.get('range', '')
            headers = {'Access-Control-Allow-Origin': '*', 'Accept-Ranges': 'bytes'}
            if byte_range.startswith('bytes='):
                first, last = byte_range[6:].split('-')
                start_byte = int(first or 0)
                end_byte = int(last) if last else len(data) - 1
                headers['Content-Range'] = f'bytes {start_byte}-{end_byte}/{len(data)}'
                route.fulfill(status=206, body=data[start_byte:end_byte + 1], content_type='audio/wav', headers=headers)
            else:
                route.fulfill(body=data, content_type='audio/wav', headers=headers)
        page.route('https://audio.test/ranges.wav', serve_ranges)
        page.evaluate("""() => {
            window.fetch = realFetch;
            window.viewUrl = () => 'https://audio.test/ranges.wav';
            openPreview({type:'audio', value:'ranges.wav', source_duration:8, trim_start:3, trim_end:5});
            window.media = document.querySelector('audio');
            window.playingTimes = [];
            media.addEventListener('playing', () => playingTimes.push(media.currentTime));
            document.querySelector('.ds-h3-play-crop-btn').click();
        }""")
        page.wait_for_function('playingTimes.length > 0')
        assert abs(page.evaluate('playingTimes[0]') - 3) < 0.15
        assert not page.evaluate('media.currentSrc.startsWith("blob:")')
        page.wait_for_function('media.paused && media.currentTime >= 5', timeout=6000)
        page.evaluate('closePreview(true)')
        print('PASS range-enabled HTTP source uses native seeking')
        page.route('https://audio.test/clip.wav', lambda route: route.fulfill(
            body=buffer.getvalue(), content_type='audio/wav', headers={'Access-Control-Allow-Origin': '*'}))
        page.evaluate("""() => {
            window.fetch = realFetch;
            window.viewUrl = () => 'https://audio.test/clip.wav';
            openPreview({type:'audio', value:'clip.wav', source_duration:8, trim_start:3, trim_end:5});
            window.media = document.querySelector('audio');
            window.playingTimes = [];
            window.httpTrace = [];
            for (const name of ['loadedmetadata','seeking','seeked','playing','pause']) media.addEventListener(name, () => httpTrace.push([name, media.currentTime, media.duration, media.seekable.length]));
            media.addEventListener('playing', () => playingTimes.push(media.currentTime));
            document.querySelector('.ds-h3-play-crop-btn').click();
        }""")
        page.wait_for_function('playingTimes.length > 0')
        start = page.evaluate('playingTimes[0]')
        assert abs(start - 3) < 0.15, f'HTTP crop began at {start}s instead of 3s: {page.evaluate("httpTrace")}'
        page.wait_for_function('media.paused && media.currentTime >= 5', timeout=6000)
        assert abs(page.evaluate('media.currentTime') - 5) < 0.15
        print('PASS HTTP audio crop starts at selected offset, not file beginning')
        blob_url = page.evaluate('media.currentSrc')
        assert blob_url.startswith('blob:')
        page.evaluate('window.revokedUrls = []; window.realRevoke = URL.revokeObjectURL; URL.revokeObjectURL = url => {revokedUrls.push(url); realRevoke(url)}')
        page.evaluate('closePreview(true)')
        assert blob_url in page.evaluate('revokedUrls')
        print('PASS closing preview releases seek fallback Blob')
        page.wait_for_timeout(100)
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    run_preview_tests()
