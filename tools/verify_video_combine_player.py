"""Real-browser regression probe against an isolated ComfyUI instance.

Run with: uv run --no-project --with playwright --with av --with numpy python
  tools/verify_video_combine_player.py --url http://127.0.0.1:8199
  --output-directory <isolated ComfyUI output directory>
Never point this probe at a user's active browser session. It uses a fresh context.
"""
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction

import av
import numpy as np
from playwright.sync_api import sync_playwright


CASES = [
    ("H.264", "libx264", "MP4", "mp4", {"preset": "ultrafast"}),
    ("H.265", "libx265", "MP4", "mp4", {"preset": "ultrafast", "x265-params": "log-level=error:pools=1"}),
    ("VP9", "libvpx-vp9", "WebM", "webm", {"cpu-used": "8"}),
    ("AV1", "libsvtav1", "WebM", "webm", {"preset": "12", "svtav1-params": "lp=2"}),
]


def create_fixtures(output):
    output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for codec, encoder, container, suffix, options in CASES:
        path = output / f"player-regression-{codec}.{suffix}"
        with av.open(str(path), "w", format=suffix, options={"movflags": "+faststart"} if suffix == "mp4" else {}) as media:
            stream = media.add_stream(encoder, rate=24, options=options)
            assert isinstance(stream, av.VideoStream)
            stream.width, stream.height = 192, 112
            stream.pix_fmt = "yuv420p"
            stream.codec_context.gop_size = 24
            for index in range(192):
                pixels = np.full((112, 192, 3), (index % 256, 60, 180), dtype=np.uint8)
                pixels[:, index % 192:index % 192 + 4] = 255
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 24)
                for packet in stream.encode(frame):
                    media.mux(packet)
            for packet in stream.encode():
                media.mux(packet)
        fixtures.append({"filename": path.name, "codec": codec, "container": container,
                         "bit_depth": 8, "width": 192, "height": 112, "fps": 24})
    return fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    fixtures = create_fixtures(args.output_directory)
    hashes = {f["filename"]: hashlib.sha256((args.output_directory / f["filename"]).read_bytes()).hexdigest() for f in fixtures}
    source = (Path(__file__).resolve().parents[1] / "js/enhanced_video_combine_preview.js").read_text()
    results, errors = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("Boolean(window.comfyAPI?.app?.app?.graph && window.LiteGraph?.registered_node_types?.DaSiWa_EnhancedVideoCombine)", timeout=90000)
        page.wait_for_timeout(3000)  # Wait for the startup workflow/template modal.
        page.keyboard.press("Escape")
        extensions = page.request.get(args.url + "/extensions").json()
        asset = next(url for url in extensions if url.endswith("/enhanced_video_combine_preview.js"))
        assert page.request.get(args.url + asset).text() == source, "Served extension differs from edited checkout"
        for modern in [False, True]:
            page.evaluate("async mode => {await comfyAPI.app.app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled', mode)}", modern)
            page.wait_for_timeout(500)
            page.evaluate("""() => {
                const app = comfyAPI.app.app;
                app.graph.clear(); app.canvas.ds.scale = 1;
                const node = LiteGraph.createNode('DaSiWa_EnhancedVideoCombine');
                node.pos = [100, 50]; app.graph.add(node); node.setSize([540, 850]);
                node.properties.autoplay = false; node.dasiwaAutoPlayCheckbox.checked = false;
                window.playerProbe = node;
            }""")
            slider = page.get_by_role("slider", name="Seek video preview")
            assert slider.is_disabled(), "Unloaded timeline should be disabled"
            for fixture in fixtures:
                page.evaluate("video => playerProbe.onExecuted({videos:[video]})", fixture)
                page.wait_for_function("playerProbe.dasiwaVideoPreview.readyState >= 2 && !document.querySelector('[aria-label=\"Seek video preview\"]').disabled", timeout=60000)
                video = page.locator(".dasiwa-enhanced-video-preview video")
                for scale in [1, 0.65]:
                    page.evaluate("scale => {comfyAPI.app.app.canvas.ds.scale=scale; comfyAPI.app.app.graph.setDirtyCanvas(true,true)}", scale)
                    page.wait_for_timeout(200)
                    before = page.evaluate("Array.from(playerProbe.pos)")
                    bounds = slider.bounding_box()
                    page.mouse.click(bounds["x"] + bounds["width"] * .7, bounds["y"] + bounds["height"] / 2)
                    page.wait_for_function("playerProbe.dasiwaVideoPreview.currentTime > 4.5 && !playerProbe.dasiwaVideoPreview.seeking")
                    # Actual dragging backwards, not just setting currentTime in JS.
                    page.mouse.move(bounds["x"] + bounds["width"] * .7, bounds["y"] + bounds["height"] / 2)
                    page.mouse.down()
                    page.mouse.move(bounds["x"] + bounds["width"] * .25, bounds["y"] + bounds["height"] / 2, steps=8)
                    page.mouse.up()
                    page.wait_for_function("playerProbe.dasiwaVideoPreview.currentTime > 1 && playerProbe.dasiwaVideoPreview.currentTime < 3 && !playerProbe.dasiwaVideoPreview.seeking")
                    keyboard_time = page.evaluate("playerProbe.dasiwaVideoPreview.currentTime")
                    slider.focus(); page.keyboard.press("ArrowRight")
                    page.wait_for_function("previous => playerProbe.dasiwaVideoPreview.currentTime > previous", arg=keyboard_time)
                    assert page.evaluate("Array.from(playerProbe.pos)") == before, "Seeking moved the graph node"
                # Playback must advance and update the slider, with a decoded frame.
                page.evaluate("() => playerProbe.dasiwaVideoPreview.play()")
                page.wait_for_function("playerProbe.dasiwaVideoPreview.currentTime > 3")
                page.evaluate("() => playerProbe.dasiwaVideoPreview.pause()")
                assert page.evaluate("playerProbe.dasiwaVideoPreview.videoWidth") == 192
                assert abs(float(slider.input_value()) - page.evaluate("playerProbe.dasiwaVideoPreview.currentTime")) < .1
                page.get_by_role("button", name="Fullscreen", exact=True).click()
                page.wait_for_function("document.fullscreenElement === playerProbe.dasiwaVideoPreview")
                page.evaluate("() => document.exitFullscreen()")
                page.wait_for_function("document.fullscreenElement === null")
                video.dblclick(position={"x": 40, "y": 30})
                page.wait_for_function("document.fullscreenElement === playerProbe.dasiwaVideoPreview")
                page.evaluate("() => document.exitFullscreen()")
                page.wait_for_function("document.fullscreenElement === null")
                state = page.evaluate("""() => ({time:playerProbe.dasiwaVideoPreview.currentTime, duration:playerProbe.dasiwaVideoPreview.duration, src:playerProbe.dasiwaVideoPreview.src, download:playerProbe.dasiwaVideoPreview.parentElement.querySelector('a').href})""")
                assert ("/view?" if fixture["codec"] == "H.264" else "/dasiwa/enhanced-video-preview?") in state["src"]
                assert "/view?" in state["download"] and fixture["filename"] in state["download"]
                response = page.request.get(state["src"], headers={"Range": "bytes=0-99"})
                assert response.status == 206, f"Range support failed: {response.status}"
                assert len(response.body()) == 100
                results.append({"renderer": "modern" if modern else "classic", "codec": fixture["codec"], "duration": state["duration"], "seek_click_drag_keyboard": True, "fullscreen_button_doubleclick": True, "range_status": response.status})
            if args.screenshot:
                page.screenshot(path=str(args.screenshot))
            page.evaluate("() => comfyAPI.app.app.graph.remove(playerProbe)")
            assert page.locator(".dasiwa-enhanced-video-preview").count() == 0
        browser.close()
    assert not errors, errors
    for filename, digest in hashes.items():
        assert hashlib.sha256((args.output_directory / filename).read_bytes()).hexdigest() == digest, "Preview changed the original output"
    print(json.dumps({"passed": results, "page_errors": errors, "originals_unchanged": True, "served_asset_matches": True}, indent=2))


if __name__ == "__main__":
    main()
