"""Record the real local viewer, then encode a portable H.264 MP4 and looping GIF."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen([sys.executable, "-m", "openrubrics_lab", "demo", "--run",
                               str(args.run), "--port", str(args.port)], stdout=subprocess.DEVNULL)
    errors = []
    try:
        for _ in range(80):
            if server.poll() is not None:
                raise RuntimeError("Demo server exited before it became available")
            try:
                with urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.25)
        with tempfile.TemporaryDirectory(prefix="openrubrics-recording-") as temporary:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(viewport={"width": 1280, "height": 900},
                                              record_video_dir=temporary,
                                              record_video_size={"width": 1280, "height": 900},
                                              device_scale_factor=1, reduced_motion="reduce")
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                page.wait_for_function("window.demoReady === true")
                page.screenshot(path=str(args.output / "demo-poster.png"), full_page=False)
                page.wait_for_timeout(3500)
                page.get_by_role("button", name="Training pipeline").click()
                page.wait_for_timeout(3500)
                page.locator("#steps .step").last.scroll_into_view_if_needed()
                page.wait_for_timeout(1500)
                page.get_by_role("button", name="Inspect a judgment").click()
                options = page.locator("#sample option")
                # Prefer the trained model's actual output, without editing it.
                selected = 0
                for i in range(options.count()):
                    if "08-arm" in options.nth(i).inner_text():
                        selected = i
                        break
                page.locator("#sample").select_option(str(selected))
                page.locator("#prediction").evaluate("e => e.parentElement.scrollIntoView({block: 'center'})")
                page.wait_for_timeout(4500)
                page.locator("#prediction").evaluate("e => e.parentElement.scrollTop = e.parentElement.scrollHeight")
                page.wait_for_timeout(2500)
                page.get_by_role("button", name="Results", exact=True).click()
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(2500)
                video = page.video
                await_path = video.path()
                await_context = context
                await_context.close()
                browser.close()
            if errors:
                raise RuntimeError(f"Browser errors: {errors}")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(await_path),
                            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(args.output / "demo.mp4")], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.output / "demo.mp4"),
                        "-filter_complex", ("fps=6,scale=960:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer"), "-loop", "0",
                        str(args.output / "demo.gif")], check=True)
        from openrubrics_lab.io import file_hash, now, write_json
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                         "format=duration:stream=codec_name,width,height", "-of", "json",
                                         str(args.output / "demo.mp4")], text=True)
        write_json(args.output / "recording.json", {"recorded_at": now(), "run": str(args.run),
                   "manifest_sha256": file_hash(args.run / "manifest.json"), "browser_errors": errors,
                   "media": json.loads(probe), "mp4_sha256": file_hash(args.output / "demo.mp4"),
                   "gif_sha256": file_hash(args.output / "demo.gif"),
                   "description": "Actual browser capture of saved outputs; not a live training time-lapse"})
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
