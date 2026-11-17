"""Verify the finished experiment viewer, recorded media, and documentation links."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import sync_playwright

from openrubrics_lab.io import file_hash, now, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/m1-cpu-training-20261116"
MEDIA = ROOT / "docs/assets"


def ready(url, process):
    for _ in range(80):
        assert process.poll() is None, "Verification server exited"
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(url)


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert manifest["status"] == "completed_small_scale_protocol"
    supplementary = ROOT / "experiments/supplementary"
    execution = json.loads((supplementary / "execution.json").read_text())
    assert execution.get("finished_utc") and all(j["status"] == "completed" for j in execution["jobs"])
    combined = json.loads((supplementary / "combined.json").read_text())
    amendment = json.loads((supplementary / "provenance/analysis-device-note.json").read_text())
    assert combined["analysis_source_sha256"] == file_hash(supplementary / "analyze.py")
    assert amendment["current_sha256"] == combined["analysis_source_sha256"]
    assert amendment["previous_sha256"] == execution["code_hashes"]["analyze.py"]
    for name, expected in execution["code_hashes"].items():
        path = supplementary / (amendment["previous_source"] if name == "analyze.py" else name)
        assert file_hash(path) == expected, name
    vendor = ROOT / "vendor/instruction_following_eval"
    verifier_hashes = json.loads((vendor / "SHA256SUMS.json").read_text())
    for relative, expected in verifier_hashes.items():
        assert file_hash(vendor / relative) == expected, relative
    split = json.loads((OUT / "ifeval-split-check.json").read_text())
    expected_keys = [r["key"] for r in split["selected"]]
    policy_variants = json.loads((ROOT / "experiments/supplementary/policy-variants.json").read_text())
    for variant in policy_variants["policies"]:
        inputs = ROOT / "experiments/supplementary" / f"ifeval-{variant['name']}" / "inputs.jsonl"
        assert [json.loads(line)["key"] for line in inputs.read_text().splitlines()] == expected_keys
    recording = json.loads((MEDIA / "recording.json").read_text())
    assert Path(recording["run"]).resolve() == RUN.resolve()
    assert recording["manifest_sha256"] == file_hash(RUN / "manifest.json")
    assert recording["mp4_sha256"] == file_hash(MEDIA / "demo.mp4")
    assert recording["gif_sha256"] == file_hash(MEDIA / "demo.gif")
    with Image.open(MEDIA / "demo.gif") as gif:
        animation = {"frames": gif.n_frames, "loop": gif.info.get("loop"), "size": gif.size}
        assert gif.n_frames > 2 and gif.info.get("loop") == 0
    errors = []
    checks = {}
    viewer = subprocess.Popen(
        [sys.executable, "-m", "openrubrics_lab", "demo", "--run", str(RUN), "--port", "8768"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    static = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8770", "--bind", "127.0.0.1"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ready("http://127.0.0.1:8768", viewer)
        ready("http://127.0.0.1:8770/docs/demo.html", static)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name, width, height in [("desktop", 1280, 900), ("mobile", 390, 844)]:
                context = browser.new_context(viewport={"width": width, "height": height})
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto("http://127.0.0.1:8768")
                page.wait_for_function("window.demoReady === true")
                overflows = {"results": page.evaluate("document.documentElement.scrollWidth > window.innerWidth")}
                page.screenshot(path=str(OUT / f"final-viewer-{name}.png"), full_page=True)
                page.get_by_role("button", name="Training pipeline").click()
                overflows["pipeline"] = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                stages = page.locator("#steps .step").count()
                assert stages == len(manifest["stages"])
                page.get_by_role("button", name="Inspect a judgment").click()
                assert page.locator("#sample option").count() > 0
                overflows["judgment"] = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                assert not any(overflows.values()), (name, overflows)
                checks[name] = {"horizontal_overflow": overflows, "stages": stages}
                context.close()
            page = browser.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:8770/docs/demo.html")
            page.wait_for_function("document.querySelector('video').currentTime > 1", timeout=15000)
            autoplay = page.locator("video").evaluate(
                "v => ({paused:v.paused, muted:v.muted, loop:v.loop, currentTime:v.currentTime, readyState:v.readyState})"
            )
            assert not autoplay["paused"] and autoplay["muted"] and autoplay["loop"]
            browser.close()
    finally:
        for server in (viewer, static):
            server.terminate()
            server.wait(timeout=10)
    assert not errors, errors
    documents = [ROOT / x for x in [
        "README.md", "README.zh-CN.md", "experiments/RESULTS.md", "experiments/RESULTS.zh-CN.md",
        "docs/PAPER_COVERAGE.md", "docs/M1_ENGINEERING.md", "docs/EXPERIMENT_PROTOCOL.md", "docs/GRADIENT_AUDIT.md",
        "experiments/supplementary/README.md",
    ]]
    local_links = 0
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            target = unquote(target.split("#", 1)[0]).strip("<>")
            if target:
                assert (document.parent / target).exists(), (document, target)
                local_links += 1
    receipt = {
        "verified_at": now(), "run": str(RUN.relative_to(ROOT)), "viewer": checks,
        "browser_errors": errors, "autoplay": autoplay, "gif": animation,
        "media_hashes_match": True, "recorded_manifest_matches": True,
        "local_document_links_checked": local_links,
        "vendored_verifier_hashes_checked": len(verifier_hashes),
        "ifeval_policies_share_verified_disjoint_inputs": True,
        "completed_supplementary_jobs": len(execution["jobs"]),
        "analysis_source_hash_and_documented_amendment_verified": True,
    }
    write_json(OUT / "final-delivery-check.json", receipt)
    write_json(OUT / "autoplay-check.json", {**autoplay, "mp4_sha256": file_hash(MEDIA / "demo.mp4")})
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
