"""Real-model recovery check; shortened decoding, not a benchmark score."""
import json
import time
from pathlib import Path

from openrubrics_lab.backend import LocalLM
from openrubrics_lab.io import file_hash, now, write_json, write_jsonl

OUT = Path(__file__).resolve().parent


def main():
    request = json.loads((OUT / "long-input-request.json").read_text())
    model = {**request["model"], "offline": True}
    generation = {**request["generation"], "max_new_tokens": 32}
    started = time.perf_counter()
    with LocalLM(**model, mps_worker_timeout=0.05) as lm:
        retry = lm.generate(**generation)
    retry_wall = time.perf_counter() - started
    write_jsonl(OUT / "retry-generations.jsonl", retry)
    assert all(r["device"] == "cpu" and r["requested_device"] == "mps" for r in retry)
    assert all("exceeded 0.05s" in r["fallback_reason"] for r in retry)
    with LocalLM(**{**model, "device": "cpu"}) as lm:
        reference = lm.generate(**generation)
    write_jsonl(OUT / "cpu-reference.jsonl", reference)
    assert [r["token_ids"] for r in retry] == [r["token_ids"] for r in reference]
    write_json(OUT / "timeout-retry.json", {
        "verified_utc": now(), "script_sha256": file_hash(Path(__file__)),
        "model": model, "decoding": {k: v for k, v in generation.items() if k != "prompt"},
        "request_sha256": file_hash(OUT / "long-input-request.json"),
        "timeout_seconds": 0.05, "retry_wall_seconds": retry_wall,
        "cpu_retry_exact_token_match": True, "votes": len(retry),
        "note": "Forced short worker deadline exercises kill/reap and fresh CPU recovery. "
                "The 32-token diagnostic does not replace the primary 320-token benchmark.",
    })
    print(f"Real timeout recovery passed: {len(retry)} votes match direct CPU tokens; {retry_wall:.2f}s", flush=True)


if __name__ == "__main__":
    main()
