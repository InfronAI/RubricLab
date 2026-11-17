"""Ordered stages with config/input fingerprints, recoverable receipts, and error artifacts."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .data import prepare_benchmarks, prompt_key
from .experiments import alternate, curate, evaluate, iterative_dpo, label_policy_pairs, online_policy
from .io import digest, environment, file_hash, now, read_jsonl, source_fingerprint, write_json
from .training import dpo, sft


def run(config, data: Path, output: Path, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    inputs = {p.name: file_hash(p) for p in (data / "train.jsonl", data / "validation.jsonl")}
    code = source_fingerprint(Path(__file__).resolve().parents[2])
    signature = digest({"config": config, "inputs": inputs, "code": code})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError("Run exists; choose a new output path or use --resume")
        if manifest["signature"] != signature:
            raise ValueError("Resume rejected: config, inputs or code changed. Use a new output directory.")
    else:
        manifest = {"signature": signature, "config": config, "inputs": inputs, "source_files": code,
                    "environment": environment(), "started_utc": now(), "stages": {},
                    "claim": "Small-scale mechanism experiments; not paper-scale reproduction"}
        write_json(manifest_path, manifest)
    train = read_jsonl(data / "train.jsonl")
    validation = read_jsonl(data / "validation.jsonl")
    if {prompt_key(r["instruction"]) for r in train} & {prompt_key(r["instruction"]) for r in validation}:
        raise ValueError("Training/validation prompt leakage detected")
    eval_rows = validation[:config["eval_samples"]]

    def stage(name, fn):
        previous = manifest["stages"].get(name, {})
        if previous.get("status") == "completed":
            for relative, expected in previous.get("artifacts", {}).items():
                path = output / relative
                if not path.is_file() or file_hash(path) != expected:
                    raise ValueError(f"Completed artifact changed or missing: {relative}")
            print(f"Resume: {name} verified", flush=True)
            return previous.get("result")
        # Never append to a partial stage's logs. Keep it as evidence and retry elsewhere.
        dest = output / name
        if dest.exists():
            dest.rename(output / f"{name}.failed-{time.time_ns()}")
        started = time.perf_counter()
        manifest["stages"][name] = {"status": "running", "started_utc": now()}
        write_json(manifest_path, manifest)
        print(f"\n=== {name} ===", flush=True)
        try:
            result = fn(dest)
            artifacts = {str(p.relative_to(output)): file_hash(p) for p in sorted(dest.rglob("*")) if p.is_file()}
            manifest["stages"][name].update({"status": "completed", "seconds": time.perf_counter()-started,
                                             "result": result, "artifacts": artifacts})
            write_json(manifest_path, manifest)
            return result
        except Exception as error:
            manifest["stages"][name].update({"status": "failed", "error": str(error),
                                             "traceback": traceback.format_exc(), "seconds": time.perf_counter()-started})
            write_json(manifest_path, manifest)
            raise

    stage("01-curation", lambda dest: curate(config, train, dest))
    stage("02-sft-rubric", lambda dest: sft(config, train, "rubric", dest))
    stage("03-sft-judge", lambda dest: sft(config, train, "judge", dest))
    rubric_init, judge_init = output / "02-sft-rubric", output / "03-sft-judge"
    stage("04-base-direct", lambda dest: evaluate(config, eval_rows, None, None, dest, "direct"))
    stage("05-base-rubric", lambda dest: evaluate(config, eval_rows, None, None, dest))
    stage("06-sft-evaluation", lambda dest: evaluate(config, eval_rows, rubric_init, judge_init, dest, swap=True))

    def arm(dest, variant="standard"):
        r, j = alternate(config, train, rubric_init, judge_init, dest, variant)
        return {"rubric": str(r), "judge": str(j)}

    checkpoints = stage("07-arm", arm)
    rubric, judge = Path(checkpoints["rubric"]), Path(checkpoints["judge"])
    stage("08-arm-evaluation", lambda dest: evaluate(config, eval_rows, rubric, judge, dest, swap=True))
    for variant in config["ablations"]:
        ablated = stage(f"09-ablation-{variant}", lambda dest, v=variant: arm(dest, v))
        stage(f"10-eval-{variant}", lambda dest, cp=ablated:
              evaluate(config, eval_rows, cp["rubric"], cp["judge"], dest))

    def benchmark_stage(dest):
        receipts = prepare_benchmarks(config, dest / "data", {prompt_key(r["instruction"]) for r in train})
        results = {}
        for name in config["benchmarks"]:
            rows = read_jsonl(dest / "data" / f"{name}.jsonl")
            results[name] = evaluate(config, rows, rubric, judge, dest / name)
        return {"data": receipts, "metrics": results}

    stage("11-public-benchmarks", benchmark_stage)

    def dpo_stage(dest):
        pairs = label_policy_pairs(config, train[:config["policy_samples"]], rubric, judge, dest / "labels")
        return dpo(config, pairs, None, None, dest / "adapter")

    stage("12-dpo-arm", dpo_stage)

    def gold_baseline(dest):
        pairs = []
        for row in train[:config["policy_samples"]]:
            a, b = row["response_a"], row["response_b"]
            pairs.append({"id": row["id"], "instruction": row["instruction"],
                          "chosen": a if row["label"] == "A" else b,
                          "rejected": b if row["label"] == "A" else a})
        return dpo(config, pairs, None, None, dest)

    stage("13-dpo-source-baseline", gold_baseline)
    stage("14-iterative-dpo", lambda dest: str(iterative_dpo(config, train, rubric, judge, dest)))
    stage("15-online-grpo", lambda dest: online_policy(config, train, rubric, judge, dest))
    manifest["finished_utc"] = now()
    manifest["status"] = "completed_small_scale_protocol"
    write_json(manifest_path, manifest)
    from .report import build_report
    build_report(output)
    return manifest
