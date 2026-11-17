"""Pinned public datasets, bounded downloads, prompt-disjoint splits and benchmark adapters."""
from __future__ import annotations

import functools
import json
import operator
import random
from collections import Counter
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

from .io import digest, file_hash, read_jsonl, write_json, write_jsonl
from .metrics import normalize_label, parse_winner
from .prompts import judge_prompt, rubric_prompt

REPOS = {
    "openrubrics": ("OpenRubrics/OpenRubric-v2", "d1048e9e0375034d1dafb53392789cd368424861"),
    "rewardbench": ("allenai/reward-bench", "168d848cdbbea9764fae4a544dc9ca1e6cca4931"),
    "rewardbench2": ("allenai/reward-bench-2", "7ff08853b0d5686e79b13fda8677024f566a104a"),
    "rmbench": ("THU-KEG/RM-Bench", "5e094adbde982855284918147e84a169f7282a2a"),
    "writing": ("m-a-p/Writing-Preference-Bench", "676d7114ce37ad5d8eff5b39f3648b855730bf77"),
}
CHAT = {"alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-medium"}
CHAT_HARD = {"mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
             "llmbar-adver-GPTOut", "llmbar-adver-manual"}


def prompt_key(text):
    return digest(" ".join(text.split()).casefold())


def validate_pair(row):
    for key in ("instruction", "response_a", "response_b"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing or empty {key}")
    label = normalize_label(row.get("winner", row.get("label", "")))
    if label is None:
        raise ValueError("Pair requires a valid A/B preference label")
    if row["response_a"] == row["response_b"]:
        raise ValueError("Identical responses cannot form a preference pair")
    return {**row, "label": label, "id": str(row.get("id", digest(row)[:16]))}


def split_by_prompt(rows, validation_count, seed=42):
    groups = {}
    for row in rows:
        groups.setdefault(prompt_key(row["instruction"]), []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) <= validation_count:
        raise ValueError("Not enough distinct prompts for nonempty train and validation sets")
    val_keys = set(keys[:validation_count])
    train = [r for k in keys if k not in val_keys for r in groups[k]]
    val = [r for k in keys if k in val_keys for r in groups[k]]
    return train, val


def bounded_source(path: Path, limit=1024, max_bytes=32*1024*1024):
    """Read only a bounded prefix of the released JSONL, closing HTTP after the limit."""
    repo, revision = REPOS["openrubrics"]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/data/open_rubrics.jsonl"
    rows, byte_count = [], 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            byte_count += len(line.encode())+1
            if byte_count > max_bytes:
                break
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    write_jsonl(path, rows)
    return {"repository": repo, "revision": revision, "url": url, "rows": len(rows),
            "bytes_read": byte_count, "row_limit": limit, "byte_limit": max_bytes,
            "sha256": file_hash(path)}


def prepare_data(config, destination: Path):
    from transformers import AutoTokenizer

    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "source.jsonl"
    receipt = bounded_source(raw_path, config.get("source_limit", 1024))
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
    eligible, rejected = [], Counter()
    seen = set()
    for raw in read_jsonl(raw_path):
        try:
            row = validate_pair(raw)
        except ValueError:
            rejected["invalid_pair"] += 1
            continue
        key = prompt_key(row["instruction"])
        if key in seen:
            rejected["duplicate_prompt"] += 1
            continue
        if parse_winner(row.get("judge", "")) != row["label"]:
            rejected["teacher_label_mismatch"] += 1
            continue
        lengths = []
        for prompt, target in (
            (rubric_prompt(row["instruction"], config["prompt_style"]), row["rubric"]),
            (judge_prompt(row["instruction"], row["rubric"], row["response_a"], row["response_b"],
                          config["prompt_style"]), row["judge"]),
        ):
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=True, add_generation_prompt=True)
            lengths.append(len(ids)+len(tokenizer.encode(target, add_special_tokens=False))+1)
        if max(lengths) > config["max_context"]:
            rejected["context_overflow"] += 1
            continue
        seen.add(key)
        row["source"] = REPOS["openrubrics"][0]
        row["source_revision"] = receipt["revision"]
        row["training_tokens"] = max(lengths)
        eligible.append(row)
    random.Random(config["seed"]).shuffle(eligible)
    count = config["train_samples"]+config["validation_samples"]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible records, need {count}; increase source_limit or context")
    selected = eligible[:count]
    train, val = split_by_prompt(selected, config["validation_samples"], config["seed"])
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "validation.jsonl", val)
    receipt.update({"eligible": len(eligible), "rejected": dict(rejected), "train": len(train),
                    "validation": len(val), "selection": "seeded shuffle after context/label validation",
                    "seed": config["seed"], "max_context": config["max_context"]})
    write_json(destination / "manifest.json", receipt)
    return receipt


def _file(repo, revision, filename):
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _read_json(path):
    obj = json.loads(path.read_text())
    return obj if isinstance(obj, list) else obj.get("data", obj)


def benchmark_rows(name):
    """Yield normalized pairs while preserving multi-response group identities."""
    repo, revision = REPOS[name]
    if name in ("rewardbench", "rewardbench2"):
        import pyarrow.parquet as pq
        filename = "data/filtered-00000-of-00001.parquet" if name == "rewardbench" else "data/test-00000-of-00001.parquet"
        raw = pq.read_table(_file(repo, revision, filename)).to_pylist()
    elif name == "rmbench":
        raw = _read_json(_file(repo, revision, "total_dataset.json"))
    elif name == "writing":
        # Explicit English files: never assume the Hub's concatenation order.
        raw = functools.reduce(operator.iadd, (_read_json(_file(repo, revision, f"WP_bench_english-{i}.json")) for i in (1, 2)), [])
    else:
        raise ValueError(f"Unknown benchmark {name}")
    for i, row in enumerate(raw):
        group = str(row.get("id", i))
        subset = row.get("subset", row.get("domain", row.get("category", "writing")))
        if name == "rewardbench":
            if subset not in CHAT | CHAT_HARD:
                continue
            subset = "chat" if subset in CHAT else "chat_hard"
        if name == "rewardbench2" and subset not in ("Precise IF", "Focus"):
            continue
        if name == "rmbench" and row["domain"] != "chat":
            continue
        chosen = row["chosen"] if isinstance(row["chosen"], list) else [row["chosen"]]
        rejected = row["rejected"] if isinstance(row["rejected"], list) else [row["rejected"]]
        for ci, a in enumerate(chosen):
            for ri, b in enumerate(rejected):
                pair = {"id": f"{name}:{group}:{ci}:{ri}", "instruction": row["prompt"],
                        "response_a": a["response"] if isinstance(a, dict) else a,
                        "response_b": b["response"] if isinstance(b, dict) else b,
                        "label": "A", "subset": str(subset),
                        "benchmark": name, "source": repo, "source_revision": revision,
                        "group_id": f"{name}:{group}:{ci}"}
                if name == "rmbench":
                    pair["difficulty"] = "normal" if ci == ri else ("hard" if ci < ri else "easy")
                yield pair


def prepare_benchmarks(config, destination: Path, excluded_prompts: set):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
    receipts = {}
    for name in config["benchmarks"]:
        rows, skipped = [], Counter()
        # Select entire prompt groups to preserve RB2 all-comparisons accuracy and RM style grids.
        groups = {}
        for row in benchmark_rows(name):
            if prompt_key(row["instruction"]) in excluded_prompts:
                skipped["training_overlap"] += 1
                continue
            key = prompt_key(row["instruction"])
            groups.setdefault(key, []).append(row)
        group_list = sorted(groups.values(), key=lambda g: g[0]["id"])
        random.Random(config["seed"]).shuffle(group_list)
        counts = Counter()
        for group in group_list:
            subset = group[0]["subset"]
            if counts[subset] >= config["benchmark_groups"]:
                continue
            # Budget for the rubric and judging scaffolding, without truncating candidate answers.
            if any(len(tok.encode(r["instruction"]+r["response_a"]+r["response_b"])) +
                   config["rubric_tokens"] + config["judge_tokens"] + 420 > config["max_context"]
                   for r in group):
                skipped["context_overflow_groups"] += 1
                continue
            counts[subset] += 1
            for r in group:
                if int(digest(r["id"]+str(config["seed"]))[:8], 16) % 2:
                    r["response_a"], r["response_b"] = r["response_b"], r["response_a"]
                    r["label"] = "B"
            rows.extend(group)
        if not rows:
            raise ValueError(f"No eligible examples for {name}")
        write_jsonl(destination / f"{name}.jsonl", rows)
        receipts[name] = {"repository": REPOS[name][0], "revision": REPOS[name][1],
                          "pairs": len(rows), "prompt_groups_by_subset": dict(counts),
                          "skipped": dict(skipped), "sha256": file_hash(destination / f"{name}.jsonl"),
                          "selection_bias": "Short contexts; bounded seeded sample, not full benchmark"}
    write_json(destination / "benchmarks.json", receipts)
    return receipts
