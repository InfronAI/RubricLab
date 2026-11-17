"""Atomic artifacts and content-addressed provenance; never overwrite a completed run."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        temporary = f.name
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = f.name
    os.replace(temporary, path)


def append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def now():
    return datetime.now(UTC).isoformat()


def environment():
    import psutil
    import torch

    def command(*args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "timestamp_utc": now(), "platform": platform.platform(), "machine": platform.machine(),
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 2), "python": sys.version,
        "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
        "packages": {k: version(k) for k in ("torch", "transformers", "peft", "datasets", "huggingface-hub")},
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
    }


def source_fingerprint(root: Path):
    paths = sorted(p for folder in ("src", "configs", "scripts", "tests")
                   for p in (root / folder).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".json", ".html", ".sh"))
    return {str(p.relative_to(root)): file_hash(p) for p in paths}
