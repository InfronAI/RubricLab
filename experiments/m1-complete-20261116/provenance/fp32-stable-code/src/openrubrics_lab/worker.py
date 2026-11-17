"""Internal fresh-process training worker; the parent retains all stage receipts."""
import json
import sys
from pathlib import Path

from . import training


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    kind = request["kind"]
    if kind not in {"sft", "grpo", "dpo"}:
        raise ValueError(f"Unknown training job: {kind}")
    arguments = request["arguments"]
    arguments["output"] = Path(arguments["output"])
    getattr(training, kind)(**arguments)


if __name__ == "__main__":
    main()
