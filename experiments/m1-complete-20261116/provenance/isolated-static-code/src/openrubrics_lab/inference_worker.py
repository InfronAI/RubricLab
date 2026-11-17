"""One inference batch per process, bounding the lifetime of Metal graph caches."""
import json
import sys
from pathlib import Path

from .backend import LocalLM
from .io import write_json


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        with LocalLM(**request["model"]) as lm:
            generations = lm.generate(**request["generation"])
            if lm.device == "mps":
                import torch
                for item in generations:
                    item["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 2**30
        payload = {"generations": generations}
    except ValueError as error:
        # Context validation remains a caller-visible error, as in the direct path.
        payload = {"error": str(error)}
    write_json(Path(request["response"]), payload)


if __name__ == "__main__":
    main()
