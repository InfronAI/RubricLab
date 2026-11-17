"""Compare the fixed-weight CPU/MPS gradient replay without changing trained weights."""
import json
from pathlib import Path

import numpy as np

from openrubrics_lab.io import file_hash, now, write_json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "online-gradient-check"


def main():
    summaries = {device: json.loads((DATA / device / "summary.json").read_text())
                 for device in ["cpu", "mps"]}
    assert summaries["cpu"]["adapter_sha256"] == summaries["mps"]["adapter_sha256"]
    arrays = {device: np.load(DATA / device / "gradients.npz", allow_pickle=False)
              for device in ["cpu", "mps"]}
    assert set(arrays["cpu"].files) == set(arrays["mps"].files)
    names = sorted(arrays["cpu"].files)
    vectors = {device: np.concatenate([arrays[device][name].astype(np.float64).ravel() for name in names])
               for device in ["cpu", "mps"]}
    cpu, mps = vectors["cpu"], vectors["mps"]
    cpu_norm, mps_norm = float(np.linalg.norm(cpu)), float(np.linalg.norm(mps))
    relative_l2 = float(np.linalg.norm(cpu - mps) / max(cpu_norm, 1e-30))
    cosine = float(np.dot(cpu, mps) / max(cpu_norm * mps_norm, 1e-30))
    max_logp_delta = max(abs(a - b) for ca, ma in zip(summaries["cpu"]["trace"], summaries["mps"]["trace"])
                        for a, b in zip(ca["logps"], ma["logps"]))
    tolerance = {"relative_l2_at_most": 0.02, "cosine_at_least": 0.999,
                 "max_logp_delta_at_most": 0.01}
    close = relative_l2 <= 0.02 and cosine >= 0.999 and max_logp_delta <= 0.01
    result = {"compared_at": now(), "same_adapter": True, "parameters": len(names),
              "gradient_elements": len(cpu), "cpu_norm": cpu_norm, "mps_norm": mps_norm,
              "relative_l2_difference": relative_l2, "cosine_similarity": cosine,
              "max_logp_absolute_difference": max_logp_delta,
              "diagnostic_tolerance": tolerance, "within_diagnostic_tolerance": close,
              "post_clip_norms": {d: summaries[d]["post_clip_norm_float64"] for d in summaries},
              "optimizer_updates": 0,
              "inputs": {str(DATA / d / "summary.json"): file_hash(DATA / d / "summary.json")
                         for d in ["cpu", "mps"]},
              "script_sha256": file_hash(Path(__file__)),
              "scope": "Numerical replay at the final saved online-policy weights, not validation of preference quality or every MPS kernel."}
    write_json(DATA / "comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
