"""Check saved CPU/MPS gradient directions against central finite differences on CPU."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from openrubrics_lab.backend import LocalLM, token_logps
from openrubrics_lab.config import lm_kwargs, load_config
from openrubrics_lab.io import file_hash, now, read_jsonl, write_json
from openrubrics_lab.training import _completion_tensors, grpo_objective

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/m1-complete-20261116/15-online-grpo"
OUT = ROOT / "experiments/acceptance/online-gradient-check"


def main():
    config = load_config(ROOT / "configs/m1.json")
    config["device"] = "cpu"
    rows = read_jsonl(RUN / "rollouts.jsonl")
    sizes = Counter(r["sample_id"] for r in rows)
    arrays = {device: np.load(OUT / device / "gradients.npz", allow_pickle=False)
              for device in ["cpu", "mps"]}
    epsilon = 0.001
    result = {"at": now(), "epsilon": epsilon, "directions": {}, "optimizer_updates": 0,
              "adapter_sha256": file_hash(RUN / "adapter_model.safetensors"),
              "script_sha256": file_hash(Path(__file__))}
    with LocalLM(**lm_kwargs(config, RUN, trainable=True)) as lm:
        lm.model.get_base_model().set_attn_implementation("eager")
        lm.model.eval()
        parameters = {name: p for name, p in lm.model.named_parameters() if p.requires_grad}
        original = {name: p.detach().clone() for name, p in parameters.items()}

        def objective():
            value = 0.0
            with torch.no_grad():
                for row in rows:
                    ids, prefix = _completion_tensors(lm, row)
                    logps = token_logps(lm.model, ids, prefix)
                    loss, _ = grpo_objective(logps, torch.tensor(row["old_logps"]),
                                            torch.tensor(row["reference_logps"]), row["advantage"],
                                            config["kl_beta"], config["clip_low"], config["clip_high"])
                    value += float(loss) / (len(sizes) * sizes[row["sample_id"]])
            return value

        result["baseline_objective"] = objective()
        for device, source in arrays.items():
            norm = float(np.sqrt(sum(np.sum(source[name].astype(np.float64) ** 2) for name in parameters)))
            direction = {name: torch.from_numpy(source[name].copy()) / norm for name in parameters}
            values = {}
            for sign in [-1, 1]:
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(original[name] + sign * epsilon * direction[name])
                values[sign] = objective()
            cpu_prediction = float(sum(np.sum(arrays["cpu"][name].astype(np.float64) *
                                              direction[name].numpy().astype(np.float64)) for name in parameters))
            derivative = (values[1] - values[-1]) / (2 * epsilon)
            result["directions"][device] = {"finite_difference": derivative,
                                             "cpu_autograd_prediction": cpu_prediction,
                                             "native_autograd_prediction": norm,
                                             "plus_objective": values[1], "minus_objective": values[-1]}
            print(device, result["directions"][device], flush=True)
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(original[name])
    write_json(OUT / "finite-difference.json", result)


if __name__ == "__main__":
    main()
