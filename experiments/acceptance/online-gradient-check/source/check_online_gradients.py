"""Replay saved online-GRPO gradients at identical final weights; no optimizer update."""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["cpu", "mps"])
    parser.add_argument("--attention", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/m1.json")
    config["device"] = args.device
    variant = args.device + ("-sdpa" if args.attention == "sdpa" else "")
    if args.no_activation_checkpointing:
        variant += "-no-checkpoint"
    output = ROOT / "experiments/acceptance/online-gradient-check" / variant
    assert not output.exists(), "Preserve prior gradient evidence"
    output.mkdir(parents=True)
    rows = read_jsonl(RUN / "rollouts.jsonl")
    group_sizes = Counter(r["sample_id"] for r in rows)
    trace = []
    with LocalLM(**lm_kwargs(config, RUN, trainable=True)) as lm:
        # Use the same attention algorithm on both devices for this numerical check.
        lm.model.get_base_model().set_attn_implementation(args.attention)
        if args.no_activation_checkpointing:
            lm.model.gradient_checkpointing_disable()
        lm.model.train()
        lm.model.zero_grad(set_to_none=True)
        for index, row in enumerate(rows):
            ids, prefix = _completion_tensors(lm, row)
            logps = token_logps(lm.model, ids, prefix)
            old = torch.tensor(row["old_logps"], device=args.device)
            reference = torch.tensor(row["reference_logps"], device=args.device)
            loss, kl = grpo_objective(logps, old, reference, row["advantage"],
                                     config["kl_beta"], config["clip_low"], config["clip_high"])
            assert torch.isfinite(loss)
            weight = 1 / (len(group_sizes) * group_sizes[row["sample_id"]])
            (loss * weight).backward()
            trace.append({"row": index, "loss": float(loss.detach()), "kl": float(kl.detach()),
                          "weight": weight, "logps": logps.detach().cpu().tolist()})
            print(f"{args.device}: replayed gradient {index + 1}/{len(rows)}", flush=True)
        gradients = {name: parameter.grad.detach().cpu().numpy().copy()
                     for name, parameter in lm.model.named_parameters()
                     if parameter.requires_grad and parameter.grad is not None}
        assert gradients and all(np.isfinite(g).all() for g in gradients.values())
        parameters = [p for p in lm.model.parameters() if p.requires_grad]
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        clipped_norm = float(torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.detach()) for p in parameters if p.grad is not None])))
        clipped_norm_float64 = float(np.sqrt(sum(
            np.sum(p.grad.detach().cpu().numpy().astype(np.float64) ** 2)
            for p in parameters if p.grad is not None)))
        assert clipped_norm <= 1.00001
    np.savez_compressed(output / "gradients.npz", **gradients)
    norms = {name: float(np.linalg.norm(g.astype(np.float64))) for name, g in gradients.items()}
    write_json(output / "summary.json", {
        "at": now(), "device": args.device, "attention": args.attention, "optimizer_updates": 0,
        "activation_checkpointing": not args.no_activation_checkpointing,
        "adapter_sha256": file_hash(RUN / "adapter_model.safetensors"),
        "rollouts_sha256": file_hash(RUN / "rollouts.jsonl"),
        "script_sha256": file_hash(Path(__file__)), "gradients_sha256": file_hash(output / "gradients.npz"),
        "pre_clip_norm": norm, "post_clip_norm": clipped_norm, "all_finite": True,
        "pre_clip_norm_float64": float(np.sqrt(sum(n ** 2 for n in norms.values()))),
        "post_clip_norm_float64": clipped_norm_float64,
        "parameter_norms": norms, "trace": trace,
        "method": "Same final adapter, saved sampled tokens/old/reference log probabilities and advantages; eager attention on both devices; no optimizer step.",
    })
    print(f"{args.device}: norm {norm:.6g} -> {clipped_norm:.6g}", flush=True)


if __name__ == "__main__":
    main()
