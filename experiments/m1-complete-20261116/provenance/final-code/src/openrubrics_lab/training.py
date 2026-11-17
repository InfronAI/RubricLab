"""Completion-only SFT, clipped token-level GRPO, and DPO with fixed references."""
from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .backend import LocalLM, seed_all, token_logps
from .config import lm_kwargs
from .io import append_jsonl, file_hash, write_json
from .prompts import judge_prompt, rubric_prompt


def isolated_training(kind, arguments, output):
    """Release every Metal allocation between stages and isolate mixed-dtype kernel caches."""
    import json
    output.mkdir(parents=True, exist_ok=True)
    request = output / "training_request.json"
    write_json(request, {"kind": kind, "arguments": arguments})
    child_env = {**os.environ, "OPENRUBRICS_TRAINING_WORKER": "1"}
    subprocess.run([sys.executable, "-m", "openrubrics_lab.worker", str(request.resolve())],
                   check=True, env=child_env)
    return json.loads((output / "summary.json").read_text())


def optimizer_for(lm, lr):
    return torch.optim.AdamW([p for p in lm.model.parameters() if p.requires_grad], lr=lr)


def safe_step(lm, optimizer):
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in lm.model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def sft(config, rows, role, output: Path):
    if os.environ.get("OPENRUBRICS_TRAINING_WORKER") != "1":
        return isolated_training("sft", {"config": config, "rows": rows, "role": role,
                                         "output": str(output)}, output)
    seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "training.jsonl"
    with LocalLM(**lm_kwargs(config, trainable=True)) as lm:
        data = []
        for row in rows:
            if role == "rubric":
                prompt, target = rubric_prompt(row["instruction"], config["prompt_style"]), row["rubric"]
            else:
                prompt = judge_prompt(row["instruction"], row["rubric"], row["response_a"], row["response_b"],
                                      config["prompt_style"])
                target = row["judge"]
            # Do not truncate a winner or silently discard a training record.
            lm.encode(prompt, target)
            data.append((prompt, target, row["id"]))
        if not data:
            raise ValueError("Cannot train on empty data")
        optimizer = optimizer_for(lm, config["sft_lr"])
        losses, cursor = [], 0
        random.Random(config["seed"]).shuffle(data)
        for step in range(config["sft_steps"]):
            started = time.perf_counter()
            lm.model.train()
            loss_sum = 0.0
            used = []
            for _ in range(config["gradient_accumulation"]):
                prompt, target, sample_id = data[cursor % len(data)]
                cursor += 1
                ids, _, prefix = lm.encode(prompt, target)
                loss = -token_logps(lm.model, ids, prefix).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite SFT loss")
                (loss/config["gradient_accumulation"]).backward()
                loss_sum += float(loss.detach()) / config["gradient_accumulation"]
                used.append(sample_id)
            norm = safe_step(lm, optimizer)
            record = {"step": step+1, "loss": loss_sum, "grad_norm": norm, "sample_ids": used,
                      "seconds": time.perf_counter()-started}
            if lm.device == "mps":
                record["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 2**30
                record["mps_tensor_allocated_gb"] = torch.mps.current_allocated_memory() / 2**30
            append_jsonl(trace, record)
            losses.append(loss_sum)
            print(f"SFT {role} {step+1}/{config['sft_steps']} loss={loss_sum:.4f}", flush=True)
        lm.save(output)
    summary = {"role": role, "steps": len(losses), "examples": len(data),
               "first_loss": losses[0], "last_loss": losses[-1],
               "adapter_sha256": file_hash(output / "adapter_model.safetensors")}
    write_json(output / "summary.json", summary)
    return summary


def advantages(rewards: torch.Tensor):
    return (rewards-rewards.mean()) / (rewards.std(unbiased=False)+1e-4)


def grpo_objective(logps, old_logps, ref_logps, advantage, beta=0.001, clip_low=0.2, clip_high=0.28):
    """GRPO per-sequence token mean, clipped importance ratio and nonnegative k3 KL."""
    ratio = (logps-old_logps).exp()
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-clip_low, 1+clip_high)*advantage)
    delta = ref_logps-logps
    kl = delta.exp()-delta-1
    return (-surrogate+beta*kl).mean(), kl.mean()


def _completion_tensors(lm, rollout):
    prefix = lm.prompt_ids(rollout["prompt"])
    ids = torch.tensor([prefix+rollout["token_ids"]], device=lm.device)
    if ids.shape[1] > lm.max_context or not rollout["token_ids"]:
        raise ValueError("Invalid rollout length")
    return ids, len(prefix)


def grpo(config, groups, actor_adapter, reference_adapter, output: Path):
    """All rollouts are scored before updating. The reference remains fixed across rounds."""
    if os.environ.get("OPENRUBRICS_TRAINING_WORKER") != "1":
        return isolated_training("grpo", {"config": config, "groups": groups,
                   "actor_adapter": str(actor_adapter) if actor_adapter else None,
                   "reference_adapter": str(reference_adapter) if reference_adapter else None,
                   "output": str(output)}, output)
    seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    flat = [r for group in groups for r in group]
    if not flat:
        raise ValueError("Empty GRPO batch")
    # FP32 for both old/reference probabilities and updates, avoiding dtype-induced initial KL.
    with LocalLM(**lm_kwargs(config, reference_adapter, trainable=True)) as reference:
        reference.model.eval()
        with torch.no_grad():
            for r in flat:
                ids, prefix = _completion_tensors(reference, r)
                r["reference_logps"] = token_logps(reference.model, ids, prefix).cpu().tolist()
    with LocalLM(**lm_kwargs(config, actor_adapter, trainable=True)) as lm:
        lm.model.eval()  # no dropout: recomputation matches rollout-policy log probabilities
        with torch.no_grad():
            for r in flat:
                ids, prefix = _completion_tensors(lm, r)
                r["old_logps"] = token_logps(lm.model, ids, prefix).cpu().tolist()
        optimizer = optimizer_for(lm, config["grpo_lr"])
        zero_variance = 0
        for group in groups:
            rewards = torch.tensor([r["reward"] for r in group])
            adv = advantages(rewards).tolist()
            zero_variance += int(float(rewards.std(unbiased=False)) == 0)
            for r, a in zip(group, adv):
                r["advantage"] = a
        for step in range(config["grpo_updates"]):
            lm.model.train()  # Enables activation checkpointing; LoRA/base dropout are zero.
            loss_sum, kl_sum = 0., 0.
            for group in groups:
                for r in group:
                    ids, prefix = _completion_tensors(lm, r)
                    logps = token_logps(lm.model, ids, prefix)
                    old = torch.tensor(r["old_logps"], device=lm.device)
                    ref = torch.tensor(r["reference_logps"], device=lm.device)
                    loss, kl = grpo_objective(logps, old, ref, r["advantage"], config["kl_beta"],
                                              config["clip_low"], config["clip_high"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite GRPO objective")
                    (loss / (len(groups)*len(group))).backward()
                    loss_sum += float(loss.detach()) / (len(groups)*len(group))
                    kl_sum += float(kl.detach()) / (len(groups)*len(group))
            norm = safe_step(lm, optimizer)
            append_jsonl(output / "training.jsonl", {"step": step+1, "loss": loss_sum,
                         "kl": kl_sum, "grad_norm": norm, "zero_variance_groups": zero_variance})
            print(f"GRPO {output.name} {step+1} loss={loss_sum:.5f} grad={norm:.5f}", flush=True)
        lm.save(output)
    from .io import write_jsonl
    write_jsonl(output / "rollouts.jsonl", flat)
    summary = {"groups": len(groups), "rollouts": len(flat), "zero_variance_groups": zero_variance,
               "reward_mean": sum(r["reward"] for r in flat)/len(flat),
               "reference_adapter": str(reference_adapter) if reference_adapter else "base",
               "actor_adapter": str(actor_adapter) if actor_adapter else "base",
               "adapter_sha256": file_hash(output / "adapter_model.safetensors")}
    write_json(output / "summary.json", summary)
    return summary


def dpo_objective(chosen, rejected, reference_chosen, reference_rejected, beta=0.1, sft_weight=0.2):
    margin = beta * ((chosen.sum()-rejected.sum())-(reference_chosen.sum()-reference_rejected.sum()))
    return -F.logsigmoid(margin)-sft_weight*chosen.mean(), margin


def dpo(config, pairs, actor_adapter, reference_adapter, output: Path):
    if os.environ.get("OPENRUBRICS_TRAINING_WORKER") != "1":
        return isolated_training("dpo", {"config": config, "pairs": pairs,
                   "actor_adapter": str(actor_adapter) if actor_adapter else None,
                   "reference_adapter": str(reference_adapter) if reference_adapter else None,
                   "output": str(output)}, output)
    seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    if not pairs:
        summary = {"status": "skipped_no_consistent_pairs", "steps": 0,
                   "reason": "The judge produced no valid preferences consistent under both orders."}
        write_json(output / "summary.json", summary)
        return summary
    cached = []
    with LocalLM(**lm_kwargs(config, reference_adapter, trainable=True)) as ref, torch.no_grad():
        for pair in pairs:
            scores = []
            for response in (pair["chosen"], pair["rejected"]):
                ids, _, prefix = ref.encode(pair["instruction"], response)
                scores.append(token_logps(ref.model, ids, prefix).cpu())
            cached.append(scores)
    with LocalLM(**lm_kwargs(config, actor_adapter, trainable=True)) as lm:
        optimizer = optimizer_for(lm, config["dpo_lr"])
        lm.model.train()
        for step in range(config["dpo_steps"]):
            idx = step % len(pairs)
            pair = pairs[idx]
            logps = []
            for response in (pair["chosen"], pair["rejected"]):
                ids, _, prefix = lm.encode(pair["instruction"], response)
                logps.append(token_logps(lm.model, ids, prefix))
            loss, margin = dpo_objective(*logps, *[s.to(lm.device) for s in cached[idx]],
                                         config["dpo_beta"], config["sft_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DPO objective")
            loss.backward()
            norm = safe_step(lm, optimizer)
            append_jsonl(output / "training.jsonl", {"step": step+1, "loss": float(loss.detach()),
                         "preference_margin": float(margin.detach()), "grad_norm": norm, "pair_id": pair["id"]})
            print(f"DPO {output.name} {step+1} loss={float(loss.detach()):.4f}", flush=True)
        lm.save(output)
    summary = {"status": "completed", "pairs": len(pairs), "steps": config["dpo_steps"],
               "adapter_sha256": file_hash(output / "adapter_model.safetensors")}
    write_json(output / "summary.json", summary)
    return summary
