from __future__ import annotations

import json
from pathlib import Path


def load_config(path: Path):
    config = json.loads(path.read_text())
    positive = ("max_context", "train_samples", "validation_samples", "sft_steps", "sft_lr",
                "gradient_accumulation", "lora_rank", "rubric_tokens", "judge_tokens", "votes",
                "arm_rounds", "arm_samples_per_round", "grpo_updates", "grpo_lr", "dpo_steps")
    for key in positive:
        if config.get(key, 0) <= 0:
            raise ValueError(f"{key} must be positive")
    if config["group_size"] < 2:
        raise ValueError("GRPO requires group_size >= 2")
    if config["votes"] % 2 == 0:
        raise ValueError("Use an odd number of judge votes")
    if config["prompt_style"] not in ("compact", "original"):
        raise ValueError("prompt_style must be compact or original")
    if config["arm_rounds"] * config["arm_samples_per_round"] > config["train_samples"]:
        raise ValueError("ARM rounds require disjoint training prompts")
    if config["eval_samples"] > config["validation_samples"]:
        raise ValueError("eval_samples exceeds validation_samples")
    return config


def lm_kwargs(config, adapter=None, trainable=False):
    return {"model": config["model"], "revision": config.get("revision"),
            "device": config["device"], "max_context": config["max_context"],
            "lora_rank": config["lora_rank"], "adapter": adapter, "trainable": trainable}
