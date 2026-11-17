"""One model at a time, FP32 on M1, explicit context bounds, LoRA adapters."""
from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(device="auto"):
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable. Use an arm64 Python and macOS 12.3+, or --device cpu.")
    if device not in ("mps", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device}")
    return device


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LocalLM:
    def __init__(self, model: str, *, revision: str | None = None, device="auto", adapter=None,
                 trainable=False, max_context=2048, lora_rank=8, offline=False):
        self.device = resolve_device(device)
        if self.device == "mps":
            # Bound allocator growth instead of letting macOS thrash unified memory.
            torch.mps.set_per_process_memory_fraction(0.8)
        self.max_context = max_context
        self.model_id = model
        self.revision = revision
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=offline)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Train in FP32 for stable optimizer updates; FP16 inference fits unified memory.
        # M1 does not require BF16 or any CUDA-only kernels.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, local_files_only=offline,
            dtype=torch.float32 if trainable or self.device == "cpu" else torch.float16,
            attn_implementation="eager" if trainable and self.device == "mps" else "sdpa",
            trust_remote_code=False,
        ).to(self.device)
        self.max_context = min(max_context, getattr(self.model.config, "max_position_embeddings", max_context))
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=trainable,
                                                   autocast_adapter_dtype=trainable)
        elif trainable:
            self.model = get_peft_model(self.model, LoraConfig(
                r=lora_rank, lora_alpha=lora_rank*2, lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none",
            ))
        if trainable:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model.eval()

    def prompt_ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )

    def encode(self, prompt, completion):
        """No silent truncation: preserve every response token and the final winner."""
        prefix = self.prompt_ids(prompt)
        suffix = self.tokenizer.encode(completion, add_special_tokens=False)
        if not suffix or suffix[-1] != self.tokenizer.eos_token_id:
            suffix.append(self.tokenizer.eos_token_id)
        if len(prefix)+len(suffix) > self.max_context:
            raise ValueError(f"Sequence exceeds context: {len(prefix)+len(suffix)} > {self.max_context}")
        ids = torch.tensor([prefix+suffix], device=self.device)
        labels = ids.clone()
        labels[:, :len(prefix)] = -100
        return ids, labels, len(prefix)

    def generate(self, prompt, max_new_tokens=192, temperature=0.7, seed=42, n=1):
        prefix = self.prompt_ids(prompt)
        if len(prefix)+max_new_tokens > self.max_context:
            raise ValueError(f"Prompt plus generation exceeds context: {len(prefix)} + "
                             f"{max_new_tokens} > {self.max_context}. Increase max_context or choose shorter data.")
        if n < 1:
            raise ValueError("n must be positive")
        result = []
        self.model.eval()
        seed_all(seed)
        ids = torch.tensor([prefix]*n, device=self.device)
        started = time.perf_counter()
        kwargs = {"temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature > 0 else {}
        with torch.inference_mode():
            batch = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, use_cache=True, **kwargs,
            )[:, len(prefix):].tolist()
        elapsed = time.perf_counter()-started
        for generated in batch:
            if self.tokenizer.eos_token_id in generated:
                generated = generated[:generated.index(self.tokenizer.eos_token_id)+1]
            result.append({"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                           "token_ids": generated, "prompt_tokens": len(prefix),
                           "tokens": len(generated), "seconds": elapsed/n,
                           "batch_seconds": elapsed, "batch_size": n, "seed": seed,
                           "tokens_per_second": len(generated)/max(elapsed/n, 1e-9),
                           "hit_token_limit": len(generated) == max_new_tokens and
                           generated[-1] != self.tokenizer.eos_token_id})
        return result

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)

    def close(self):
        del self.model
        release_memory()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def token_logps(model, ids, prefix_length, chunk_size=64):
    """Exact suffix log likelihood without allocating [context, vocabulary] logits.

    Qwen/Llama-family decoder + output head, including active PEFT layers. Head
    chunks are recomputed in backward, so long judge trajectories fit 16 GB.
    """
    from torch.utils.checkpoint import checkpoint

    causal = model.get_base_model() if isinstance(model, PeftModel) else model
    hidden = causal.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    hidden = hidden[:, prefix_length-1:-1, :]
    targets = ids[:, prefix_length:]
    head = causal.get_output_embeddings()

    def logps_chunk(h, target):
        logits = head(h).float()
        return -torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                  target.reshape(-1), reduction="none")

    values = []
    for start in range(0, hidden.shape[1], chunk_size):
        h = hidden[:, start:start+chunk_size]
        target = targets[:, start:start+chunk_size]
        if torch.is_grad_enabled():
            values.append(checkpoint(logps_chunk, h, target, use_reentrant=False))
        else:
            values.append(logps_chunk(h, target))
    return torch.cat(values)
