"""CPU autograd and isolated MPS inference on M1, explicit bounds, LoRA adapters."""
from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
import tempfile
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


def resolve_device(device="auto", *, trainable=False, allow_unverified_mps_backward=False):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable. Use an arm64 Python and macOS 12.3+, or --device cpu.")
    if device not in ("mps", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device}")
    # Real fixed-weight replay and finite differences found incorrect MPS
    # backward gradients on the tested M1 stack, despite finite losses. Keep
    # GPU forward inference, but use verified CPU autograd for all training and
    # reference/old-policy scoring. The override exists only for numerical audits.
    if device == "mps" and trainable and not allow_unverified_mps_backward:
        return "cpu"
    return device


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LocalLM:
    def __init__(self, model: str, *, revision: str | None = None, device="auto", adapter=None,
                 trainable=False, max_context=2048, lora_rank=8, offline=False,
                 allow_unverified_mps_backward=False):
        self.requested_device = resolve_device(device)
        self.device = resolve_device(device, trainable=trainable,
                                     allow_unverified_mps_backward=allow_unverified_mps_backward)
        self.trainable = trainable
        if self.requested_device == "mps" and self.device == "cpu":
            print("Apple Silicon training/scoring uses CPU autograd; MPS remains available for inference.",
                  flush=True)
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
        # Metal graph caches survive model deletion and empty_cache(). Keep the
        # coordinator free of model weights; each inference batch exits its own
        # process. Training already has a separate worker and needs direct access.
        self.isolated_inference = (self.device == "mps" and not trainable and
                                   os.environ.get("OPENRUBRICS_INFERENCE_WORKER") != "1")
        if self.isolated_inference:
            self.worker_model_args = {"model": model, "revision": revision, "device": self.device,
                                      "adapter": str(adapter) if adapter else None,
                                      "max_context": max_context, "offline": offline}
            return
        # FP32 + eager attention avoid M1 SDPA/FP16 non-finite gradients and
        # sampling probabilities observed during long, repeated model transitions.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, local_files_only=offline,
            dtype=torch.float32 if trainable or self.device in ("cpu", "mps") else torch.float16,
            attn_implementation="eager" if self.device == "mps" else "sdpa",
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
        if getattr(self, "isolated_inference", False):
            return self._generate_isolated(prompt, max_new_tokens, temperature, seed, n)
        result = []
        self.model.eval()
        seed_all(seed)
        ids = torch.tensor([prefix]*n, device=self.device)
        started = time.perf_counter()
        kwargs = {"temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature > 0 else {}
        if self.device == "mps":
            # Fixed KV shapes avoid a new Metal graph for every decoding length.
            # torch.compile is not needed (or supported by this M1 path).
            kwargs.update(cache_implementation="static", disable_compile=True)
        requested_device, fallback_reason = self.device, None

        def sample():
            with torch.inference_mode():
                return self.model.generate(
                    ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, use_cache=True, **kwargs,
                )[:, len(prefix):].tolist()

        try:
            batch = sample()
        except RuntimeError as error:
            message = str(error).lower()
            if self.device != "mps" or not any(t in message for t in
                    ("probability tensor", "not implemented for", "mps backend out of memory")):
                raise
            # Retry the same model, prompt and seed on CPU. Never mask NaNs, invent
            # a decision or drop a failed sample. CPU failure still propagates.
            fallback_reason = str(error)
            print(f"MPS generation fallback to CPU: {fallback_reason}", flush=True)
            # StaticCache is not a registered module buffer: model.to() does not
            # migrate it. Discard the failed device's cache before the CPU retry.
            causal = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
            if hasattr(causal, "_cache"):
                del causal._cache
            kwargs["cache_implementation"] = "dynamic"
            self.model.to("cpu")
            self.device = "cpu"
            ids = ids.cpu()
            release_memory()
            seed_all(seed)
            batch = sample()
        elapsed = time.perf_counter()-started
        for generated in batch:
            if self.tokenizer.eos_token_id in generated:
                generated = generated[:generated.index(self.tokenizer.eos_token_id)+1]
            result.append({"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                           "token_ids": generated, "prompt_tokens": len(prefix),
                           "tokens": len(generated), "seconds": elapsed/n,
                           "batch_seconds": elapsed, "batch_size": n, "seed": seed,
                           "device": self.device, "requested_device": requested_device,
                           "fallback_reason": fallback_reason,
                           "tokens_per_second": len(generated)/max(elapsed/n, 1e-9),
                           "hit_token_limit": len(generated) == max_new_tokens and
                           generated[-1] != self.tokenizer.eos_token_id})
        return result

    def _generate_isolated(self, prompt, max_new_tokens, temperature, seed, n):
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="openrubrics-inference-") as temporary:
            request, response = Path(temporary) / "request.json", Path(temporary) / "response.json"
            request.write_text(json.dumps({"model": self.worker_model_args,
                "generation": {"prompt": prompt, "max_new_tokens": max_new_tokens,
                               "temperature": temperature, "seed": seed, "n": n},
                "response": str(response)}, ensure_ascii=False))
            subprocess.run([sys.executable, "-m", "openrubrics_lab.inference_worker", str(request)],
                           env={**os.environ, "OPENRUBRICS_INFERENCE_WORKER": "1"}, check=True)
            payload = json.loads(response.read_text())
        if "error" in payload:
            raise ValueError(payload["error"])
        for item in payload["generations"]:
            item["worker_wall_seconds"] = time.perf_counter()-started
            item["process_isolated"] = True
        return payload["generations"]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        from .io import write_json
        write_json(path / "execution_device.json", {
            "requested_device": self.requested_device, "actual_device": self.device,
            "trainable": self.trainable, "dtype": str(next(self.model.parameters()).dtype),
            "apple_silicon_policy": "CPU autograd; MPS forward inference only",
        })

    def close(self):
        if hasattr(self, "model"):
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
