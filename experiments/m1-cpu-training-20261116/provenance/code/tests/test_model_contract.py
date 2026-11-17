import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from openrubrics_lab.backend import LocalLM, token_logps


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3, 4, 5]


def test_apple_silicon_uses_verified_cpu_autograd_and_keeps_mps_inference(monkeypatch):
    from openrubrics_lab.backend import resolve_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto", trainable=True) == "cpu"
    assert resolve_device("mps", trainable=True) == "cpu"
    assert resolve_device("auto", trainable=False) == "mps"
    assert resolve_device("cpu", trainable=True) == "cpu"
    assert resolve_device("cuda", trainable=True) == "cuda"
    assert resolve_device("mps", trainable=True, allow_unverified_mps_backward=True) == "mps"


def test_completion_mask_and_no_silent_truncation():
    lm = object.__new__(LocalLM)
    lm.device, lm.max_context, lm.tokenizer = "cpu", 20, TinyTokenizer()
    lm.prompt_ids = lambda _: [1, 6, 7]
    ids, labels, prefix = lm.encode("prompt", "answer")
    assert labels[0, :prefix].tolist() == [-100]*3
    assert labels[0, prefix:].tolist() == [3, 4, 5, 2]
    assert ids[0, -1] == 2
    lm.max_context = 4
    with pytest.raises(ValueError, match="exceeds context"):
        lm.encode("prompt", "answer")


def test_suffix_log_probs_match_full_cross_entropy_and_backpropagate():
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    ids = torch.tensor([[1, 3, 4, 5, 2]])
    labels = ids.clone()
    labels[:, :3] = -100
    loss = model(ids, labels=labels).loss
    logps = token_logps(model, ids, 3)
    assert logps.shape == (2,)
    assert torch.allclose(-logps.mean(), loss, atol=1e-6)
    (-logps.mean()).backward()
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_mps_sampling_failure_retries_on_cpu_and_records_reason(monkeypatch):
    from openrubrics_lab import backend

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def decode(self, tokens, skip_special_tokens=True):
            return "a real retry"

    class Model:
        calls = 0

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            assert not hasattr(self, "_cache"), "Static KV tensors must not survive the device switch"
            return self

        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self._cache = object()
                raise RuntimeError("probability tensor contains either inf, nan or element < 0")
            assert kwargs["cache_implementation"] == "dynamic"
            return torch.LongTensor([[1, 3, 4, 2]])

    lm = object.__new__(LocalLM)
    lm.device, lm.max_context, lm.tokenizer, lm.model = "mps", 20, Tokenizer(), Model()
    lm.prompt_ids = lambda _: [1, 3]
    original_tensor = torch.tensor
    monkeypatch.setattr(backend.torch, "tensor", lambda x, **kw: original_tensor(x, device="cpu"))
    monkeypatch.setattr(backend, "release_memory", lambda: None)
    monkeypatch.setattr(backend, "seed_all", lambda _: None)
    result = lm.generate("request", max_new_tokens=4)[0]
    assert lm.model.calls == 2
    assert result["device"] == "cpu"
    assert result["requested_device"] == "mps"
    assert "probability tensor" in result["fallback_reason"]
    assert result["text"] == "a real retry"


def test_mps_inference_coordinator_does_not_allocate_model(monkeypatch):
    from openrubrics_lab import backend

    class Tokenizer:
        pad_token_id = 0

    def forbidden_load(*args, **kwargs):
        raise AssertionError("The coordinator must not retain model weights or Metal graph caches")

    monkeypatch.delenv("OPENRUBRICS_INFERENCE_WORKER", raising=False)
    monkeypatch.setattr(backend, "resolve_device", lambda *args, **kwargs: "mps")
    monkeypatch.setattr(backend.torch.mps, "set_per_process_memory_fraction", lambda _: None)
    monkeypatch.setattr(backend.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(backend.AutoModelForCausalLM, "from_pretrained", forbidden_load)
    with LocalLM("test/model", revision="pinned") as lm:
        assert lm.isolated_inference
        assert not hasattr(lm, "model")
        assert lm.worker_model_args["revision"] == "pinned"


def test_static_cache_preserves_greedy_generation():
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        eos_token_id=2, pad_token_id=0)).eval()
    ids = torch.tensor([[1, 3, 4]])
    kwargs = {"attention_mask": torch.ones_like(ids), "max_new_tokens": 6, "do_sample": False}
    dynamic = model.generate(ids, cache_implementation="dynamic", **kwargs)
    static = model.generate(ids, cache_implementation="static", disable_compile=True, **kwargs)
    assert torch.equal(dynamic, static)
