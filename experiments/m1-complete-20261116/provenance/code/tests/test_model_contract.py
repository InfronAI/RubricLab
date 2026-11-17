import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from openrubrics_lab.backend import LocalLM, token_logps


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3, 4, 5]


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
