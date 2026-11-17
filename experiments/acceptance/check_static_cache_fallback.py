"""Hardware fault injection with tiny random models, never benchmark measurements."""
import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from openrubrics_lab.backend import LocalLM
from openrubrics_lab.io import write_json


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):
        return str(tokens)


def check(wrapped):
    torch.manual_seed(42)
    lm = object.__new__(LocalLM)
    lm.device, lm.max_context, lm.tokenizer = 'mps', 16, Tokenizer()
    lm.prompt_ids = lambda _: [1, 3]
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, eos_token_id=2, pad_token_id=0))
    if wrapped:
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    lm.model = model.to('mps').eval()
    original, calls = lm.model.generate, [0]

    def injected(*args, **kwargs):
        calls[0] += 1
        result = original(*args, **kwargs)
        if calls[0] == 1:
            raise RuntimeError('probability tensor injected failure after real static cache allocation')
        return result

    lm.model.generate = injected
    try:
        output = lm.generate('fault injection', max_new_tokens=2, temperature=0)
        assert calls[0] == 2 and output[0]['device'] == 'cpu'
        assert output[0]['requested_device'] == 'mps'
        result = {'status': 'passed', 'peft_wrapped': wrapped, 'calls': calls[0], 'output': output}
    except RuntimeError as error:
        result = {'status': 'failed', 'peft_wrapped': wrapped, 'calls': calls[0], 'error': str(error)}
    finally:
        lm.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [check(False), check(True)]
    write_json(args.output, {'purpose': __doc__, 'cases': results})
    print([(r['peft_wrapped'], r['status']) for r in results])
    assert all(r['status'] == 'passed' for r in results)


if __name__ == '__main__':
    main()
