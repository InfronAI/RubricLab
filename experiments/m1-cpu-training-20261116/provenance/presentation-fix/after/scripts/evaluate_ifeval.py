"""Generate policy responses and run the pinned official strict/loose IFEval verifiers."""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))


def main():
    from instruction_following_eval import evaluation_lib
    from langdetect import DetectorFactory

    from openrubrics_lab.backend import LocalLM
    from openrubrics_lab.config import lm_kwargs, load_config
    from openrubrics_lab.io import append_jsonl, environment, file_hash, write_json, write_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "vendor/instruction_following_eval/data/input_data.jsonl")
    args = parser.parse_args()
    if args.samples < 1 or args.max_tokens < 1:
        parser.error("samples and max-tokens must be positive")
    if args.output.exists():
        parser.error("Output exists; use a new path to preserve results")
    config = load_config(args.config)
    DetectorFactory.seed = config["seed"]
    inputs = evaluation_lib.read_prompt_list(str(args.input))
    random.Random(config["seed"]).shuffle(inputs)
    args.output.mkdir(parents=True)
    selected, responses = [], {}
    with LocalLM(**lm_kwargs(config, args.adapter)) as lm:
        for item in inputs:
            if len(lm.prompt_ids(item.prompt))+args.max_tokens > config["max_context"]:
                continue
            generation = lm.generate(item.prompt, max_new_tokens=args.max_tokens, temperature=0,
                                     seed=config["seed"])[0]
            responses[item.prompt] = generation["text"]
            selected.append(item)
            append_jsonl(args.output / "responses.jsonl", {"key": item.key, "prompt": item.prompt,
                         "response": generation["text"], "generation": generation})
            if len(selected) >= args.samples:
                break
    if len(selected) < args.samples:
        raise ValueError("Insufficient context-fitting IFEval prompts")
    write_jsonl(args.output / "inputs.jsonl", [dataclasses.asdict(i) for i in selected])
    summary = {"n": len(selected), "model": config["model"], "adapter": str(args.adapter),
               "seed": config["seed"], "max_tokens": args.max_tokens, "environment": environment(),
               "verifier": json.loads((ROOT / "vendor/instruction_following_eval/PROVENANCE.json").read_text()),
               "input_sha256": file_hash(args.input),
               "scope": "Bounded official-verifier evaluation; not the full IFEval benchmark"}
    for mode, fn in (("strict", evaluation_lib.test_instruction_following_strict),
                     ("loose", evaluation_lib.test_instruction_following_loose)):
        outputs = [fn(item, responses) for item in selected]
        write_jsonl(args.output / f"{mode}.jsonl", [dataclasses.asdict(o) for o in outputs])
        checks = [x for o in outputs for x in o.follow_instruction_list]
        summary[mode] = {"prompt_accuracy": sum(o.follow_all_instructions for o in outputs)/len(outputs),
                         "instruction_accuracy": sum(checks)/len(checks), "instructions": len(checks)}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
