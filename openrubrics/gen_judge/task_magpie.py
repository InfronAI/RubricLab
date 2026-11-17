import os
import json
import argparse
import asyncio
import random

from runner import JudgeGenerator
from prompter import make_judge_prompt_pairwise as make_judge_prompt

from pathlib import Path
script_path = Path(__file__)
script_stem = script_path.stem  
benchmark = script_stem.removeprefix('task_')
_ROOT = Path(__file__).resolve().parent.parent


def load_data(max_samples=None, cached_result_to_keep=None, input_path=None) -> list:
    fild_path = input_path or str(_ROOT / "rubrics/magpie/magpie-o4-mini.jsonl")
    
    if not os.path.exists(fild_path):
        raise FileNotFoundError(f"File not found: {fild_path}")
    
    cached_instruct = set()
    start_sample_id = 0
    if cached_result_to_keep is not None and os.path.exists(cached_result_to_keep):
        with open(cached_result_to_keep, "r") as f:
            for line in f:
                item = json.loads(line)
                instruction = item["instruction"]
                rubric = item["rubric"]
                key = (instruction, rubric)
                cached_instruct.add(key)
                start_sample_id += 1
        print(f"Loaded {len(cached_instruct)} cached instructions from {cached_result_to_keep}")

    data = []
    flip_responses = random.random() < 0.5
    # flip_responses = False
    with open(fild_path, "r") as f:
        for sample_id, line in enumerate(f):
            if max_samples is not None and sample_id >= max_samples:
                break
            item = json.loads(line)

            # Avoid duplicate generation for the same instruction + rubric pair.
            instruction = item["instruction"]
            rubric = item["rubric"]
            if (instruction, rubric) in cached_instruct:
                continue

            responses = item["responses"]
            scores = item["scores"]
            
            sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            item["responses"] = [responses[idx] for idx in sorted_indices]
            item["scores"] = [scores[idx] for idx in sorted_indices]

            # The responses have been sorted by socres. 
            for i in range(len(item["responses"]) - 1):
                for j in range(i+1, len(item["responses"])):
                    # Avoid duplicate responses.
                    if item["responses"][i] == item["responses"][j]:
                        continue
                    
                    # Avoid tie cases.
                    if item["scores"][i] == item["scores"][j]:
                        continue

                    new_item = {
                        "instruction": item["instruction"],
                        "rubric": item["rubric"],
                        "responses": [item["responses"][i], item["responses"][j]],
                        "scores": [item["scores"][i], item["scores"][j]],
                    }
                    flip_responses = random.random() < 0.5
                    # flip_responses = not flip_responses

                    if flip_responses:
                        new_item["responses"].reverse()
                        new_item["scores"].reverse()                        
                    data.append(new_item)

    return data, f"{benchmark}_pairwise", start_sample_id


async def main():
    parser = argparse.ArgumentParser(description=f"judge generation runner")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument('--model_name', type=str, default='o4-mini', help='OpenAI model name')
    parser.add_argument("--concurrency_limit", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=4096)
    
    parser.add_argument("--filter_by_acc", type=float, default=0.5, help="Filter results by accuracy threshold, between 0 and 1")
    parser.add_argument("--save_raw_output", type=int, default=1)

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--dryrun", type=int, default=0, help="Dryrun to test the pipeline")
    parser.add_argument("--refresh", type=int, default=0, help="refresh the cached results")
    args = parser.parse_args()
    random.seed(args.seed)

    runner = JudgeGenerator(
        benchmark_slug=benchmark.lower(),
        load_dataset=lambda **kwargs: load_data(**kwargs, input_path=args.input_path),
        make_judge_prompt=make_judge_prompt,
    )
    await runner.run(args)

if __name__ == "__main__":
    asyncio.run(main())

