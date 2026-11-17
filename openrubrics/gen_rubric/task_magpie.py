import json
import argparse
import asyncio
import os

from runner import RubricGenerator
from prompter import make_rubric_prompt_listwise as make_rubric_prompt

from pathlib import Path
script_path = Path(__file__)
script_stem = script_path.stem  
benchmark = script_stem.removeprefix('task_')


def load_data(max_samples=None, input_path=None) -> list:
    fild_path = input_path or os.getenv("MAGPIE_INPUT")
    if not fild_path:
        raise ValueError("Provide --input_path or set MAGPIE_INPUT to a listwise JSONL file")
    
    data = []
    with open(fild_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            item = json.loads(line)
            
            responses = item.get("responses", [])
            scores = item.get("scores", [])
            if len(responses) < 2 or len(responses) != len(scores):
                raise ValueError(f"Line {i + 1}: responses and scores need equal lengths >= 2")
            
            sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            item["responses"] = [responses[idx] for idx in sorted_indices]
            item["scores"] = [scores[idx] for idx in sorted_indices]
            item["request"] = item.get("input", item.get("instruction", item.get("request")))
            if not item["request"]:
                raise ValueError(f"Line {i + 1}: missing input/instruction/request")
            item.pop("input", None)

            data.append(item)

    return data, "magpie_listwise"


async def main():
    parser = argparse.ArgumentParser(description=f"rubric generation runner")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--input_path", default=os.getenv("MAGPIE_INPUT"))

    parser.add_argument('--model_name', type=str, default='o4-mini', help='OpenAI model name')
    parser.add_argument("--concurrency_limit", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--dryrun", type=int, default=0, help="Dryrun to test the pipeline")
    args = parser.parse_args()

    runner = RubricGenerator(
        benchmark_slug=benchmark.lower(),
        load_dataset=lambda max_samples: load_data(max_samples, args.input_path),
        make_rubric_prompt=make_rubric_prompt,
    )
    await runner.run(args)

if __name__ == "__main__":
    asyncio.run(main())

