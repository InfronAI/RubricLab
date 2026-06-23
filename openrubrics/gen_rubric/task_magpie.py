import json
import argparse
import asyncio

from runner import RubricGenerator
from prompter import make_rubric_prompt_listwise as make_rubric_prompt

from pathlib import Path
script_path = Path(__file__)
script_stem = script_path.stem  
benchmark = script_stem.removeprefix('task_')


def load_data(max_samples=None) -> list:
    fild_path = \
        "PATH/TO/MAGPIE_LISTWISE.jsonl"  # placeholder: raw data not bundled (see README)
    
    data = []
    with open(fild_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            item = json.loads(line)
            
            responses = item.get("responses", [])
            scores = item.get("scores", [])
            
            sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            item["responses"] = [responses[idx] for idx in sorted_indices]
            item["scores"] = [scores[idx] for idx in sorted_indices]
            item["request"] = item["input"]
            item.pop("input", None)

            data.append(item)

    return data, "magpie_listwise"


async def main():
    parser = argparse.ArgumentParser(description=f"rubric generation runner")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument('--model_name', type=str, default='o4-mini', help='OpenAI model name')
    parser.add_argument("--concurrency_limit", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--dryrun", type=int, default=0, help="Dryrun to test the pipeline")
    args = parser.parse_args()

    runner = RubricGenerator(
        benchmark_slug=benchmark.lower(),
        load_dataset=load_data,
        make_rubric_prompt=make_rubric_prompt,
    )
    await runner.run(args)

if __name__ == "__main__":
    asyncio.run(main())

