import datetime
import argparse
import asyncio
import json 
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio

from api import generate_response
from utils import default_results_path

class RubricGenerator:
    def __init__(self, benchmark_slug, load_dataset, make_rubric_prompt):
        self.load_dataset = load_dataset
        self.benchmark_slug = benchmark_slug.lower()
        self.make_rubric_prompt = make_rubric_prompt

    async def run(self, args) -> None:
        if args.max_samples is None or args.max_samples <= 0:
            args.max_samples = None
        if args.dryrun:
            args.max_samples = min(args.max_samples or 5, 5)

        data, dataname = self.load_dataset(max_samples=args.max_samples)

        # apply prompt formatting
        for item in data:
            item["prompt"] = self.make_rubric_prompt(
                request=item.get("request", ""),
                responses=item.get("responses", ""),
                context=item.get("context", ""),
            )


        if args.output_file is None:
            output_file = default_results_path(dataset_slug=self.benchmark_slug, model_name=args.model_name)
        else:
            output_file = args.output_file
        print(f"Output file: {output_file}, max_samples: {args.max_samples}")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_file) + ".metadata.json", "w") as f:
            meta = {
                "rubric_generation_info": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "total_samples": len(data),
                    "model": args.model_name,
                    "dataset": dataname,
                }
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        result = await self.perform_rubric_generations(
            data_samples=data,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            concurrency_limit=args.concurrency_limit,
        )

        with open(output_file, "w") as f:
            for i, item in enumerate(data):
                r = next((r for r in result if r and r["sample_id"] == i), None)
                out = {
                    "sample_id": i,
                    "instruction": item["request"],
                    "responses": item["responses"],
                    "scores": item.get("scores", []),
                    "rubric": r["text"] if r else "",
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        
        print(f"Results written to:\n{output_file}")

    async def perform_rubric_generations(
        self,
        data_samples: List[Dict[str, Any]],
        model_name: str, 
        max_tokens: int,
        concurrency_limit: int = 10
    ) -> List[Dict[str, Any]]:

        # clean up old client to avoid possible model_name mismatch
        global client 
        client = None 

        semaphore = asyncio.Semaphore(concurrency_limit)

        tasks = []
        for i, s in enumerate(data_samples):
            tasks.append(
                generate_response(
                    user_prompt=s["prompt"],
                    model_name=model_name,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                    sample_id=i,
                )
            )
        
        results = await tqdm_asyncio.gather(*tasks, desc="Generating rubrics", total=len(tasks))
        return results

