import os 
import datetime
import asyncio
import json 
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio

from api import generate_response
from utils import default_results_path, parse_winner 

class JudgeGenerator:
    def __init__(self, benchmark_slug, load_dataset, make_judge_prompt):
        self.load_dataset = load_dataset
        self.benchmark_slug = benchmark_slug.lower()
        self.make_judge_prompt = make_judge_prompt

    async def run(self, args) -> None:
        if args.max_samples is None or args.max_samples <= 0:
            args.max_samples = None
        if args.dryrun:
            args.max_samples = min(args.max_samples or 5, 5)
        if args.output_file is None:
            output_file = default_results_path(dataset_slug=self.benchmark_slug, model_name=args.model_name)
        else:
            output_file = args.output_file

        print(f"Output file: {output_file}, max_samples: {args.max_samples}")
        if args.dryrun or args.save_raw_output:
            raw_output_file = output_file.replace(".jsonl", "_raw.jsonl")
        else:
            raw_output_file = None        

        if not args.refresh:
            cached_result_to_keep = raw_output_file or output_file
        else:
            cached_result_to_keep = None
        data, dataname, start_sample_id = self.load_dataset(max_samples=args.max_samples, cached_result_to_keep=cached_result_to_keep)

        # apply prompt formatting
        for item in data:
            item["prompt"] = self.make_judge_prompt(
                instruction=item.get("instruction", ""),
                rubric=item.get("rubric", ""),
                responses=item.get("responses", ""),
            )

        with open(output_file, "w") as f:
            meta = {
                "judge_generation_info": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "total_samples": len(data),
                    "model": args.model_name,
                    "dataset": dataname,
                }
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        result = await self.perform_judge_generations(
            data_samples=data,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            concurrency_limit=args.concurrency_limit,
            start_sample_id=start_sample_id,
        )

        clean_result = []
        if cached_result_to_keep is not None and os.path.exists(cached_result_to_keep):
            print(f"Appending to existing cached results from {cached_result_to_keep}")
            with open(cached_result_to_keep, "r") as f:
                for line in f:
                    item = json.loads(line)
                    clean_result.append(item)

        for i, item in enumerate(data):
            i = i + start_sample_id
            r = next((r for r in result if r and r["sample_id"] == i), None)
            out = {
                "sample_id": i,
                "instruction": item["instruction"],
                "responses": item["responses"],
                "scores": item.get("scores", []),
                "rubric": item["rubric"],
                "judge": r["text"] if r else "",
            }
            clean_result.append(out)

        if raw_output_file is not None:
            with open(raw_output_file, "w") as f:
                for out in clean_result:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            
            print(f"Raw results written to:\n{raw_output_file}")

        if 0 < args.filter_by_acc < 1:
            clean_result = self.filter_by_acc(clean_result)
        
        # Anyway, keep only correct judges
        final_result = []
        for r in clean_result:
            judge = r["judge"]
            if not judge:
                continue
            winner = parse_winner(judge)
            label = "a" if r["scores"][0] > r["scores"][1] else "b"
            if winner.lower() == label.lower():
                final_result.append(r)

        with open(output_file, "w") as f:
            for out in final_result:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        
        print(f"Results written to:\n{output_file}")

    async def perform_judge_generations(
        self,
        data_samples: List[Dict[str, Any]],
        model_name: str, 
        max_tokens: int,
        concurrency_limit: int = 10,
        start_sample_id: int = 0,
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
                    sample_id=i+start_sample_id,
                )
            )
        
        results = await tqdm_asyncio.gather(*tasks, desc="Generating judges", total=len(tasks))
        return results

    def filter_by_acc(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        prompts = {}        
        for r in results:
            key = (r["instruction"], r["rubric"])
            if key not in prompts:
                prompts[key] = []
            judge = r["judge"]
            if not judge:
                prompts[key].append(0.)
            else:            
                winner = parse_winner(judge)
                label = "a" if r["scores"][0] > r["scores"][1] else "b"
                prompts[key].append(1. if winner.lower() == label.lower() else 0.)

        valid_prompts = {k for k, v in prompts.items() if sum(v)/len(v) >= 0.5}

        filtered_results = []
        for r in results:
            judge = r["judge"]
            if not judge:
                continue

            if (r["instruction"], r["rubric"]) in valid_prompts:
                filtered_results.append(r)
        
        return filtered_results
    