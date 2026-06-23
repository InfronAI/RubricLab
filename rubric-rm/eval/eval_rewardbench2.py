import os 
import gc
from datetime import datetime
import torch
from base_judge import Judge, APIJudge
from base_rubric import Rubric, APIRubric
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from common import resolve_models, is_api_model


def _format_timestamp_suffix(dt: datetime) -> str:
    """Return a suffix like '20250908_142233' for filenames."""
    return dt.strftime("%Y%m%d_%H%M%S")


def _make_api_rubric(model_name: str, max_tokens: int = 4096, concurrency: int = 10):
    """
    Compatible with the various possible APIRubric __init__ signatures:
    - APIRubric(model_name, tokenizer, sp, ...)
    - APIRubric(model=model_name, ...)
    """
    try:
        return APIRubric(model_name, None, None, max_tokens=max_tokens, concurrency=concurrency)
    except TypeError:
        try:
            return APIRubric(model=model_name, max_tokens=max_tokens, concurrency=concurrency)
        except TypeError:
            return APIRubric(model_name, None, None)


def _make_api_judge(model_name: str, max_tokens: int = 4096, concurrency: int = 10):
    """
    Same as above; compatible with the various APIJudge __init__ signatures.
    """
    try:
        return APIJudge(model_name, None, None, default_max_tokens=max_tokens, concurrency=concurrency)
    except TypeError:
        try:
            return APIJudge(model=model_name, default_max_tokens=max_tokens, concurrency=concurrency)
        except TypeError:
            return APIJudge(model_name, None, None)


def main():
    from pathlib import Path
    script_path = Path(__file__)
    script_stem = script_path.stem  
    benchmark = script_stem.removeprefix('eval_')

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, default=f"./outputs/ours[{{size}}B]/{benchmark}")
    parser.add_argument("--rubric-model", type=str, default=None,
                        help="Override rubric model (HF repo, local path, or API name).")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model (HF repo, local path, or API name).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=str, default="8")
    args = parser.parse_args()

    rubric_model, judge_model = resolve_models(args.size, args.rubric_model, args.judge_model)

    args.outfile = args.outfile.format(size=args.size)
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    rubric_backend = "api" if is_api_model(rubric_model) else "vllm"
    judge_backend = "api" if is_api_model(judge_model) else "vllm"
    print(f"[Backend] rubric={rubric_backend} ({rubric_model}) | judge={judge_backend} ({judge_model})")

    # =========================
    # 2) Init Rubric generator
    # =========================
    tokenizer = None
    rubric_core = None
    rubric_maker = None

    if rubric_backend == "vllm":
        tokenizer = AutoTokenizer.from_pretrained(rubric_model, trust_remote_code=True)
        rubric_core = LLM(
            model=rubric_model,
            trust_remote_code=True,
            seed=args.seed,
        )
        sp_rubric = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
        )
        rubric_maker = Rubric(rubric_core, tokenizer, sp_rubric)
    else:
        # API mode: use rubric_model directly as the model_name (e.g. "gpt-4.1"/"gemini"/"deepseek")
        rubric_maker = _make_api_rubric(rubric_model, max_tokens=4096, concurrency=10)

    rubric_all = {}

    # =========================
    # 3) Load benchmark and generate rubrics
    # =========================
    ds_full = load_dataset("allenai/reward-bench-2", split="test")
    subfiles = ["Precise IF", "Focus"]

    output_files = {}
    for subset in subfiles:
        print(f"--- Processing subset: {subset} ---")

        save_dt = datetime.now()
        suffix = _format_timestamp_suffix(save_dt)
        outfile_path = args.outfile + f"_{subset}_{suffix}.json"
        output_files[subset] = outfile_path
        print("Expected outfile:", outfile_path)

        ds_subset = ds_full.filter(lambda ex: subset in ex["subset"])
        prompts = []
        labels = []

        ds = []
        for item in ds_subset:
            for i in range(len(item['chosen'])):
                for j in range(len(item['rejected'])):
                    chosen = item['chosen'][i]
                    rejected = item['rejected'][j]
                    ds.append({
                        "id": item['id']+f"_{i}_{j}",
                        "prompt": item['prompt'],
                        "chosen": chosen,
                        "rejected": rejected, 
                    })

        # Data preparation logic remains the same
        for idx, ex in enumerate(ds):
            prompt_data = {
                "id": ex["id"],
                "prompt": ex['prompt'],
                "A": ex["chosen"],
                "B": ex["rejected"]
            }
            if idx % 2 == 0:
                prompts.append(prompt_data)
                labels.append("A")
            else:
                # Swap chosen and rejected for the second case
                prompt_data["A"], prompt_data["B"] = prompt_data["B"], prompt_data["A"]
                prompts.append(prompt_data)
                labels.append("B")

        ins = [pr["prompt"] for pr in prompts]
        resa = [pr["A"] for pr in prompts]
        resb = [pr["B"] for pr in prompts]

        rubrics = rubric_maker.gen_rubric(ins)
        rubric_all[subset] = rubrics

    # cleanup rubric vLLM
    if rubric_backend == "vllm":
        del rubric_maker, rubric_core
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # nothing to free on the API path (no GPU memory)
        del rubric_maker

    # =========================
    # 4) Init Judge
    # =========================
    judge_maker = None

    if judge_backend == "vllm":
        # load the judge model's own tokenizer (don't reuse the rubric tokenizer)
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)

        sp = SamplingParams(
            max_tokens=4096,
            temperature=0.7,
        )
        judge_core = LLM(
            model=judge_model,
            trust_remote_code=True,
            seed=args.seed,
        )
        judge_maker = Judge(judge_core, judge_tokenizer, sp)
    else:
        # API mode: use judge_model directly as the model_name
        judge_maker = _make_api_judge(judge_model, max_tokens=4096, concurrency=10)

    # =========================
    # 5) Judge loop
    # =========================
    for subset in subfiles:
        print(f"--- Processing subset: {subset} ---")
        if subset not in rubric_all:
            continue

        outfile_path = output_files[subset]
        rubrics = rubric_all[subset]

        ds_subset = ds_full.filter(lambda ex: subset in ex["subset"])
        prompts = []
        labels = []

        ds = []
        for item in ds_subset:
            for i in range(len(item['chosen'])):
                for j in range(len(item['rejected'])):
                    chosen = item['chosen'][i]
                    rejected = item['rejected'][j]
                    ds.append({
                        "id": item['id']+f"_{i}_{j}",
                        "prompt": item['prompt'],
                        "chosen": chosen,
                        "rejected": rejected, 
                    })

        # Data preparation logic remains the same
        for idx, ex in enumerate(ds):
            prompt_data = {
                "id": ex["id"],
                "prompt": ex['prompt'],
                "A": ex["chosen"],
                "B": ex["rejected"]
            }
            if idx % 2 == 0:
                prompts.append(prompt_data)
                labels.append("A")
            else:
                # Swap chosen and rejected for the second case
                prompt_data["A"], prompt_data["B"] = prompt_data["B"], prompt_data["A"]
                prompts.append(prompt_data)
                labels.append("B")

        ins = [pr["prompt"] for pr in prompts]
        resa = [pr["A"] for pr in prompts]
        resb = [pr["B"] for pr in prompts]

        preds = []
        n = 5
        for _ in range(n):
            judges = judge_maker.gen_judge(ins, rubrics, resa, resb)
            preds.append(judges)

        # Process the results after all tasks are done
        results = []
        results.append({
            "rubric": rubric_model,
            "judge": judge_model,
        })
        for i, rubric in enumerate(rubrics):
            pr = prompts[i]
            prediction = [preds[j][i] for j in range(n)]
            results.append({
                "id": pr["id"],
                "prompt": pr["prompt"],
                "label": labels[i],
                "prediction": prediction,
                "rubric": rubric,
                "response_A": pr["A"],
                "response_B": pr["B"],
            })

        with open(outfile_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} examples to {outfile_path}")

    # cleanup judge vLLM
    if judge_backend == "vllm":
        del judge_maker
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()