"""Scientific stages. Gold labels are never passed to an inference prompt."""
from __future__ import annotations

import copy
from pathlib import Path

from .backend import LocalLM
from .config import lm_kwargs
from .io import append_jsonl, digest, write_json, write_jsonl
from .metrics import format_reward, parse_winner, summarize
from .prompts import contrastive_prompt, direct_prompt, judge_prompt, rubric_prompt
from .training import dpo, grpo


def _generate(lm, prompt, config, role, seed, n=1, temperature=0.):
    return lm.generate(prompt, max_new_tokens=config[f"{role}_tokens"], temperature=temperature, seed=seed, n=n)


def curate(config, rows, output: Path):
    """Run actual CRG and consistency filtering; never replace failed outputs with labels."""
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    with LocalLM(**lm_kwargs(config)) as lm:
        for i, row in enumerate(rows[:config["curation_samples"]]):
            ordered = [row["response_a"], row["response_b"]]
            if row["label"] == "B":
                ordered.reverse()
            prompt = contrastive_prompt(row["instruction"], ordered, config["prompt_style"])
            crg = _generate(lm, prompt, config, "rubric", config["seed"]+i)[0]
            judge = _generate(lm, judge_prompt(row["instruction"], crg["text"], row["response_a"],
                              row["response_b"], config["prompt_style"]), config, "judge", config["seed"]+i)[0]
            valid_rubric = "[Hard Rule]" in crg["text"] or "[Principle]" in crg["text"]
            kept = valid_rubric and parse_winner(judge["text"]) == row["label"]
            record = {**row, "rubric": crg["text"], "judge": judge["text"], "kept": kept,
                      "rubric_generation": crg, "judge_generation": judge,
                      "reason": "consistent" if kept else "invalid_rubric_or_inconsistent_judgment"}
            generated.append(record)
            append_jsonl(output / "raw.jsonl", record)
            print(f"CRG {i+1}: kept={kept}", flush=True)
    kept = [r for r in generated if r["kept"]]
    write_jsonl(output / "filtered.jsonl", kept)
    summary = {"attempted": len(generated), "retained": len(kept),
               "retention": len(kept)/len(generated) if generated else None,
               "teacher": config["model"],
               "sft_source": "Published, already curated OpenRubric-v2 subset; fresh local CRG is a separate experiment"}
    write_json(output / "summary.json", summary)
    return summary


def evaluate(config, rows, rubric_adapter, judge_adapter, output: Path, variant="rubric", swap=False):
    output.mkdir(parents=True, exist_ok=True)
    records = copy.deepcopy(rows)
    rubrics = {}
    if variant != "direct":
        with LocalLM(**lm_kwargs(config, rubric_adapter)) as lm:
            for i, row in enumerate(records):
                key = digest(row["instruction"])
                if key not in rubrics:
                    rubrics[key] = _generate(lm, rubric_prompt(row["instruction"], config["prompt_style"]),
                                            config, "rubric", config["seed"]+i)[0]
                row["rubric_generation"] = rubrics[key]
                row["rubric"] = rubrics[key]["text"]
        write_jsonl(output / "rubrics.jsonl", [{"prompt_hash": k, **v} for k, v in rubrics.items()])
    with LocalLM(**lm_kwargs(config, judge_adapter)) as lm:
        for i, row in enumerate(records):
            a, b = row["response_a"], row["response_b"]
            prompt = direct_prompt(row["instruction"], a, b) if variant == "direct" else judge_prompt(
                row["instruction"], row["rubric"], a, b, config["prompt_style"])
            try:
                predictions = _generate(lm, prompt, config, "judge", config["seed"]+i*101,
                                        n=config["votes"], temperature=0.7)
                row["predictions"] = [p["text"] for p in predictions]
                row["generation_stats"] = predictions
                if swap:
                    reverse = direct_prompt(row["instruction"], b, a) if variant == "direct" else judge_prompt(
                        row["instruction"], row["rubric"], b, a, config["prompt_style"])
                    reversed_pred = _generate(lm, reverse, config, "judge", config["seed"]+i*101)[0]
                    row["swapped_prediction"] = reversed_pred["text"]
                    original = parse_winner(row["predictions"][0])
                    reversed_label = parse_winner(reversed_pred["text"])
                    row["order_consistent"] = original is not None and reversed_label is not None and original != reversed_label
            except ValueError as error:
                row["predictions"] = [""] * config["votes"]
                row["generation_stats"] = []
                row["error"] = str(error)
                row["order_consistent"] = False
            append_jsonl(output / "predictions.jsonl", row)
            print(f"Eval {output.name} {i+1}/{len(records)} "
                  f"{[parse_winner(p) for p in row['predictions']]} gold={row['label']}", flush=True)
    summary = summarize(records)
    summary["by_subset"] = {subset: summarize([r for r in records if r.get("subset", "validation") == subset])
                            for subset in sorted({r.get("subset", "validation") for r in records})}
    if any("difficulty" in r for r in records):
        summary["by_difficulty"] = {d: summarize([r for r in records if r.get("difficulty") == d])
                                    for d in ("easy", "normal", "hard")}
    generations = [g for r in records for g in r["generation_stats"]]
    seconds = sum(g["seconds"] for g in generations)
    tokens = sum(g["tokens"] for g in generations)
    summary.update({"generation_seconds": seconds, "tokens": tokens,
                    "actual_devices": sorted({g.get("device", "unknown") for g in generations}),
                    "fallback_generations": sum(bool(g.get("fallback_reason")) for g in generations),
                    "tokens_per_second": tokens/seconds if seconds else None,
                    "length_limit_rate": sum(g["hit_token_limit"] for g in generations)/len(generations) if generations else None,
                    "rubric_unique_prompts": len(rubrics),
                    "rubric_generation_seconds": sum(r["seconds"] for r in rubrics.values()),
                    "order_consistency": sum(r.get("order_consistent", False) for r in records)/len(records) if swap else None})
    write_json(output / "summary.json", summary)
    return summary


def judge_phase(config, rows, rubric_adapter, judge_adapter, reference, output):
    prompts = []
    with LocalLM(**lm_kwargs(config, rubric_adapter)) as lm:
        for i, row in enumerate(rows):
            rubric = _generate(lm, rubric_prompt(row["instruction"], config["prompt_style"]), config,
                               "rubric", config["seed"]+i)[0]["text"]
            # Rebalance positions in each phase without reusing a gold-conditioned trajectory.
            item = copy.deepcopy(row)
            if i % 2 == 0:
                item["response_a"], item["response_b"] = item["response_b"], item["response_a"]
                item["label"] = "B" if item["label"] == "A" else "A"
            prompt = judge_prompt(item["instruction"], rubric, item["response_a"], item["response_b"], config["prompt_style"])
            prompts.append((item, rubric, prompt))
    groups = []
    with LocalLM(**lm_kwargs(config, judge_adapter)) as lm:
        for i, (row, rubric, prompt) in enumerate(prompts):
            completions = _generate(lm, prompt, config, "judge", config["seed"]+i*31,
                                    config["group_size"], 1.)
            group = []
            for completion in completions:
                acc = float(parse_winner(completion["text"]) == row["label"])
                fmt = format_reward(completion["text"], rubric)
                group.append({**completion, "prompt": prompt, "sample_id": row["id"], "rubric": rubric,
                              "accuracy_reward": acc, "format_reward": fmt,
                              "reward": acc+config["format_weight"]*fmt})
            groups.append(group)
    return grpo(config, groups, judge_adapter, reference, output)


def rubric_phase(config, rows, rubric_adapter, judge_adapter, reference, output):
    groups = []
    with LocalLM(**lm_kwargs(config, rubric_adapter)) as lm:
        for i, row in enumerate(rows):
            prompt = rubric_prompt(row["instruction"], config["prompt_style"])
            completions = _generate(lm, prompt, config, "rubric", config["seed"]+i*31,
                                    config["group_size"], 1.)
            groups.append([{**c, "prompt": prompt, "sample_id": row["id"]} for c in completions])
    with LocalLM(**lm_kwargs(config, judge_adapter)) as lm:
        for row, group in zip(rows, groups):
            for completion in group:
                prompt = judge_prompt(row["instruction"], completion["text"], row["response_a"],
                                      row["response_b"], config["prompt_style"])
                prediction = _generate(lm, prompt, config, "judge", config["seed"])[0]
                completion["judge_prediction"] = prediction["text"]
                completion["reward"] = float(parse_winner(prediction["text"]) == row["label"])
    return grpo(config, groups, rubric_adapter, reference, output)


def alternate(config, rows, rubric_initial, judge_initial, output: Path, variant="standard"):
    output.mkdir(parents=True, exist_ok=True)
    rubric, judge = rubric_initial, judge_initial
    schedule = ("rubric", "judge") if variant == "reverse" else ("judge", "rubric")
    local_config = {**config, "format_weight": 0.0} if variant == "no_format" else config
    history = []
    for cycle in range(config["arm_rounds"]):
        n = config["arm_samples_per_round"]
        part = rows[cycle*n:(cycle+1)*n]
        for phase in schedule:
            dest = output / f"round-{cycle+1}-{phase}"
            print(f"ARM {variant} round {cycle+1}: updating {phase}", flush=True)
            if phase == "judge":
                summary = judge_phase(local_config, part, rubric, judge, judge_initial, dest)
                judge = dest
            else:
                summary = rubric_phase(local_config, part, rubric, judge, rubric_initial, dest)
                rubric = dest
            history.append({"round": cycle+1, "phase": phase, "sample_ids": [r["id"] for r in part], **summary})
            write_json(output / "schedule.json", history)
    write_json(output / "checkpoints.json", {"rubric": str(rubric), "judge": str(judge)})
    return rubric, judge


def label_policy_pairs(config, rows, rubric_adapter, judge_adapter, output):
    """Filter preference pairs using both response orders, as in ARM §4.3."""
    output.mkdir(parents=True, exist_ok=True)
    rubrics = []
    with LocalLM(**lm_kwargs(config, rubric_adapter)) as lm:
        for i, row in enumerate(rows):
            rubrics.append(_generate(lm, rubric_prompt(row["instruction"], config["prompt_style"]), config,
                                     "rubric", config["seed"]+i)[0]["text"])
    kept, all_rows = [], []
    with LocalLM(**lm_kwargs(config, judge_adapter)) as lm:
        for row, rubric in zip(rows, rubrics):
            answers = [row["response_a"], row["response_b"]]
            judgments = []
            for a, b in (answers, answers[::-1]):
                prompt = judge_prompt(row["instruction"], rubric, a, b, config["prompt_style"])
                judgments.append(_generate(lm, prompt, config, "judge", config["seed"])[0]["text"])
            labels = [parse_winner(t) for t in judgments]
            consistent = labels[0] is not None and labels[1] is not None and labels[0] != labels[1]
            if consistent and answers[0] != answers[1]:
                idx = 0 if labels[0] == "A" else 1
                kept.append({"id": row["id"], "instruction": row["instruction"],
                             "chosen": answers[idx], "rejected": answers[1-idx], "label_source": "rubric_arm"})
            all_rows.append({**row, "rubric": rubric, "judgments": judgments, "consistent": consistent})
    write_jsonl(output / "raw.jsonl", all_rows)
    write_jsonl(output / "pairs.jsonl", kept)
    write_json(output / "summary.json", {"attempted": len(rows), "retained": len(kept)})
    return kept


def iterative_dpo(config, rows, rubric_adapter, judge_adapter, output):
    actor, history = None, []
    for cycle in range(config["iter_dpo_rounds"]):
        generated = []
        with LocalLM(**lm_kwargs(config, actor)) as lm:
            for i, row in enumerate(rows[:config["policy_samples"]]):
                pair = _generate(lm, row["instruction"], config, "policy", config["seed"]+cycle*100+i, 2, 1.)
                generated.append({"id": f"iter-{cycle+1}:{row['id']}", "instruction": row["instruction"],
                                  "response_a": pair[0]["text"], "response_b": pair[1]["text"]})
        pairs = label_policy_pairs(config, generated, rubric_adapter, judge_adapter, output / f"round-{cycle+1}-labels")
        dest = output / f"round-{cycle+1}-dpo"
        summary = dpo(config, pairs, actor, None, dest)
        if summary["steps"]:
            actor = dest
        history.append(summary)
    write_json(output / "summary.json", {"rounds": history, "final_adapter": str(actor) if actor else None})
    return actor


def online_policy(config, rows, rubric_adapter, judge_adapter, output):
    rows = rows[:config["policy_samples"]]
    rubrics, groups, baselines = [], [], []
    with LocalLM(**lm_kwargs(config, rubric_adapter)) as lm:
        for i, row in enumerate(rows):
            rubrics.append(_generate(lm, rubric_prompt(row["instruction"], config["prompt_style"]), config,
                                     "rubric", config["seed"]+i)[0]["text"])
    with LocalLM(**lm_kwargs(config)) as lm:
        for i, row in enumerate(rows):
            prompt = row["instruction"]
            baselines.append(_generate(lm, prompt, config, "policy", config["seed"]+i)[0]["text"])
            group = _generate(lm, prompt, config, "policy", config["seed"]+i*31, config["group_size"], 1.)
            groups.append([{**c, "prompt": prompt, "sample_id": row["id"]} for c in group])
    with LocalLM(**lm_kwargs(config, judge_adapter)) as lm:
        for row, rubric, baseline, group in zip(rows, rubrics, baselines, groups):
            for c in group:
                labels, predictions = [], []
                for a, b in ((c["text"], baseline), (baseline, c["text"])):
                    prompt = judge_prompt(row["instruction"], rubric, a, b, config["prompt_style"])
                    prediction = _generate(lm, prompt, config, "judge", config["seed"])[0]["text"]
                    predictions.append(prediction)
                    labels.append(parse_winner(prediction))
                c.update({"reward": (float(labels[0] == "A")+float(labels[1] == "B"))/2,
                          "judge_predictions": predictions, "greedy_baseline": baseline, "rubric": rubric})
    return grpo(config, groups, None, None, output)
