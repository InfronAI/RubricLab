#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare RubricRM SFT data for LLaMA-Factory.

This script downloads the curated OpenRubrics preference data from the Hugging
Face Hub and assembles it into the JSON formats consumed by LLaMA-Factory. It
builds two independent SFT datasets:

  * Rubric model  -> ShareGPT format  (generate a rubric from a request)
  * Judge  model  -> Alpaca  format  (rubric-aware pairwise judgement)

It then registers both datasets in LLaMA-Factory's ``data/dataset_info.json`` so
they can be referenced by name from the training YAMLs in
``LLaMA-Factory/examples/train_full/``.

The Hugging Face dataset (e.g. ``OpenRubrics/OpenRubric-v2``) provides one record
per line with the fields::

    instruction, response_a, response_b, winner, rubric, judge, source

We only wrap these fields with the prompt templates below; no rubric/judge
generation happens here.

Usage
-----
    pip install -U huggingface_hub

    # Uses the bundled LLaMA-Factory by default:
    python prepare_data.py --hf-repo OpenRubrics/OpenRubric-v2
    # Or point at your own checkout:
    python prepare_data.py --lf-root /path/to/LLaMA-Factory

After it finishes, the training YAMLs reference the datasets by name:

    dataset: openrubric_v2_rubric   # examples/train_full/qwen3_8b_full_sft_rubric.yaml
    dataset: openrubric_v2_judge    # examples/train_full/qwen3_8b_full_sft_judge.yaml
"""
import argparse
import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Prompt templates (kept verbatim from the training-data construction)
# --------------------------------------------------------------------------- #
RUBRIC_PROMPT_TEMPLATE = (
    "Your task is to extract a set of rubric-style instructions from a user's request.\n"
    "These rubrics will be used as evaluation criteria to check if a response fully meets the request.\n"
    "Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), it is automatically invalid.\n"
    "\n"
    "- **Two Distinct Categories:**\n"
    "  - [Hard Rule]: Derived strictly from explicit requirements stated in the <request> (format, length, structure, forbidden/required elements, etc.).\n"
    "  - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria (e.g., clarity, correctness, sound reasoning, pedagogy).\n"
    "\n"
    "- **Comprehensiveness:**\n"
    "  The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards.\n"
    "\n"
    "- **Conciseness & Uniqueness:**\n"
    "  Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.\n"
    "\n"
    "- **Format Requirements:**\n"
    "  - Use a numbered list.\n"
    "  - Each item starts with \"The response\" phrased in third person.\n"
    "  - Append [Hard Rule] or [Principle] at the end of each item.\n"
    "  - Do not include reasoning, explanations, or examples in the final output—only the rubrics.\n"
    "\n"
    "Here is the request:\n"
    "{prompt}\n"
    "\n"
    "Please generate the rubrics for the above request."
)

JUDGE_PROMPT_TEMPLATE = (
    "You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' "
    "based on a given instruction and a rubric. You will conduct this evaluation in distinct "
    "phases as outlined below.\n\n"
    "### Phase 1: Compliance Check Instructions\n"
    "First, identify the single most important, objective 'Gatekeeper Criterion' from the rubric.\n"
    "- **A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. "
    "Key examples are: word/paragraph limits, required output format (e.g., JSON validity), "
    "required/forbidden sections, or forbidden content.**\n"
    "- **Conversely, a rule is subjective if it requires interpretation or qualitative judgment. "
    "Subjective rules about quality are NOT Gatekeepers. Examples include criteria like \"be creative,\" "
    "\"write clearly,\" \"be engaging,\" or \"use a professional tone.\"**\n"
    "Think step-by-step to determine this single most important Gatekeeper, then write a 1–2 sentence explanation of your decision.\n\n"
    "### Phase 2: Analyze Each Response\n"
    "Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate each "
    "response item by item.\n"
    "For each item, think step-by-step and cite concrete evidence from the response before assigning your judgment.\n\n"
    "### Phase 3: Final Judgment Instructions\n"
    "Based on the results from the previous phases, determine the winner using these simple rules. "
    "Provide a final justification explaining your decision first and then give your decision.\n"
    "Think step-by-step to aggregate the findings and make the decision; keep the reasoning explicit and concise.\n\n"
    "---\n"
    "### REQUIRED OUTPUT FORMAT\n"
    "You must follow this exact output format below.\n\n"
    "--- Compliance Check ---\n"
    "Gatekeeper Reasoning: <1–2 sentences citing the relevant rubric text>\n"
    "Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>\n\n"
    "--- Analysis ---\n"
    "**Response A:**\n"
    "- Criterion 1 [Hard Rule]: Justification: <...>\n"
    "- Criterion 2 [Hard Rule]: Justification: <...>\n"
    "- Criterion 3 [Principle]: Justification: <...>\n"
    "- ... (and so on for all other criteria)\n\n"
    "**Response B:**\n"
    "- Criterion 1 [Hard Rule]: Justification: <...>\n"
    "- Criterion 2 [Hard Rule]: Justification: <...>\n"
    "- Criterion 3 [Principle]: Justification: <...>\n"
    "- ... (and so on for all other criteria)\n\n"
    "--- Final Judgment ---\n"
    "Aggregation Summary: <1–3 sentences explaining how Gatekeeper and other criteria led to the decision>\n"
    "Justification: <...>\n"
    "Winner: <Response A / Response B>\n\n\n"
    "Task to Evaluate:\n"
    "Instruction:\n{instruction}\n\n"
    "Rubric:\n{rubric}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}"
)


# --------------------------------------------------------------------------- #
# Data loading / formatting
# --------------------------------------------------------------------------- #
def load_records(hf_repo: str, hf_file: str):
    """Download the curated JSONL from the Hub and yield one dict per record."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_file,
        repo_type="dataset",
    )
    with open(local_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_rubric_dataset(records):
    """ShareGPT: (request) -> rubric. Deduplicated by (instruction, rubric)."""
    seen = set()
    out = []
    for r in records:
        key = (r["instruction"], r["rubric"])
        if key in seen:
            continue
        seen.add(key)
        prompt = RUBRIC_PROMPT_TEMPLATE.format(prompt=r["instruction"])
        out.append(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": r["rubric"]},
                ],
                "dataset": "ultrachat",
            }
        )
    return out


def build_judge_dataset(records):
    """Alpaca: (instruction + rubric + responses) -> rubric-aware judgement."""
    out = []
    for r in records:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=r["instruction"],
            rubric=r["rubric"],
            response_a=r["response_a"],
            response_b=r["response_b"],
        )
        out.append({"instruction": prompt, "output": r["judge"]})
    return out


# --------------------------------------------------------------------------- #
# dataset_info.json registration
# --------------------------------------------------------------------------- #
def register_datasets(lf_data_dir: Path, rubric_rel: str, judge_rel: str,
                      rubric_key: str, judge_key: str):
    info_path = lf_data_dir / "dataset_info.json"
    info = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    info[rubric_key] = {
        "file_name": rubric_rel,
        "formatting": "sharegpt",
    }
    info[judge_key] = {
        "file_name": judge_rel,
        "columns": {"prompt": "instruction", "response": "output"},
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info_path


def main():
    ap = argparse.ArgumentParser(
        description="Download OpenRubrics data from the Hub and prepare it for LLaMA-Factory."
    )
    default_lf_root = (Path(__file__).resolve().parent.parent / "LLaMA-Factory")
    ap.add_argument("--lf-root", default=str(default_lf_root),
                    help="Path to the LLaMA-Factory repository root "
                         "(default: the LLaMA-Factory bundled in this repo).")
    ap.add_argument("--hf-repo", default="OpenRubrics/OpenRubric-v2",
                    help="Hugging Face dataset repo id (default: OpenRubrics/OpenRubric-v2).")
    ap.add_argument("--hf-file", default="data/open_rubrics.jsonl",
                    help="Path of the JSONL file inside the dataset repo.")
    ap.add_argument("--out-subdir", default="openrubric_v2",
                    help="Subdirectory under LLaMA-Factory/data to write the JSON files.")
    ap.add_argument("--rubric-key", default="openrubric_v2_rubric",
                    help="dataset_info.json key for the rubric dataset.")
    ap.add_argument("--judge-key", default="openrubric_v2_judge",
                    help="dataset_info.json key for the judge dataset.")
    args = ap.parse_args()

    lf_data_dir = Path(args.lf_root).expanduser().resolve() / "data"
    if not lf_data_dir.is_dir():
        raise SystemExit(f"LLaMA-Factory data dir not found: {lf_data_dir}")

    out_dir = lf_data_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading {args.hf_repo}:{args.hf_file} ...")
    records = load_records(args.hf_repo, args.hf_file)
    print(f"      loaded {len(records)} records")

    print("[2/4] Building rubric (ShareGPT) and judge (Alpaca) datasets ...")
    rubric_data = build_rubric_dataset(records)
    judge_data = build_judge_dataset(records)
    print(f"      rubric: {len(rubric_data)} examples | judge: {len(judge_data)} examples")

    rubric_rel = f"{args.out_subdir}/rubric.json"
    judge_rel = f"{args.out_subdir}/judge.json"
    with open(lf_data_dir / rubric_rel, "w", encoding="utf-8") as f:
        json.dump(rubric_data, f, ensure_ascii=False, indent=2)
    with open(lf_data_dir / judge_rel, "w", encoding="utf-8") as f:
        json.dump(judge_data, f, ensure_ascii=False, indent=2)
    print(f"[3/4] Wrote {rubric_rel} and {judge_rel} under {lf_data_dir}")

    info_path = register_datasets(lf_data_dir, rubric_rel, judge_rel,
                                  args.rubric_key, args.judge_key)
    print(f"[4/4] Registered '{args.rubric_key}' and '{args.judge_key}' in {info_path}")

    print(
        "\nDone. From the LLaMA-Factory root, train with:\n"
        f"  llamafactory-cli train examples/train_full/qwen3_8b_full_sft_rubric.yaml  # dataset: {args.rubric_key}\n"
        f"  llamafactory-cli train examples/train_full/qwen3_8b_full_sft_judge.yaml   # dataset: {args.judge_key}"
    )


if __name__ == "__main__":
    main()
