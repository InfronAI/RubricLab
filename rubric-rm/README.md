# RubricRM

> **Apple Silicon / CPU:** use the [portable workflow](../README.md) from the repository root. The instructions below describe the preserved original API/GPU environment.

Training and evaluation code for **RubricRM**, the rubric-based reward model from
OpenRubrics. RubricRM is made of two independently trained models:

| Model      | Role                        | Input → Output                                        |
| ---------- | --------------------------- | ----------------------------------------------------- |
| **Rubric** | Rubric generator            | a user request → a list of rubric items               |
| **Judge**  | Rubric-aware pairwise judge | request + rubric + two responses → judgement & winner |

Both are released in 4B and 8B sizes (the configs here use Qwen3-8B as the
example). Training is plain SFT done with
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a copy of which is
bundled under `LLaMA-Factory/` for convenience.

## Layout

```
rubric-rm/
├── README.md
├── scripts/
│   └── prepare_data.py                 # download HF data → LLaMA-Factory JSON
├── eval/                                # evaluation
│   ├── common.py                        # released model ids + backend selection
│   ├── api.py                           # hosted-API baselines (GPT/Gemini/DeepSeek)
│   ├── base_rubric.py                   # rubric generation (vLLM / API)
│   ├── base_judge.py                    # rubric-aware judging (vLLM / API)
│   ├── eval_rewardbench.py              # RewardBench       (allenai/reward-bench)
│   ├── eval_rewardbench2.py             # RewardBench 2     (allenai/reward-bench-2)
│   ├── eval_rmbench.py                  # RM-Bench          (THU-KEG/RM-Bench)
│   ├── eval_write.py                    # Writing-Pref-Bench(m-a-p/Writing-Preference-Bench)
│   ├── compute_acc.py                   # score the prediction JSONs
│   └── run_eval.sh
└── LLaMA-Factory/                       # bundled training framework
    ├── examples/train_full/
    │   ├── qwen3_8b_full_sft_rubric.yaml
    │   └── qwen3_8b_full_sft_judge.yaml
    └── data/
        └── dataset_info.json            # trimmed to the example datasets
```

## Setup

Recreate the conda environment we used (`synyue`) and install the bundled
LLaMA-Factory:

```bash
conda env create -f environment.yml
conda activate synyue

cd LLaMA-Factory
pip install -e ".[torch,deepspeed]"
```

## Step 1 — Prepare the data

The SFT data is released on the Hugging Face Hub
([OpenRubrics/OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2)).
`prepare_data.py` downloads it and assembles the two JSON files LLaMA-Factory
expects, then registers them in `data/dataset_info.json`:

- **Rubric** → ShareGPT format, deduplicated by `(request, rubric)`.
- **Judge**  → Alpaca format, one example per preference pair.

```bash
python scripts/prepare_data.py --hf-repo OpenRubrics/OpenRubric-v2
```

By default this writes into the bundled `LLaMA-Factory/data/openrubric_v2/`
(`rubric.json` and `judge.json`) and adds the dataset keys
`openrubric_v2_rubric` and `openrubric_v2_judge`, which the training configs
reference. Pass `--lf-root /path/to/LLaMA-Factory` to target a different
checkout.

## Step 2 — Train

From the `LLaMA-Factory/` directory:

```bash
# Rubric generator
llamafactory-cli train examples/train_full/qwen3_8b_full_sft_rubric.yaml

# Rubric-aware judge
llamafactory-cli train examples/train_full/qwen3_8b_full_sft_judge.yaml
```

To train a different size, change `model_name_or_path` in the config to the
corresponding base model (e.g. `Qwen/Qwen3-4B`).

## Evaluation

The pipeline is two-stage: the **Rubric** model generates a rubric from each
prompt, then the **Judge** model uses that rubric to pick the better of the two
responses. Each example is judged 5 times for majority voting. By default the
scripts pull the released checkpoints from the Hub
(`OpenRubrics/RubricRM-{4,8}B-{Rubric,Judge}-v2`) and serve them with vLLM.

```bash
cd eval
./run_eval.sh -g 0 -s 8          # GPU 0, 8B models, all HF-loadable benchmarks
# or run one benchmark:
python eval_rewardbench.py --size 8
python compute_acc.py "./outputs/ours[8B]"
```

**Benchmarks.** All four load their dataset directly from the Hub and run
out-of-the-box: `rewardbench` (allenai/reward-bench), `rewardbench2`
(allenai/reward-bench-2), `rmbench` (THU-KEG/RM-Bench), and `write`
(m-a-p/Writing-Preference-Bench).

**Baselines / overrides.** Pass `--rubric-model` / `--judge-model` to score a
local checkpoint or a hosted-API baseline instead of the released models. An API
name such as `gpt-4.1`, `gemini`, or `deepseek` routes through `api.py`; set the
matching credentials (`OPENAIKEY` + `AZURE_OPENAI_ENDPOINT`, `GOOGLE_API_KEY`,
or AWS env vars).
