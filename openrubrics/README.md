# OpenRubrics data generation

> **Apple Silicon / CPU:** use the [portable workflow](../README.md) from the repository root. The instructions below describe the preserved original API/GPU environment.

The two-stage pipeline that produces the OpenRubrics preference data
([OpenRubrics/OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2)),
which [RubricRM](../rubric-rm) is trained on.

```
prompt + ranked responses ──▶ gen_rubric ──▶ rubrics/<ds>/… ──▶ gen_judge ──▶ judges/<ds>/…
```

1. **`gen_rubric/`** — given a request and its ranked responses, prompt a model
   to extract a list of rubric items, each tagged `[Hard Rule]` or `[Principle]`.
   Output: `rubrics/<dataset>/<dataset>-<model>.jsonl`.
2. **`gen_judge/`** — given the request, the generated rubric, and a pair of
   responses, prompt a model for a rubric-grounded pairwise judgement. Only
   judgements whose winner agrees with the original preference label are kept
   (`--filter_by_acc`). Output: `judges/<dataset>/<dataset>-<model>.jsonl`.

The kept judgements (instruction + responses + rubric + judgement) are what gets
released as the training data.

## Layout

```
gen_rubric/                 gen_judge/
├── task_magpie.py          ├── task_magpie.py       # data loader + entry point (example)
├── runner.py               ├── runner.py            # async generation loop
├── prompter.py             ├── prompter.py          # prompt template + builder
├── api.py                  ├── api.py               # model backends
├── utils.py                ├── utils.py
└── gen_rubric.sh           └── gen_judge.sh
```

We ship a single worked example, `magpie`, to keep the demonstration small. The
same `runner.py` / `prompter.py` drive every dataset, so adding another is just
a new `task_<dataset>.py` with its own `load_data()`.

## Input data

The raw Magpie preference data is **not bundled**. Supply its path with
`--input_path /path/to/magpie.jsonl` or the `MAGPIE_INPUT` environment variable.
No source-code edit is needed. The portable root workflow provides a bounded
released OpenRubric-v2 slice for local experiments.

`load_data()` shows the expected schema: each line has a request and a list of
`responses` with `scores`. The rubric prompt is **listwise** — responses are
ranked best-first and rendered as `Rank 1. … / Rank 2. …`. A pairwise dataset is
just the 2-element case (`responses = [chosen, rejected]`, `scores = [1, 0]`)
fed through the same listwise prompt.

`gen_judge` reads what `gen_rubric` writes, so run the rubric stage first. Its
loader defaults to `rubrics/<dataset>/<dataset>-o4-mini.jsonl` (i.e. the rubrics
produced with `--model_name o4-mini`).

## Models / credentials

Both stages call a frontier model through `api.py`, selected by `--model_name`:

| name        | backend                       | env |
| ----------- | ----------------------------- | --- |
| `o4-mini` / other GPT | Azure OpenAI          | `OPENAIKEY`, `AZURE_OPENAI_ENDPOINT` |
| `gemini`    | google-genai (judge only)     | `GOOGLE_API_KEY` |
| `deepseek`  | Amazon Bedrock                | standard AWS credentials |

Defaults: rubric generation uses `o4-mini`, judge generation uses `gemini`.

## Usage

```bash
export OPENAIKEY=...                       # Azure OpenAI key
export AZURE_OPENAI_ENDPOINT=https://...   # Azure OpenAI endpoint
export GOOGLE_API_KEY=...                  # for the judge stage (gemini)

export MAGPIE_INPUT=/path/to/magpie.jsonl
cd gen_rubric
./gen_rubric.sh -d magpie -t 1             # -t 1 = dry run (5 samples); -t 0 = full
#   override the model per task with --model_name

cd ../gen_judge
./gen_judge.sh -d magpie -t 1
```

`-d` accepts a comma-separated list to run several datasets in sequence (once
you have added their `task_<dataset>.py`).
