# RubricARM GRPO 

This folder contains scripts for alternating reinforcement
learning of rubric-based reward models with ms-swift GRPO. Local paths,
checkpoint names, dataset names, and experiment identifiers have been replaced
with placeholders or environment variables.

See `ENVIRONMENT.md` for ms-swift environment installation instructions.


## Files

- `reward_plugin_parallel.py`: reward plugin for rubric-generator GRPO. The
  policy completion is treated as a rubric and scored by an external pairwise
  judge.
- `judge_plugin_parallel.py`: reward plugin for judge GRPO. It parses the
  generated `Winner:` line and applies the original auxiliary format reward.
- `run_grpo.sh`: rubric-generator GRPO entrypoint.
- `run_grpo_judge.sh`: judge GRPO entrypoint.
- `serve_policy.sh`: starts the ms-swift rollout server.
- `serve_reward.sh`: starts the external judge model with `swift deploy`.
- `data/*.example.jsonl`: minimal examples for the two JSONL schemas.

## Data Format

### Rubric-generator GRPO

Use `data/rubric_grpo_train.example.jsonl` as the schema reference.

Required fields:

- `messages`: prompt consumed by the rubric policy.
- `instruction`: original task instruction.
- `response_a`: first candidate answer.
- `response_b`: second candidate answer.
- `winner`: gold preference label, accepted values include `A`, `B`,
  `Response A`, and `Response B`.

The policy completion becomes the candidate rubric. `reward_plugin_parallel.py`
passes that rubric to the external judge and gives +1 when the judge decision
matches `winner`, otherwise -1.

### Judge GRPO

Use `data/judge_grpo_train.example.jsonl` as the schema reference.

Required fields:

- `messages`: prompt consumed by the judge policy.
- `instruction`: original task instruction.
- `rubric`: rubric text used by the format reward.
- `response_a`: first candidate answer.
- `response_b`: second candidate answer.
- `winner`: gold preference label.

The judge completion must include a final `Winner:` line. The format reward
expects the same section names and criterion style shown in the example prompt.

## Running

Use separate terminals for the reward server, rollout server, and training job.

### 1. Start the external judge reward service

This service is required for rubric-generator training.

```bash
cd /path/to/rubric_arm_release

CUDA_VISIBLE_DEVICES=0 \
MODEL_TYPE=your_ms_swift_model_type \
REWARD_MODEL_PATH=/path/to/pairwise-judge-checkpoint \
bash serve_reward.sh
```

Configure the reward plugin to call the deployed judge:

```bash
export PAIRWISE_REWARD_ENDPOINT=http://127.0.0.1:8000
export PAIRWISE_REWARD_MODEL=/model-id-reported-by-the-server
export PAIRWISE_REWARD_WORKERS=32
export PAIRWISE_REWARD_SERVER_BATCH=1024
export PAIRWISE_REWARD_TIMEOUT=60
```

If `PAIRWISE_REWARD_MODEL` is unset, the plugin queries `/v1/models` and uses
the first returned model id. If you use a custom batched reward service, set
`PAIRWISE_REWARD_ENDPOINT` to a URL ending in `/score`.

### 2. Start the rollout server

For rubric-generator training, use the initial rubric-policy checkpoint. For
judge training, use the initial judge-policy checkpoint.

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_TYPE=your_ms_swift_model_type \
POLICY_MODEL_PATH=/path/to/policy-checkpoint \
POLICY_PORT=8010 \
bash serve_policy.sh
```

### 3. Train the rubric generator

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
MODEL_TYPE=your_ms_swift_model_type \
POLICY_MODEL_PATH=/path/to/initial-rubric-policy-checkpoint \
TRAIN_DATASET=/path/to/rubric_train.jsonl \
VAL_DATASET=/path/to/rubric_val.jsonl \
OUTPUT_DIR=/path/to/output/rubric_grpo \
VLLM_SERVER_PORT=8010 \
bash run_grpo.sh
```

### 4. Train the pairwise judge

This stage computes rewards locally from the judge completion and does not
require `serve_reward.sh`.

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
NPROC_PER_NODE=7 \
MODEL_TYPE=your_ms_swift_model_type \
POLICY_MODEL_PATH=/path/to/initial-judge-policy-checkpoint \
TRAIN_DATASET=/path/to/judge_train.jsonl \
VAL_DATASET=/path/to/judge_val.jsonl \
OUTPUT_DIR=/path/to/output/judge_grpo \
VLLM_SERVER_PORT=8010 \
bash run_grpo_judge.sh
```

To resume judge training, additionally set:

```bash
export RESUME_FROM_CHECKPOINT=/path/to/checkpoint
```

## Alternating Training

A typical loop is:

1. Train or initialize a rubric-conditioned pairwise judge.
2. Train the rubric generator with `run_grpo.sh`, using the judge as the
   external reward model.
3. Build judge-training prompts with the latest rubrics and preference labels.
4. Train the judge with `run_grpo_judge.sh`.
5. Repeat with the new checkpoints.
