#!/usr/bin/env bash
set -euo pipefail

# Rubric-conditioned judge GRPO training. This preserves the original training
# arguments; only private paths are exposed as environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=(
  swift rlhf
  --rlhf_type grpo
  --model_type "${MODEL_TYPE:?Set MODEL_TYPE to the ms-swift model type}"
  --train_type full
  --vllm_mode server
  --vllm_server_host "${VLLM_SERVER_HOST:-127.0.0.1}"
  --vllm_server_port "${VLLM_SERVER_PORT:-8010}"
  --output_dir "${OUTPUT_DIR:?Set OUTPUT_DIR to the judge GRPO output directory}"
  --use_vllm true
  --dataset "${TRAIN_DATASET:?Set TRAIN_DATASET to the judge GRPO training JSONL}"
  --val_dataset "${VAL_DATASET:?Set VAL_DATASET to the judge GRPO validation JSONL}"
  --reward_funcs pairwise_judge_reward pairwise_judge_format_reward
  --external_plugins "${EXTERNAL_PLUGIN:-${SCRIPT_DIR}/judge_plugin_parallel.py}"
  --num_generations 7
  --max_completion_length 1024
  --generation_batch_size 224
  --per_device_train_batch_size 8
  --gradient_accumulation_steps 4
  --learning_rate 1e-6
  --deepspeed zero3
  --temperature 1.0
  --top_p 0.95
  --logging_steps 1
  --save_steps 60
  --save_total_limit 1
  --torch_dtype bfloat16
  --log_completions true
  --report_to tensorboard
  --num_train_epochs 1
  --use_hf true
  --use_liger_kernel true
  --lr_scheduler_type constant
  --epsilon_high 0.28
  --num_iterations 2
  --eval_strategy "steps"
  --eval_steps "60"
  --per_device_eval_batch_size 1
  --greater_is_better true
  --metric_for_best_model reward
  --load_best_model_at_end true
  --create_checkpoint_symlink true
  --beta 0.001
  --eval_generation_config '{"max_new_tokens":1024,"temperature":1.0}'
  --model "${POLICY_MODEL_PATH:?Set POLICY_MODEL_PATH to the initial judge policy checkpoint or base model}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-7}" \
"${CMD[@]}"
