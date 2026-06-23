#!/usr/bin/env bash
set -euo pipefail

# Rubric-generator GRPO training. This preserves the original training
# arguments; only private paths are exposed as environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-6}" \
swift rlhf \
  --rlhf_type grpo \
  --model_type "${MODEL_TYPE:?Set MODEL_TYPE to the ms-swift model type}" \
  --train_type full \
  --vllm_mode server \
  --vllm_server_host "${VLLM_SERVER_HOST:-127.0.0.1}" \
  --vllm_server_port "${VLLM_SERVER_PORT:-8010}" \
  --output_dir "${OUTPUT_DIR:?Set OUTPUT_DIR to the rubric GRPO output directory}" \
  --use_vllm true \
  --dataset "${TRAIN_DATASET:?Set TRAIN_DATASET to the rubric GRPO training JSONL}" \
  --val_dataset "${VAL_DATASET:?Set VAL_DATASET to the rubric GRPO validation JSONL}" \
  --reward_funcs pairwise_judge_reward \
  --reward_weights 1.0 \
  --external_plugins "${EXTERNAL_PLUGIN:-${SCRIPT_DIR}/reward_plugin_parallel.py}" \
  --num_generations 6 \
  --max_completion_length 512 \
  --generation_batch_size 288 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 3 \
  --learning_rate 1e-6 \
  --deepspeed zero3 \
  --temperature 1.0 \
  --top_p 0.95 \
  --logging_steps 1 \
  --save_steps 60 \
  --save_total_limit 1 \
  --greater_is_better true \
  --torch_dtype bfloat16 \
  --log_completions true \
  --log_entropy false \
  --report_to tensorboard \
  --num_train_epochs 1 \
  --use_hf true \
  --lr_scheduler_type constant \
  --epsilon_high 0.28 \
  --num_iterations 2 \
  --eval_strategy "steps" \
  --eval_steps "60" \
  --per_device_eval_batch_size 1 \
  --beta 0.001 \
  --use_liger_kernel true \
  --metric_for_best_model reward \
  --load_best_model_at_end true \
  --create_checkpoint_symlink true \
  --eval_generation_config '{"max_new_tokens":512,"temperature":0.0}' \
  --model "${POLICY_MODEL_PATH:?Set POLICY_MODEL_PATH to the initial rubric policy checkpoint or base model}"
