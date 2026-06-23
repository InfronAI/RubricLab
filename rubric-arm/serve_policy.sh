#!/usr/bin/env bash
set -euo pipefail

# Start the ms-swift rollout server used by GRPO training.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
swift rollout \
  --model_type "${MODEL_TYPE:?Set MODEL_TYPE to the ms-swift model type}" \
  --vllm_tensor_parallel_size 1 \
  --use_hf true \
  --response_prefix '<think>\n\n</think>\n\n' \
  --port "${POLICY_PORT:-8010}" \
  --model "${POLICY_MODEL_PATH:?Set POLICY_MODEL_PATH to the policy checkpoint or base model}"
