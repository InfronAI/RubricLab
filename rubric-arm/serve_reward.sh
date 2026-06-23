#!/usr/bin/env bash
set -euo pipefail

# Start the external rubric-conditioned pairwise judge as a vLLM service.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-DEBUG}" \
swift deploy \
    --model_type "${MODEL_TYPE:?Set MODEL_TYPE to the ms-swift model type}" \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 1 \
    --response_prefix '<think>\n\n</think>\n\n' \
    --max_new_tokens 1024 \
    --vllm_gpu_memory_utilization 0.9 \
    --model "${REWARD_MODEL_PATH:?Set REWARD_MODEL_PATH to the pairwise judge checkpoint or base model}"
