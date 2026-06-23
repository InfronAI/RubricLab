#!/bin/bash
# Evaluate RubricRM on the reward-modeling benchmarks.
#
# Usage:
#   ./run_eval.sh [-g GPU] [-s SIZE]
#     -g  CUDA_VISIBLE_DEVICES (default: 0)
#     -s  model size, 4 or 8 (default: 8)
#
# Rubric/judge checkpoints are pulled from the Hugging Face Hub
# (OpenRubrics/RubricRM-*). To evaluate a local checkpoint or an API baseline,
# pass --rubric-model / --judge-model to the eval scripts directly.

set -e

gpu=0
size=8

while getopts g:s: flag; do
    case "${flag}" in
        g) gpu=${OPTARG};;
        s) size=${OPTARG};;
    esac
done

export CUDA_VISIBLE_DEVICES=$gpu

for bench in rewardbench rewardbench2 rmbench write; do
    echo "==================== eval_${bench} (size=${size}) ===================="
    python "eval_${bench}.py" --size "$size"
done

# Score everything written under ./outputs/ours[${size}B]
python compute_acc.py "./outputs/ours[${size}B]"
