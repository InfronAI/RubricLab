#!/bin/bash
# Train RubricRM with the bundled LLaMA-Factory.
#
# Usage:
#   ./run.sh rubric                 # train the rubric model
#   ./run.sh judge                  # train the judge model
#   ./run.sh rubric -g 0,1,2,3      # choose GPUs (default: 0,1,2,3)
#
# Prerequisite: run scripts/prepare_data.py once to download the data and
# register the datasets in LLaMA-Factory/data/dataset_info.json.
set -e

which=${1:?usage: $0 [rubric|judge] [-g GPUS]}
shift

gpus=0,1,2,3
while getopts g: flag; do
    case "${flag}" in
        g) gpus=${OPTARG};;
    esac
done

case "$which" in
    rubric) config=examples/train_full/qwen3_8b_full_sft_rubric.yaml ;;
    judge)  config=examples/train_full/qwen3_8b_full_sft_judge.yaml ;;
    *) echo "usage: $0 [rubric|judge] [-g GPUS]"; exit 1 ;;
esac

# Activate the conda env so torchrun / python come from it (not base conda),
# otherwise the torchrun workers fail with "No module named 'llamafactory'".
eval "$(conda shell.bash hook)"
conda activate synyue

export CUDA_VISIBLE_DEVICES=$gpus
export FORCE_TORCHRUN=1                                         # launch multi-GPU via torchrun
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_$USER}  # keep triton cache off NFS

cd "$(dirname "$0")/LLaMA-Factory"
llamafactory-cli train "$config"
