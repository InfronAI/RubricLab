#!/bin/bash

# --- Defaults ---
length=0
dataset="magpie" # default dataset
dryrun=1
cuda="0" # default value for cuda

# --- Parse command-line arguments ---
# Note: the -d value is now stored in dataset_list
while getopts g:e:l:d:t: flag
do
    case "${flag}" in
        g) cuda=${OPTARG};;
        l) length=${OPTARG};;
        d) dataset_list=${OPTARG};; # store the comma-separated list here
        t) dryrun=${OPTARG};;
    esac
done

# If the user did not pass -d, fall back to the default
if [ -z "$dataset_list" ]; then
    dataset_list=$dataset
fi

# --- Set environment variables (once, outside the loop) ---
export CUDA_VISIBLE_DEVICES=$cuda
export CUDA_LAUNCH_BLOCKING=1

echo "Using CUDA_VISIBLE_DEVICES=$cuda"
echo "Setting max_samples=$length"
echo "Setting dryrun=$dryrun"

# --- Activate the Conda environment (once, outside the loop) ---
eval "$(conda shell.bash hook)"
conda activate synyue

# --- Run each dataset in turn ---

# 1. Split the comma-separated string into a Bash array using IFS
IFS=',' read -r -a dataset_array <<< "$dataset_list"

echo "Found ${#dataset_array[@]} dataset(s) to run: ${dataset_array[*]}"

# 2. Iterate over each dataset in the array and run the python command
for ds in "${dataset_array[@]}"; do
    echo ""
    echo "============================================="
    echo "Starting task for dataset: $ds"
    echo "============================================="

    python task_${ds}.py \
        --dryrun $dryrun \
        --max_samples $length \
        --model_name o4-mini \

    echo "---------------------------------------------"
    echo "Finished task for dataset: $ds"
    echo "---------------------------------------------"
done

echo "All tasks completed."
