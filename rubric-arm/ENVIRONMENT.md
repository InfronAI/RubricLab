# Environment Setup

These scripts are intended to run in an ms-swift environment with GPU support.
For reproducibility, this release includes `environment.ms-swift.yml`, exported
from the conda environment used for the experiments.

## Recommended System

- Linux machine with NVIDIA GPUs.
- CUDA-capable driver installed on the host.
- Python 3.10 or 3.11.
- Enough GPU memory to run:
  - one rollout server,
  - one reward/judge deployment server for rubric-generator GRPO,
  - the GRPO training workers.

The exact GPU count depends on the base model size and your batch settings.
The release scripts keep the original multi-GPU defaults, but you can change
`CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` when launching.

## Create a Python Environment

### Option A: reproduce the exported environment

Use this when you want the closest match to the experiment environment:

```bash
conda env create -f environment.ms-swift.yml
conda activate ms-swift
```

The exported environment includes the full package/version list. Key versions:

- Python 3.10.19
- ms-swift 3.12.1
- torch 2.8.0
- transformers 4.57.1
- vllm 0.10.2
- deepspeed 0.18.1
- trl 0.23.1
- peft 0.17.1
- datasets 3.6.0
- accelerate 1.11.0
- modelscope 1.31.0

### Option B: create a fresh environment manually

Use this if your CUDA/PyTorch stack needs site-specific versions.

Using conda:

```bash
conda create -n ms-swift python=3.10 -y
conda activate ms-swift

python -m pip install --upgrade pip setuptools wheel
```

Using venv:

```bash
python3.10 -m venv .venv-rubricarm
source .venv-rubricarm/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

## Install PyTorch

Install a PyTorch build that matches your CUDA driver/runtime. Use the official
PyTorch selector if your cluster has a specific CUDA version.

Example for a CUDA 12.x environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your cluster already provides PyTorch through a module or container, use that
instead and verify it:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

## Install ms-swift and Runtime Dependencies

Minimal install:

```bash
pip install -U ms-swift
```

Full install, useful when you want vLLM/deployment/eval extras managed together:

```bash
pip install -U "ms-swift[all]"
```

If you prefer source installation:

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

Install or upgrade packages used directly by this release:

```bash
pip install -U requests transformers accelerate datasets deepspeed vllm
```

Optional but commonly useful:

```bash
pip install -U tensorboard liger-kernel
```

## Verify the Environment

Check that the `swift` CLI is available:

```bash
swift --help
swift rlhf --help
swift rollout --help
swift deploy --help
```

Check that Python can import the key packages:

```bash
python - <<'PY'
import requests
import torch
import transformers
import vllm
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY
```

Check that the release plugins are syntactically valid:

```bash
python -m py_compile reward_plugin_parallel.py judge_plugin_parallel.py
bash -n serve_policy.sh
bash -n serve_reward.sh
bash -n run_grpo.sh
bash -n run_grpo_judge.sh
```

## Notes for Cluster Runs

- Keep the reward server, rollout server, and training workers on separate GPU
  sets when possible.
- Set `MODEL_TYPE` to the ms-swift model type for your released model.
- Set `POLICY_MODEL_PATH`, `REWARD_MODEL_PATH`, `TRAIN_DATASET`,
  `VAL_DATASET`, and `OUTPUT_DIR` before running the scripts.
- If your site uses environment modules, load CUDA/NCCL/PyTorch modules before
  activating the Python environment.
- If `vllm` installation conflicts with your PyTorch build, reinstall vLLM
  after PyTorch, or use a container image with PyTorch, vLLM, and ms-swift
  already matched.

## Official References

- ms-swift installation docs: https://swift.readthedocs.io/
- ms-swift GitHub repository: https://github.com/modelscope/ms-swift
- PyTorch install selector: https://pytorch.org/get-started/locally/
- vLLM documentation: https://docs.vllm.ai/
