# Saved experimental data

`m1/train.jsonl` and `m1/validation.jsonl` are bounded samples from
[OpenRubrics/OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2),
revision `d1048e9e0375034d1dafb53392789cd368424861`. `m1/manifest.json` records
selection, exclusions, counts and the hash of the downloaded prefix. Text and
teacher trajectories are original dataset content, not generated benchmark results.
The downloaded prefix remains locally in `source.jsonl` and is excluded from Git.

Benchmark slices in `m1/benchmarks/` retain source IDs, pinned revisions, groups,
preferences and the complete untruncated response strings. `benchmarks.json`
documents context filtering and selection. Repeated prompts in evaluation are
intentional when the benchmark defines multiple candidate/style comparisons.

Sources and dataset licenses are separate from this repository's MIT software license:

| Source | License / attribution |
|---|---|
| [OpenRubric-v2](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2) | Consult the dataset card and constituent source datasets; no blanket license is inferred |
| [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) | Open Data Commons Attribution (ODC-BY), including constituent-source notices |
| [RewardBench2](https://huggingface.co/datasets/allenai/reward-bench-2) | ODC-BY |
| [RM-Bench](https://huggingface.co/datasets/THU-KEG/RM-Bench) | ODC-BY |
| [WritingPreferenceBench](https://huggingface.co/datasets/m-a-p/Writing-Preference-Bench) | Apache-2.0 |

The selection is deliberately small and biased toward shorter sequences. These
files must not be presented as the complete official test sets.
