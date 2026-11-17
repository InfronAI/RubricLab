# Local experiment analysis

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Profile: `m1-small-scale`.

These measurements establish execution of a reduced protocol. They do not reproduce the papers' model sizes, datasets, training budgets or leaderboard scores.

| Experiment | Pairs | Single accuracy | Voting accuracy | Invalid generations |
|---|---:|---:|---:|---:|
| 04-base-direct | 2 | 0.0% | 0.0% | 10.0% |
| 05-base-rubric | 2 | 0.0% | 0.0% | 60.0% |
| 06-sft-evaluation | 2 | 0.0% | 0.0% | 100.0% |
| 08-arm-evaluation | 2 | 0.0% | 0.0% | 100.0% |
| 10-eval-no_format | 2 | 0.0% | 0.0% | 100.0% |
| 10-eval-reverse | 2 | 0.0% | 0.0% | 100.0% |
| 11-public-benchmarks/rewardbench | 4 | 0.0% | 0.0% | 100.0% |
| 11-public-benchmarks/rewardbench2 | 6 | 0.0% | 0.0% | 90.0% |
| 11-public-benchmarks/rmbench | 9 | 0.0% | 0.0% | 95.6% |
| 11-public-benchmarks/writing | 4 | 0.0% | 0.0% | 90.0% |

## Interpretation

Invalid or ambiguous Winner lines count as incorrect; a vote needs more than half of all requested generations. Empty results are never reported as successful accuracy. The saved raw text distinguishes format failure, length exhaustion and an incorrect decision.

The sample is deliberately bounded and filtered to fit the context window. Training and validation are prompt-disjoint, but the selected subset is not population-representative. Multiple style pairs share prompts; the displayed pairwise Wilson intervals ignore that correlation and are descriptive, not a significance test. RewardBench2 additionally records group-all-correct accuracy.

Training loss compares different examples across steps. A falling trace alone does not establish generalization. Zero-variance GRPO groups provide no preference-gradient signal; see each round's summary. Invalid judges may yield zero retained DPO pairs; these stages explicitly skip optimization instead of substituting gold labels. The source-label DPO baseline is separate.

Fresh CRG uses the small local model. SFT uses the published teacher-curated dataset. Local CRG retention is measured separately and must not be interpreted as frontier-teacher quality.

## Execution

| Stage | Status | Seconds |
|---|---|---:|
| 01-curation | completed | 122.5 |
| 02-sft-rubric | completed | 31.5 |
| 03-sft-judge | completed | 138.5 |
| 04-base-direct | completed | 23.5 |
| 05-base-rubric | completed | 210.9 |
| 06-sft-evaluation | completed | 396.0 |
| 07-arm | completed | 1089.8 |
| 08-arm-evaluation | completed | 364.0 |
| 09-ablation-no_format | completed | 917.3 |
| 09-ablation-reverse | completed | 877.5 |
| 10-eval-no_format | completed | 254.1 |
| 10-eval-reverse | completed | 267.0 |
| 11-public-benchmarks | completed | 2876.6 |
| 12-dpo-arm | completed | 230.6 |
| 13-dpo-source-baseline | completed | 19.2 |
| 14-iterative-dpo | completed | 402.8 |
| 15-online-grpo | completed | 424.8 |

Artifacts include exact model/data revisions, configuration, code hashes, input hashes, per-sample predictions, sampled token IDs, rewards, optimizer traces and checkpoint hashes.
