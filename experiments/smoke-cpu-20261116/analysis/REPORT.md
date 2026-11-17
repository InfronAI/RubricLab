# Local experiment analysis

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`. Profile: `smoke-mechanism-check`.

These measurements establish execution of a reduced protocol. They do not reproduce the papers' model sizes, datasets, training budgets or leaderboard scores.

| Experiment | Pairs | Single accuracy | Voting accuracy | Invalid generations |
|---|---:|---:|---:|---:|
| 04-base-direct | 2 | 0.0% | 0.0% | 100.0% |
| 05-base-rubric | 2 | 0.0% | 0.0% | 100.0% |
| 06-sft-evaluation | 2 | 0.0% | 0.0% | 100.0% |
| 08-arm-evaluation | 2 | 0.0% | 0.0% | 100.0% |

## Interpretation

Invalid or ambiguous Winner lines count as incorrect; a vote needs more than half of all requested generations. Empty results are never reported as successful accuracy. The saved raw text distinguishes format failure, length exhaustion and an incorrect decision.

The sample is deliberately bounded and filtered to fit the context window. Training and validation are prompt-disjoint, but the selected subset is not population-representative. Multiple style pairs share prompts; the displayed pairwise Wilson intervals ignore that correlation and are descriptive, not a significance test. RewardBench2 additionally records group-all-correct accuracy.

Training loss compares different examples across steps. A falling trace alone does not establish generalization. Zero-variance GRPO groups provide no preference-gradient signal; see each round's summary. Invalid judges may yield zero retained DPO pairs; these stages explicitly skip optimization instead of substituting gold labels. The source-label DPO baseline is separate.

Fresh CRG uses the small local model. SFT uses the published teacher-curated dataset. Local CRG retention is measured separately and must not be interpreted as frontier-teacher quality.

## Execution

| Stage | Status | Seconds |
|---|---|---:|
| 01-curation | completed | 16.7 |
| 02-sft-rubric | completed | 13.6 |
| 03-sft-judge | completed | 36.8 |
| 04-base-direct | completed | 41.2 |
| 05-base-rubric | completed | 76.4 |
| 06-sft-evaluation | completed | 91.4 |
| 07-arm | completed | 98.6 |
| 08-arm-evaluation | completed | 104.9 |
| 11-public-benchmarks | completed | 0.8 |
| 12-dpo-arm | completed | 24.3 |
| 13-dpo-source-baseline | completed | 12.1 |
| 14-iterative-dpo | completed | 48.0 |
| 15-online-grpo | completed | 34.1 |

Artifacts include exact model/data revisions, configuration, code hashes, input hashes, per-sample predictions, sampled token IDs, rewards, optimizer traces and checkpoint hashes.
