# Paper-to-implementation audit

This audit covers the two papers implemented in this repository. RUBRIC-ARROW is a
separate project linked by upstream; it is not an implementation in this checkout.
The upstream revision inspected was `1a40c14cb7827ae10bfdf0f6d61ca35a36770b5b`.

Primary sources:

- [OpenRubrics, arXiv:2510.07743v1](https://arxiv.org/html/2510.07743v1)
- [Rubric-ARM, arXiv:2602.01511v1](https://arxiv.org/html/2602.01511v1)
- [Released training data](https://huggingface.co/datasets/OpenRubrics/OpenRubric-v2)
- [Original repository](https://github.com/wanghaoyu0408/OpenRubrics)

## Method coverage

| Component | Original checkout | Portable implementation | Fidelity / limits |
|---|---|---|---|
| Contrastive rubric generation | Hosted APIs; raw-data placeholder | `experiments.curate`, ranked responses, local causal LM | Same contrastive conditioning and rule/principle distinction; smaller teacher in local profile |
| Preference consistency filtering | Hosted judge runner | Strict rubric validity + winner-label agreement, raw rejected outputs retained | Local filtering measured independently of published SFT data |
| Rubric generator SFT | LLaMA-Factory, CUDA/full tuning | Completion-only next-token loss and LoRA | Same objective; different capacity, optimizer budget and parameterization |
| Rubric-conditioned judge SFT | LLaMA-Factory, CUDA/full tuning | Independent adapter, complete teacher trajectory | Final decision is never silently truncated |
| Two-stage inference | vLLM required at import | Transformers MPS/CPU, one model resident, prompt rubric cache | Compact or original prompt templates |
| Voting@5 | String plurality; empty inputs unsafe | Five sampled judgments, strict majority, explicit abstention | Conservative invalid-output handling; differs from legacy plurality |
| Alternating GRPO | Two shell entrypoints; manual handoff | Three disjoint rounds, judge then rubric, frozen counterpart | Token-level clipping + fixed-reference KL + group-normalized advantages |
| Format shaping | Does not recognize numbered source criteria | Both published numbered and `Criterion N` styles | Bounded 0–0.5 shaping; correctness is 0/1 as in ARM equations |
| Reverse-order / no-format ablations | No runner | Both branch from identical SFT adapters | Same subset and seed for comparison; local sample too small for conclusions |
| Position bias | Partial order shuffling | Deterministic shuffled pair labels, swapped-order evaluation | Gold never enters evaluation prompts |
| Offline DPO | No project-level policy workflow | ARM labels accepted only when both orders agree; fixed-reference DPO + SFT term | No replacement of invalid judge labels; source-label baseline recorded separately |
| Iterative DPO | Missing orchestration | Fresh policy samples, repeated order-consistent relabeling and DPO | Empty accepted sets explicitly skip updates |
| Online policy GRPO | Missing policy integration | Greedy baseline, two-order pairwise reward, GRPO | Uses ARM Eq. 16; no fabricated reward signal |
| Experiment provenance | Ad hoc outputs | Source/model/data revisions, hashes, raw outputs, checkpoint hashes, stage receipts | Device kernels may be nondeterministic; seeds do not imply bitwise equivalence |

## Evaluation coverage and unresolved reproduction requirements

| Experiment family | Available local execution | What is still required for the original experiment |
|---|---|---|
| RewardBench Chat / Chat-Hard | Official pinned data, bounded complete prompt groups | Full evaluation and original checkpoints/baselines |
| RM-Bench Chat | All 3×3 style comparisons per selected prompt, easy/normal/hard | Full corpus and original model budgets |
| RewardBench2 Precise IF / Focus | All chosen–rejected pairs for selected prompts, group-all-correct metric | Full corpus; official aggregation for leaderboard submission |
| WritingPreferenceBench | Explicit English files, nested response parsing | Full English/Chinese genre evaluation |
| PPE-IFEval reward-model evaluation | Pinned public source; bounded pairs with opposing official strict/loose labels exported by the supplementary script | Exact paper pair selection and full PPE Best-of-K aggregation are distinct from this diagnostic |
| IFBench reward-model evaluation | Pinned THU-KEG/IFBench chosen/rejected pairs with constraint-violation evidence | Full corpus and the authors' exact aggregation; this is the Peng et al. benchmark, not AllenAI's namesake |
| FollowBench, InfoBench reward-model evaluation | Generic normalized-pair evaluator accepts verified exported pairs | Paper-specific generated answer pairs and exact verifier/version manifests are not bundled upstream |
| HelpSteer3 | Pinned preference-validation source; non-tie single-turn pairs retain the signed overall human preference | Full multi-turn coverage and the paper's exact sample/aggregation protocol |
| IFEval downstream policy evaluation | Pinned official strict/loose verifiers and bounded policy evaluation script | Full task set and original model/training budget |
| InfoBench/IFBench downstream policy results | Policy outputs and adapters are produced by local training | Official task verifiers and complete policy evaluation; pairwise accuracy is not an instruction-following score |
| Arena-Hard, AlpacaEval, WildBench, Creative Writing v3 | Learned policy artifacts available | Official evaluation code, reference outputs, judge models/API budget and controlled scoring |
| HealthBench / biomedical transfer | Original generic rubric/judge algorithm remains applicable | Science-domain training slice, domain checkpoints, official HealthBench evaluation and original judge |
| Model-size comparisons and external reward-model baselines | Original GPU scripts retained | 4B/8B and 7B–32B checkpoints; corresponding compute and decoding budgets |
| Efficiency / cases | Actual local generation times, token counts, cached rubrics, raw examples | Matched original vLLM/GPU stack for fair comparison to published throughput |
| Variance analysis / theoretical claims | GRPO rewards and zero-variance groups logged | Large repeated training trials; the code does not claim an empirical proof of the theorem |

**This checkout is not a 100% reproduction of every paper experiment.** A 16 GB M1
cannot accommodate the published full-parameter 8B training configuration. FP32
weights alone need about 32 GB; gradients and Adam moments add substantially more.
LoRA and a smaller model permit execution of the central algorithms, not recovery
of published results. Several derived evaluation datasets and external-judge
protocols are absent from the upstream release. Their absence is recorded rather
than filled with synthetic benchmark numbers.

Benchmark identity matters: Rubric-ARM cites Peng et al. (2025) for IFBench,
corresponding to [THU-KEG/IFBench](https://huggingface.co/datasets/THU-KEG/IFBench)
and [Agentic Reward Modeling](https://aclanthology.org/2025.acl-long.775/).
AllenAI's `IFBench` is a different benchmark; installing its similarly named
verifier package would not reproduce this reward-model evaluation.

## Scaling without changing the method

Use `prompt_style: original` to select the preserved upstream templates. Raise the
data, context, update and decoding budgets explicitly; use a machine that can fit
them. The original `rubric-rm/LLaMA-Factory` full-SFT configurations and
`rubric-arm` ms-swift scripts remain available for the original GPU stack.
The portable trainer always uses LoRA; setting an 8B model ID does not turn it into
the paper's full-parameter trainer.

To evaluate an additional verified pair set, export one JSON object per line with
`id`, `instruction`, `response_a`, `response_b`, and `label` (`A` or `B`). Include
`source`, `source_revision`, `subset` and `group_id` when available, then use
`openrubrics evaluate --data pairs.jsonl --output experiments/additional-eval`.
Do not label arbitrary generated pairs with the small local judge and then call
the result an official benchmark.
