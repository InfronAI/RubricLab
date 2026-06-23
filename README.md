# OpenRubrics

Official implementation for the **OpenRubric family**: a line of work on
rubric-based reward modeling for LLMs.


## Works

### OpenRubrics
**OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment**
· [paper](https://arxiv.org/abs/2510.07743)

> Reward modeling lies at the core of reinforcement learning from human feedback (RLHF), yet most existing reward models rely on scalar or pairwise judgments that fail to capture the multifaceted nature of human preferences. Recent studies have explored *rubrics-as-rewards* (RaR) that uses structured criteria to capture multiple dimensions of response quality. However, producing rubrics that are both reliable and scalable remains a key challenge. In this work, we introduce OpenRubrics, a diverse, large-scale collection of (prompt, rubric) pairs for training rubric-generation and rubric-based reward models. To elicit discriminative and comprehensive evaluation signals, we introduce *Contrastive Rubric Generation* (CRG), which derives both hard rules (explicit constraints) and principles (implicit qualities) by contrasting preferred and rejected responses. We further remove noisy rubrics via preserving preference–label consistency. Across multiple reward-modeling benchmarks, our rubric-based reward model, Rubric-RM, surpasses strong size-matched baselines by 8.4%. These gains transfer to policy models on instruction-following and biomedical benchmarks. The model weights and datasets are publicly available at <https://huggingface.co/OpenRubrics>.


| Folder | What it is |
| ------ | ---------- |
| [`openrubrics/`](openrubrics/) | Data-generation pipeline — prompt a frontier model to extract rubrics from preference data, then produce rubric-grounded pairwise judgements. Produces the released training data. |
| [`rubric-rm/`](rubric-rm/) | **RubricRM** reward model — SFT (via the bundled LLaMA-Factory) of the rubric generator and the rubric-aware judge, plus evaluation on reward-modeling benchmarks. |

### Rubric-ARM
**Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training**
· [paper](https://arxiv.org/abs/2602.01511)


> Standard reward models typically predict scalar scores that fail to capture the multifaceted nature of response quality in non-verifiable domains, such as creative writing or open-ended instruction following. To address this limitation, we propose RUBRIC-ARM, a framework that jointly optimizes a rubric generator and a judge using reinforcement learning from preference feedback. Unlike existing methods that rely on static rubrics or disjoint training pipelines, our approach treats rubric generation as a latent action learned to maximize judgment accuracy. We introduce an alternating optimization strategy to mitigate the nonstationarity of simultaneous updates, providing theoretical analysis that demonstrates how this schedule reduces gradient variance during training. Extensive experiments show that RUBRIC-ARM achieves strong performance among baselines on multiple benchmarks and significantly improves downstream policy alignment in both offline and online reinforcement learning settings.

| Folder | What it is |
| ------ | ---------- |
| [`rubric-arm/`](rubric-arm/) | GRPO training that uses the rubric generator and rubric-aware judge as the reward signal. |

### Rubric-ARROW
**RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains**
· [paper](https://arxiv.org/abs/2605.29156) · [code](https://github.com/Haoxiang03/RUBRIC-ARROW)

> Pointwise reward modeling offers critical signals for LLM post-training, yet struggles with absolute scoring in subjective, non-verifiable settings. Rubric-based methods address this by decomposing evaluation into explicit criteria, but existing approaches typically depend on frontier LLMs and suffer from ties caused by hard Boolean aggregation. We present RUBRIC-ARROW, an alternating framework that jointly trains a rubric generator and a rubric-conditioned judge, with its RL stage using only pairwise preference data. Our method couples a probability-based scoring rule that reduces ties with phase-specific preference-based rewards and an alternating GRPO scheme that together train the pointwise evaluator. Extensive experiments show that RUBRIC-ARROW achieves competitive reward-modeling accuracy and yields consistent gains for downstream policy post-training.

Code lives in a separate repository: <https://github.com/Haoxiang03/RUBRIC-ARROW>.


## Models & data

All released weights and datasets live on the Hugging Face Hub:
**<https://huggingface.co/OpenRubrics>**

## Layout

```
openrubrics/   data generation         ┐
rubric-rm/     training + evaluation    ┘  OpenRubrics
rubric-arm/    RL with rubric rewards      Rubric-ARM
```

Each folder has its own README with setup and usage.

## Citation

If you find our work helpful, please consider citing our papers:

```bibtex
@misc{jiang2026rubric,
      title={RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains},
      author={Haoxiang Jiang and Zihan Dong and Tianci Liu and Wanying Wang and Ran Xu and Tony Yu and Linjun Zhang and Haoyu Wang},
      year={2026},
      eprint={2605.29156},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.29156},
}

@misc{xu2026alternating,
      title={Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training},
      author={Ran Xu and Tianci Liu and Zihan Dong and Tony Yu and Ilgee Hong and Carl Yang and Linjun Zhang and Tuo Zhao and Haoyu Wang},
      year={2026},
      eprint={2602.01511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.01511},
}

@misc{liu2025openrubrics,
      title={OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment},
      author={Tianci Liu and Ran Xu and Tony Yu and Ilgee Hong and Carl Yang and Tuo Zhao and Haoyu Wang},
      year={2025},
      eprint={2510.07743},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07743},
}
```

## License

See [LICENSE](LICENSE).
