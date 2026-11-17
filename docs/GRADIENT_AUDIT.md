# Why Apple Silicon training uses CPU autograd

On the tested M1 Pro / macOS / PyTorch 2.8.0 stack, a completed online-GRPO
phase reported finite pre-clipping gradient norms near 72,000. A fixed-weight
replay found substantial CPU/MPS backward disagreement even though forward log
probabilities differed by at most 0.000221. Finite losses and gradient clipping
were therefore insufficient evidence of a correct training implementation.

The audit uses the same final adapter, six saved sampled completions, advantages,
old-policy log probabilities and fixed-reference log probabilities. It performs
no optimizer updates. Full gradient arrays and scripts are retained in
[`experiments/acceptance/online-gradient-check/`](../experiments/acceptance/online-gradient-check/).

| Backward path | Gradient norm | Relative L2 difference from CPU eager | Cosine similarity |
|---|---:|---:|---:|
| CPU eager reference | 0.950484 | 0 | 1 |
| CPU SDPA | 0.950483 | 0.0000174 | 0.99999999985 |
| MPS eager | 2.412736 | 2.4250 | 0.30786 |
| MPS SDPA | 2.412752 | 2.4250 | 0.30786 |
| MPS eager without activation checkpointing | 24995.4177 | 26297.5573 | 0.02086 |

The gradient vectors were copied to CPU and their norms recomputed in float64,
so this is not merely a disagreement in an MPS norm-reduction operation.
An isolated vocabulary-sized cross-entropy backward test agreed across devices;
the audit does not identify a single failing primitive or claim every MPS model
has this behavior.

Central finite differences on the CPU objective, with epsilon 0.001 and no
checkpoint writes, independently checked two unit directions:

| Direction | Finite-difference derivative | CPU autograd prediction | Native-device autograd prediction |
|---|---:|---:|---:|
| CPU gradient direction | 0.950252 | 0.950484 | 0.950484 |
| MPS gradient direction | 0.292853 | 0.292618 | 2.412736 |

See [variant comparisons](../experiments/acceptance/online-gradient-check/variants.json),
[finite differences](../experiments/acceptance/online-gradient-check/finite-difference.json),
and the [cross-entropy isolation check](../experiments/acceptance/cross-entropy-mps-check.json).
The CPU objective is consistent with its gradients; the MPS backward path does
not agree with that validated reference closely enough to support training claims.

The portable backend now selects **CPU for training and reference/old-policy
scoring on Apple Silicon**, while retaining isolated FP32 MPS forward inference.
CPU SDPA agrees with CPU eager in this audit. Each saved adapter records its
requested and actual execution device. An explicit internal override is reserved
for numerical diagnostics and is not exposed by the experiment CLI.

The earlier [`m1-complete-20261116`](../experiments/m1-complete-20261116/validity.json)
run remains intact as a diagnostic attempt. Its MPS-trained checkpoints and
derived evaluations are excluded from final training conclusions. The corrected
[`m1-cpu-training-20261116`](../experiments/m1-cpu-training-20261116/) run starts
from the original base model and reruns every stage; it reuses no trained weights.
The independently completed CPU smoke run is unaffected by this MPS finding.
