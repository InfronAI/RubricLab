# Reproducible local protocol

The default profile is a bounded **Qwen2.5-0.5B-Instruct / LoRA** experiment on a
16 GB Apple M1 Pro. `configs/smoke.json` uses SmolLM2-135M-Instruct for a shorter
mechanism check. Neither profile is a substitute for the published 4B/8B models.

1. Pin the model revision and the official OpenRubric-v2 revision. Download at
   most 1,024 source lines and 32 MiB. Validate decisions, remove repeated prompts
   and exclude over-context training examples before selecting a seeded subset.
2. Split by normalized prompt, so the same request cannot leak across training
   and validation. Save both selected JSONL files, the selection policy and hashes.
3. Run local contrastive rubric generation and preference-label filtering. Save
   every attempt. The local teacher's retention rate is an experiment, not a
   prerequisite for the released teacher-data SFT baseline.
4. Train separate rubric and judge LoRA adapters on published, already-filtered
   teacher trajectories. Mask all prompt tokens from the SFT objective. Preserve
   complete responses and final Winner lines; never silently truncate targets.
5. Compare a direct base judge, base rubric + judge, and SFT rubric + judge.
   Generate five judgments with temperature 0.7. Invalid generations count in the
   denominator; only more than half of all votes determines a winner.
6. Run three judge→rubric rounds. Each round uses a disjoint training slice. The
   counterpart is frozen while the actor updates; the reference stays at the SFT
   checkpoint. Cache rollouts before optimization. Population-standardized group
   rewards, asymmetric ratio clipping (0.2/0.28), token-mean loss, k3 KL penalty
   and gradient clipping are explicit in `training.py`.
7. Repeat with rubric→judge ordering and with zero format-reward weight, both
   starting from the same SFT checkpoints. Evaluate held-out prompts, including
   swapped response orders for the main SFT and ARM comparisons.
8. Evaluate a bounded, context-fitting slice of each supported public benchmark.
   Keep whole prompt groups: RM-Bench has nine style pairs, and RewardBench2
   preserves all comparisons for each chosen response. Sample counts and
   selection bias accompany all scores.
9. Relabel policy pairs with the final ARM judge in both orders. Train DPO only
   on consistent decisions; keep a separate source-label DPO baseline. IterDPO
   generates new pairs from the current policy at every round.
10. Generate a greedy policy baseline and sampled responses. Average the two
    order-corrected preference indicators to obtain the online GRPO reward.
11. Build CSV/JSON summaries, figures and a Markdown analysis from the saved raw
    predictions. Inspect invalid outputs, token-limit rates, zero-variance groups
    and empty DPO batches before interpreting any accuracy.

## Artifacts and recovery

Each experiment directory contains `manifest.json`, numbered stage directories,
model adapter files, tokenizer metadata, sampled outputs, training traces and
`analysis/`. Adapter tensors are retained locally but excluded from Git. Selected
data, model/data revisions, metrics and checkpoint SHA-256 hashes are versionable.
The large bounded source prefix is retained in `data/*/source.jsonl` locally and
can be fetched again from its immutable revision.

`--resume` verifies the configuration, input hashes, code fingerprint and every
completed stage artifact. A partial stage is moved aside as `.failed-*` and rerun
from its previous stage boundary. Mid-optimizer-step resume is not implemented;
the system does not pretend to restore optimizer state. Ordinary resume rejects
source changes. Use a new experiment directory for a changed method, or the
explicit code-fix procedure below for a reviewed implementation correction.

Models are loaded sequentially. Each MPS inference batch runs in a fresh process,
so Metal graph caches cannot accumulate over a long evaluation. The recorded
`batch_seconds` measures generation; `worker_wall_seconds` also includes process
startup and model loading. On Apple Silicon, training and old/reference-policy scoring
use CPU autograd; MPS is used for forward generation. Both use float32. Actual
training devices are saved in each checkpoint's `execution_device.json`. No BF16, bitsandbytes, FlashAttention, DeepSpeed, Triton
or vLLM is needed for this path. Sampled votes share a batched prefill; wall time
is amortized per sample and the original batch duration is also retained.

## Interpreting small runs

Tiny samples have wide uncertainty and context filtering causes selection bias.
Repeated pairs from the same prompt are correlated. Wilson intervals in the
report are descriptive per-pair intervals, not a claim of statistical significance.
SFT losses on different examples are not a validation-loss curve. A stage can
execute successfully yet have zero useful preference gradient. A DPO stage with
no consistent pairs records zero steps and a skip reason. These outcomes matter
more than a green process exit code.

## Official IFEval policy check

The optional policy check uses Google's original strict and loose verifiers,
vendored under `vendor/instruction_following_eval/` at commit
`e6890f85757dd84e27ca6df2dd30651dafad28e0`, with its Apache-2.0 license.
It reports both prompt-level and instruction-level accuracy. The four-prompt
local sample uses seed 42 and an explicitly recorded generation budget; it is
not the full benchmark and not the paper's downstream policy result.

```bash
python -m pip install -e '.[ifeval]'
python -m nltk.downloader punkt punkt_tab
python scripts/evaluate_ifeval.py --samples 4 --max-tokens 512 \
  --output experiments/ifeval-base
python scripts/evaluate_ifeval.py --samples 4 --max-tokens 512 \
  --adapter experiments/my-m1-run/13-dpo-source-baseline \
  --output experiments/ifeval-source-dpo
```

Tokenizer resources can also be supplied through `NLTK_DATA` for offline use.
The original verifier is retained unchanged; its inputs, outputs and source
hashes are saved with the measurements.

## Offline continuation

Once the pinned model/tokenizer files and benchmark source files are cached,
`HF_HUB_OFFLINE=1 openrubrics run ... --resume` avoids optional Hub network probes.
This environment setting does not change the numerical configuration. The retired
MPS-training attempt verified the cache; the corrected CPU-training run starts
offline using that cache. See `experiments/acceptance/offline-cache-check.json`.
Missing cached files remain explicit errors.

## Metal stability and process isolation

Real testing first exposed non-finite gradients, then a more serious disagreement
between finite MPS gradients and a CPU reference verified by central finite
differences. Neither eager/SDPA attention nor disabling activation checkpointing
resolved the full-model backward discrepancy. See the [gradient audit](GRADIENT_AUDIT.md).
The default therefore uses CPU autograd on Apple Silicon and reruns all affected
stages from the base model. MPS-trained checkpoints from the earlier attempt are
diagnostic artifacts only. Every SFT/GRPO/DPO phase still has an isolated worker
and saved `training_request.json`; activation checkpointing and exact chunked
completion loss remain enabled on CPU. MPS inference uses FP32 eager attention.
Finite-loss/gradient checks and norm clipping remain enabled, but are not treated
as a substitute for gradient correctness.

Generation needs the same protection: after repeated model transitions and
variable-length decoding, the resident inference process reached a 19.5 GB peak
physical footprint, despite calls to `empty_cache()`. That run was interrupted
and its process sample retained in `experiments/acceptance/`. The MPS coordinator
now keeps only tokenizers; each batch loads and releases its model in a child
process. This adds model-loading overhead but bounds the lifetime of graph caches.
MPS generation also uses a static KV cache with compilation disabled. One actual
five-vote comparison produced identical token sequences with dynamic and static
caches (95.21 s versus 70.42 s generation time; 4.60 versus 3.60 GiB reported MPS
driver allocations). See `experiments/acceptance/cache-comparison.json`. These
single-run diagnostics on a shared machine are not a controlled throughput study.

The chunked decoder/head path targets the Qwen and Llama architectures used by the supplied profiles (SmolLM2 is Llama-compatible). Other causal model architectures may require a backend adapter; arbitrary Hub IDs are not promised to work.

The initial 320-token judge budget can truncate a full rubric-grounded explanation. Inspect `length_limit_rate` before interpreting invalid decisions. The supplementary long-decoding check keeps the trained adapters fixed and raises only the context/output budget; it is reported separately from the primary profile.

The main profile's single reversed judgment uses greedy decoding, while its
forward votes are sampled. Treat `order_consistency` there as a mixed-decoding
diagnostic, not an isolated estimate of position bias. The supplementary order
check matches five votes, temperature and seeds in both response orders and
reports the number of valid comparable decisions explicitly.

For an explicitly reviewed code fix, `--resume --code-fix-reason "reason" --rerun-stage STAGE` records code lineage and archives each named affected stage. Configuration and input-data changes remain forbidden. Name every affected completed stage; unrelated verified artifacts can be retained. Ordinary `--resume` still rejects source changes.
