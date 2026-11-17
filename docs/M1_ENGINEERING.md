# Apple Silicon engineering notes

The portable path is separate from the retained CUDA implementations. It uses
Transformers and PEFT with the same rubric-conditioned learning structure, while
making model size, parameterization and budgets explicit in JSON configuration.
It does not translate the papers' full-parameter 8B training into a 16 GB workload.

| Change | Why it is needed | Validation / tradeoff |
|---|---|---|
| CPU autograd on Apple Silicon | Full-model finite MPS gradients disagreed with a validated CPU reference | [Fixed-weight gradient and finite-difference audit](GRADIENT_AUDIT.md); all affected stages rerun from scratch, with actual devices saved |
| Completion-only output projection | A full `[sequence, vocabulary]` logit tensor is expensive for the 151k-token Qwen vocabulary | Chunked suffix cross entropy matches full causal-LM loss and backpropagates in the regression test |
| Activation checkpointing and LoRA | Reduce saved activations and trainable optimizer state | Used by the CPU trainer; this changes the trainable parameterization relative to full tuning |
| FP32 eager attention for MPS inference | FP16/SDPA and repeated model transitions produced non-finite sampling probabilities | Forward generation remains on MPS; finite backward values alone were insufficient, so training uses CPU |
| Separate training workers | Keep each phase's model and allocator lifetime bounded | Saved training requests are replayable; process exit releases the phase's allocations |
| Separate inference worker per batch | A long resident generation process reached a 19.5 GB peak physical footprint and started swapping | Parent holds no model weights; a regression test prevents accidental coordinator allocation |
| Static KV cache, compilation disabled | Dynamic sequence lengths cause additional Metal graph variants | Tiny-model greedy parity test and exact token parity on a real five-vote MPS diagnostic |
| Batched votes and rubric cache | Avoid repeating prompt prefill and rubric generation for every vote/pair | All requested votes are retained; per-generation time is an amortized batch time |
| Recorded CPU fallback | Some MPS operations or sampling failures remain device-specific | Fault-injected recovery test; retry uses the same weights, input and seed and records the actual device |
| Five-minute MPS worker deadline | A long-input worker exceeded 20 minutes under heavy swapping | Kill and reap the worker, retry the unchanged request in a fresh CPU process, preserve retry reason and total wall time |

Static KV tensors are not registered model buffers, so moving a model to CPU does
not migrate its old static cache. A real tiny-model fault injection reproduced
that device-mismatch error. The fallback now discards the static cache and uses a
fresh dynamic cache on CPU. Both bare and PEFT-wrapped models passed the hardware
retry test; see `experiments/acceptance/static-cache-fallback-after.json`.

## Measurements and their limits

The corrected run completed 24 CPU SFT steps per actor in 35.3 seconds for the
rubric generator and 159.1 seconds for the judge, including model loading and
checkpoint serialization. These are actual stage durations on the shared M1 Pro,
not a controlled CPU-versus-GPU speed comparison. Checkpoint device receipts
confirm CPU execution.

The real cache diagnostic used the same Qwen2.5-0.5B SFT adapter, prompt, seed 143,
five votes and a 320-token output cap. All five generated token sequences matched:

| Measurement | Dynamic KV cache | Static KV cache |
|---|---:|---:|
| Batch generation time | 95.21 s | 70.42 s |
| Reported MPS driver allocation at completion | 4.60 GiB | 3.60 GiB |
| Sampled process physical footprint | 8.0 GB | 4.9 GB |

The process samples were taken during generation, not continuously at peak. Driver
allocation and process physical footprint measure different things. The machine
was shared with other applications, so these are diagnostic observations, not a
controlled speed or memory benchmark. Do not infer a universal speedup from them.
Raw outputs and process samples are in [`experiments/acceptance/`](../experiments/acceptance/).

A longer RewardBench2 input in the corrected run reached an 8.4 GB process
footprint with five votes. At the saved observation, 5.2 GB of its writable
regions were swapped out; system-wide swap usage was about 24 GB on this shared
machine. The total system swap cannot be attributed to OpenRubrics. This explains
why the short-prompt cache measurement is not a memory ceiling or a reliable
runtime estimate for longer inputs. Avoid concurrent memory-heavy workloads when
running the full profile on 16 GB. The [process sample](../experiments/acceptance/long-context-memory.txt)
and [system context](../experiments/acceptance/long-context-memory-context.json)
retain the observation; they are not a controlled performance comparison.

That worker was interrupted after more than 20 minutes. The incomplete public
benchmark stage is archived, and the complete stage is rerun after adding the
five-minute deadline. The earlier CPU training stages remain valid: this change
adds a recovery path and does not change losses, weights, prompts or decoding
budgets. The largest saved worker duration in the retained stages was 188.6 s.
The run's code-fix history records the boundary and both source snapshots.

The deadline includes model loading. `subprocess.run(timeout=...)` kills and reaps
the MPS worker before a separate CPU worker starts. The retry retains the model,
adapter, prompt, seed, temperature, token budget and vote count; a CPU error still
propagates. Saved `failed_worker_seconds` and `worker_wall_seconds` include the
failed attempt, while generation timing is measured inside the successful retry.
The [real-model recovery check](../experiments/acceptance/bounded-inference/timeout-retry.json)
compares all five shortened retry token sequences with direct CPU generation.
The [regular-path replay](../experiments/acceptance/bounded-inference/regular-mps-parity.json)
also matches all four RewardBench rubrics and 20 judgment token sequences before
and after the guard. These checks cover recovery and an unchanged successful MPS
path; they do not promise identical sampled tokens across CPU and MPS devices.
The [full-budget primary retry](../experiments/acceptance/bounded-inference/primary-recovery.json)
then completed the original 320-token, five-vote long-input request: CPU generation
took 131.1 s, and total worker wall time was 442.4 s including the timed-out MPS
attempt and process switching. Its prompt, seed and vote count match the interrupted
request. This is a recovery observation under shared load, not a device speed benchmark.

The retired MPS SFT attempt recorded maximum driver allocations of about 2.82 GiB
for the rubric model and 5.73 GiB for the judge. These remain memory diagnostics,
not validation of those gradients or the resource profile of the corrected CPU
trainer. They are sampled allocator values, not whole-system memory requirements.
The inference coordinator adds no resident model,
but every child still needs enough unified memory for its active batch.

## Recovery and reproducibility

An ordinary resume verifies the exact source fingerprint, configuration, input
hashes and completed-stage artifacts. It restores a completed stage boundary,
not optimizer state. A reviewed code correction requires an explicit reason and
an explicit list of stages to archive and rerun. The retired MPS attempt retains
its inference corrections and snapshots; the CPU-autograd correction starts a
fresh run and reuses no trained weights. Failed and superseded
predictions are excluded from current aggregate reports.

The profile and seed fix the requested experiment; device kernels are not promised
to be bitwise reproducible. Saved token IDs, raw text, rewards, updates and adapter
hashes are the evidence of what actually executed. Small-model invalid outputs,
zero-variance rewards and skipped empty DPO updates remain visible.
