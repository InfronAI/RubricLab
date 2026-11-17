# Experiment artifacts

The primary local protocol is in [`m1-cpu-training-20261116/`](m1-cpu-training-20261116/).
It uses CPU autograd and MPS generation after a real gradient audit invalidated
the earlier MPS-training attempt. Its manifest records the fresh stages. Superseded outputs
remain in explicitly named `.superseded-*` directories and are excluded from the
current report. The input data and configuration cannot change during resume.

| Directory | Meaning |
|---|---|
| `m1-cpu-training-20261116/` | Fresh 0.5B protocol with CPU training/scoring and MPS inference; no MPS-trained weights reused |
| `m1-complete-20261116/` | Executed 17 stages, then failed a CPU/MPS gradient audit; retained for diagnosis only, see `validity.json` |
| `smoke-cpu-20261116/` | Completed 135M CPU protocol on M1; one ARM round, no public benchmark sweep, empty judge-label batches explicitly skipped |
| `acceptance/` | Mathematical, browser, memory, isolated-training and recovery validation |
| `supplementary/` | Fixed-weight decoding budgets, controlled answer order, additional public pairs and official IFEval checks |
| `m1-20261116/` | Initial run interrupted after excessive memory use |
| `m1-optimized-20261116/` | Explicit context-limit failure after a prompt change; not a completed experiment |
| `m1-final-20261116/` | Archived non-finite-gradient failure |
| `m1-verified-20261116/` | Archived failure before process isolation |

Failures are retained as portability evidence, not combined with successful
benchmark results. The selected small dataset is not representative of the full
paper's corpus. Model adapter tensors and tokenizer copies are saved locally but
excluded from Git; manifests, hashes, training traces and raw predictions are
versionable. The project does not download or bundle the papers' large model
weights into Git.

Inside the retired MPS attempt, `provenance/code/` snapshots its initial source; the subsequent
source snapshot is in `provenance/fp32-stable-code/`. The manifest's
`code_fix_history` explains which completed stages were retained and which were
archived and rerun. This preserves both what failed and what actually executed.

`provenance/isolated-static-code/` records the subsequent inference memory fix:
each MPS batch runs in a fresh process with a fixed-shape KV cache. The physical
memory samples and exact-token cache comparison are retained under `acceptance/`.

The retired attempt's final source snapshot is `provenance/final-code/`. It additionally clears
device-specific static caches before CPU retries; the original fault and the
passing bare/LoRA model checks are retained in `acceptance/`. The short direct
baseline was rerun, while completed stages using the unchanged successful MPS
path were verified and retained.

The corrected CPU-training run has its own `m1-cpu-training-20261116/provenance/code/`
snapshot and manifest fingerprint. Its fresh execution does not inherit stages
or checkpoints from any retired attempt.

Within that CPU-training run, `provenance/bounded-inference-code/` snapshots the
later five-minute MPS worker deadline and fresh CPU retry. The interrupted public
benchmark stage is archived under `.superseded-*` and rerun in full. Its earlier
completed stages, trained entirely with CPU autograd, are verified and retained;
the manifest records this reviewed recovery change.

After numerical experiments completed, `provenance/presentation-fix/` retained the
prior manifest and before/after sources for a mobile-layout correction and direct
loopback readiness checks in the recorder. Only the viewer HTML and recording
script changed. All model/data/evaluation sources, stage artifacts and the model
completion time remain unchanged; the manifest records this presentation-only
boundary. Reports and media are regenerated against the resulting source record.
