# Supplementary diagnostics

These experiments are separate from the frozen main M1 profile. `plan.json`
records their budgets before model evaluation. `execution.json` records actual
jobs as they run; a planned entry is not evidence of execution.

The current protocol uses adapters from `m1-cpu-training-20261116`, after a
gradient audit invalidated the earlier MPS-training attempt. The previous queue,
input declarations and one completed long-decoding job remain under
`provenance/mps-training-attempt/` and `long-decode-sft.mps-training-diagnostic/`.
They are not combined with the corrected run's performance results.

- **Long decoding:** the same two held-out pairs, SFT adapters, seed and five votes,
  with context 4096 and a 768-token judge budget. This tests whether truncation
  explains missing final decisions. It does not retrain the model.
- **Joint two-stage budget diagnostic:** after auditing teacher lengths, keep the
  same SFT weights and pairs but increase rubric/judge caps to 320/1024 tokens
  with context 4096. All 24 training judgments exceed the primary 320-token cap;
  19/24 teacher rubrics exceed 128 tokens. This exploratory intervention changes
  both generation budgets and is not a single-factor comparison. Its declaration
  predates supplementary model execution; `teacher-target-lengths.json` retains
  the counts and input hashes.
- **Controlled response-order check:** swap the same two held-out pairs for SFT
  and ARM, matching the original five votes, temperature 0.7 and seeds. The main
  profile's single reversed generation is greedy, so that field alone cannot
  isolate order sensitivity from decoding differences. Invalid votes are counted
  separately; no valid order comparison is not evidence of zero position bias.
- **IFEval policy evaluation:** four seeded official prompts, a 512-token budget,
  and Google's pinned strict/loose verifiers, applied to the base model, the
  source-label DPO policy and the online-GRPO policy. ARM-DPO and IterDPO are also
  evaluated if their stages actually produce trained adapters; empty optimization
  sets never receive substitute policy scores.
- **Additional public pairs:** two pairs each from HelpSteer3, PPE-IFEval and THU-KEG IFBench, using
  final ARM adapters. The exports are selected before running the judge.

## Additional data provenance

[HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) provides human
preferences from -3 to +3. Negative values prefer response 1, positive values
prefer response 2, and ties are excluded. The export uses its official preference
validation file, retains single-turn context completely, and excludes overlap
with the main training and validation prompts. The source is licensed CC-BY-4.0.

[PPE-IFEval](https://huggingface.co/datasets/lmarena-ai/PPE-IFEval-Best-of-K) contains
32 responses per prompt and stored correctness checks. This diagnostic chooses
one response passing both prompt-level strict/loose checks and one failing both.
The two exported pairs' labels were independently reproduced using the vendored
Google verifiers; see `ppe-label-verification.json`. It is not the full PPE
Best-of-K metric or a reconstruction of the ARM authors' exact pair subset.
Prompts use Apache-2.0; generated responses follow the source providers' terms.

[THU-KEG/IFBench](https://huggingface.co/datasets/THU-KEG/IFBench) is the Peng et al.
benchmark cited by Rubric-ARM. Its chosen/rejected pairs include descriptions of
violated constraints. The export preserves those labels and descriptions without
executing embedded checker code. Its license is Apache-2.0. AllenAI's same-named
benchmark is a separate task and is not substituted here.

`additional-data-v2/manifest.json` pins source revisions, source-file hashes,
selection exclusions, exporter hash and output hashes for all three sources.
The initial two-source export remains under `additional-data/`, with its exact
exporter snapshot. Its HelpSteer3 and PPE files match the v2 output hashes.

```bash
python experiments/supplementary/prepare_extra_benchmarks.py \
  --samples 2 --output experiments/my-additional-pairs
openrubrics evaluate --data experiments/my-additional-pairs/helpsteer3.jsonl \
  --rubric-adapter experiments/my-m1-run/07-arm/round-3-rubric \
  --judge-adapter experiments/my-m1-run/07-arm/round-3-judge \
  --output experiments/my-helpsteer3-evaluation
```

All sample sizes are too small for benchmark claims. Context filtering, single-turn
selection and strict pass/fail pairing impose further selection bias. Reports
must distinguish invalid output, truncation, an incorrect preference, and no
accepted optimization pairs.
