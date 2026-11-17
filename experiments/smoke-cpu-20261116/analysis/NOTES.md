# CPU end-to-end acceptance analysis

This completed run used SmolLM2-135M-Instruct on the Apple M1 Pro's CPU, with the
committed smoke configuration and `--device cpu`. The original invocation took
about 599 seconds. Individual stage durations remain in the manifest; a subsequent
verified resume rebuilt the report and refreshed the invocation completion time.

- Both SFT actors completed four actual optimizer steps with finite gradients.
- One judge→rubric round completed, with two GRPO updates per actor.
- The two held-out prompts were evaluated with five votes before and after training.
- ARM-labeled DPO and both IterDPO rounds retained no order-consistent labels, so
  they correctly recorded zero updates. No fabricated labels were inserted.
- The separate source-label DPO baseline completed two optimizer steps.
- Online GRPO executed two updates over two policy rollouts. Its single reward
  group had zero variance and zero mean reward, so it supplied no preference signal.
- The public benchmark sweep and ablations were disabled in this smoke profile.
- All 13 completed stage receipts passed artifact-hash verification on resume.

The 96-token judge budget and very small model produced no parseable Winner lines
in the saved held-out judgments. Accuracy and voting accuracy were therefore zero.
This is an engineering acceptance run, not evidence that the model learned a useful
reward function. It validates execution and honest handling of empty/invalid cases;
the larger MPS protocol separately measures the three-round and benchmark paths.

See [REPORT.md](REPORT.md), the parent [manifest](../manifest.json), and
[resume verification](../../acceptance/smoke-resume-check.txt).
