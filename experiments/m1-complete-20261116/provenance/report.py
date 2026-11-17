"""Rebuild numerical reports from immutable artifacts, including failed/empty outcomes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .io import read_jsonl, write_json
from .metrics import summarize


def build_report(run: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = run / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    results = []
    for path in sorted(run.rglob("predictions.jsonl")):
        if ".failed-" in str(path):
            continue
        rows = read_jsonl(path)
        metrics = summarize(rows)
        results.append({"experiment": str(path.parent.relative_to(run)), **metrics})
    write_json(dest / "metrics.json", results)
    fields = ["experiment", "n", "accuracy", "majority_accuracy", "invalid_rate", "abstention_rate",
              "choose_a_rate_valid", "group_all_correct"]
    with (dest / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    if results:
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(10, max(3, len(results)*0.43)))
        y = list(range(len(results)))
        values = [100*(r["majority_accuracy"] or 0) for r in results]
        intervals = [r.get("majority_wilson95") or [0, 0] for r in results]
        ax.barh(y, values, color="#0f8b8d", height=0.55)
        ax.errorbar(values, y,
                    xerr=[[max(0., v-100*c[0]) for v, c in zip(values, intervals)],
                          [max(0., 100*c[1]-v) for v, c in zip(values, intervals)]],
                    fmt="none", ecolor="#17324d", capsize=3)
        ax.set_yticks(y, [f"{r['experiment']} (n={r['n']})" for r in results])
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("Majority accuracy (%) · Wilson 95% interval per pair")
        ax.set_title("Local small-scale experiments — not paper-scale results", loc="left", weight="bold")
        fig.tight_layout()
        fig.savefig(dest / "accuracy.png", dpi=160)
        plt.close(fig)
    curves = []
    for path in sorted(run.glob("0*-sft-*/training.jsonl")):
        records = read_jsonl(path)
        curves.append((path.parent.name, records))
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, records in curves:
            ax.plot([r["step"] for r in records], [r["loss"] for r in records], marker=".", label=name)
        ax.set(xlabel="Optimizer step", ylabel="Completion-only cross entropy", title="SFT training traces")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "training.png", dpi=160)
        plt.close(fig)
    lines = ["# Local experiment analysis", "",
             f"Model: `{manifest['config']['model']}`. Profile: `{manifest['config']['profile']}`.", "",
             ("These measurements establish execution of a reduced protocol. They do not reproduce the "
             "papers' model sizes, datasets, training budgets or leaderboard scores."), "",
             "| Experiment | Pairs | Single accuracy | Voting accuracy | Invalid generations |",
             "|---|---:|---:|---:|---:|"]
    def percent(x):
        return "n/a" if x is None else f"{100*x:.1f}%"
    for r in results:
        lines.append(f"| {r['experiment']} | {r['n']} | {percent(r['accuracy'])} | "
                     f"{percent(r['majority_accuracy'])} | {percent(r['invalid_rate'])} |")
    lines += ["", "## Interpretation", "",
              ("Invalid or ambiguous Winner lines count as incorrect; a vote needs more than half of all "
              "requested generations. Empty results are never reported as successful accuracy. The saved "
              "raw text distinguishes format failure, length exhaustion and an incorrect decision."), "",
              ("The sample is deliberately bounded and filtered to fit the context window. Training and validation "
              "are prompt-disjoint, but the selected subset is not population-representative. Multiple style "
              "pairs share prompts; the displayed pairwise Wilson intervals ignore that correlation and are "
              "descriptive, not a significance test. RewardBench2 additionally records group-all-correct accuracy."), "",
              ("Training loss compares different examples across steps. A falling trace alone does not establish "
              "generalization. Zero-variance GRPO groups provide no preference-gradient signal; see each "
              "round's summary. Invalid judges may yield zero retained DPO pairs; these stages explicitly skip "
              "optimization instead of substituting gold labels. The source-label DPO baseline is separate."), "",
              ("Fresh CRG uses the small local model. SFT uses the published teacher-curated dataset. Local CRG "
              "retention is measured separately and must not be interpreted as frontier-teacher quality."), "",
              "## Execution", "", "| Stage | Status | Seconds |", "|---|---|---:|"]
    for name, stage in manifest["stages"].items():
        lines.append(f"| {name} | {stage['status']} | {stage.get('seconds', 0):.1f} |")
    lines += ["", ("Artifacts include exact model/data revisions, configuration, code hashes, input hashes, "
              "per-sample predictions, sampled token IDs, rewards, optimizer traces and checkpoint hashes."), ""]
    (dest / "REPORT.md").write_text("\n".join(lines))
    print(f"Report: {dest / 'REPORT.md'}", flush=True)


def dashboard_data(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    results, samples = [], []
    for path in sorted(run.rglob("predictions.jsonl")):
        if ".failed-" in str(path):
            continue
        rows = read_jsonl(path)
        results.append({"name": str(path.parent.relative_to(run)), **summarize(rows)})
        for row in rows[:2]:
            samples.append({"experiment": str(path.parent.relative_to(run)), "instruction": row["instruction"],
                            "rubric": row.get("rubric", "Direct judge baseline"),
                            "response_a": row["response_a"], "response_b": row["response_b"],
                            "prediction": row["predictions"][0] if row["predictions"] else "",
                            "gold": row["label"]})
    stages = [{"name": k, "status": v["status"], "seconds": v.get("seconds", 0)}
              for k, v in manifest["stages"].items()]
    return {"model": manifest["config"]["model"], "profile": manifest["config"]["profile"],
            "environment": manifest["environment"], "stages": stages, "results": results, "samples": samples}
