"""Single portable entry point; no hosted APIs or GPU frameworks are imported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openrubrics", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Report hardware and supported accelerators")
    prepare = subs.add_parser("prepare", help="Download a pinned, bounded official dataset subset")
    prepare.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    prepare.add_argument("--output", type=Path, default=Path("data/m1"))
    run = subs.add_parser("run", help="Run all configured stages with durable artifacts")
    run.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    run.add_argument("--data", type=Path, default=Path("data/m1"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"])
    run.add_argument("--resume", action="store_true")
    report = subs.add_parser("report", help="Rebuild plots and analysis from saved predictions")
    report.add_argument("run", type=Path)
    evaluate = subs.add_parser("evaluate", help="Evaluate normalized local JSONL pairs")
    evaluate.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--rubric-adapter", type=Path)
    evaluate.add_argument("--judge-adapter", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    judge = subs.add_parser("judge", help="Generate a rubric and judge a pair locally")
    judge.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    judge.add_argument("--instruction", required=True)
    judge.add_argument("--a", required=True)
    judge.add_argument("--b", required=True)
    judge.add_argument("--rubric-adapter", type=Path)
    judge.add_argument("--judge-adapter", type=Path)
    judge.add_argument("--output", type=Path)
    serve = subs.add_parser("demo", help="Serve a local, artifact-backed experiment viewer")
    serve.add_argument("--run", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    from .config import load_config
    from .io import environment, read_jsonl, write_json
    if args.command == "doctor":
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        from .data import prepare_data
        if (args.output / "manifest.json").exists():
            parser.error("Data exists; choose a new output directory to preserve provenance")
        print(json.dumps(prepare_data(load_config(args.config), args.output), indent=2))
    elif args.command == "run":
        from .runner import run
        config = load_config(args.config)
        if args.device:
            config["device"] = args.device
        run(config, args.data, args.output, args.resume)
    elif args.command == "report":
        from .report import build_report
        build_report(args.run)
    elif args.command == "evaluate":
        from .data import validate_pair
        from .experiments import evaluate
        if args.output.exists():
            parser.error("Evaluation output exists; use a new directory")
        rows = [validate_pair(r) for r in read_jsonl(args.data)]
        print(evaluate(load_config(args.config), rows, args.rubric_adapter, args.judge_adapter, args.output))
    elif args.command == "judge":
        from .backend import LocalLM
        from .config import lm_kwargs
        from .metrics import parse_winner
        from .prompts import judge_prompt, rubric_prompt
        config = load_config(args.config)
        with LocalLM(**lm_kwargs(config, args.rubric_adapter)) as lm:
            rubric = lm.generate(rubric_prompt(args.instruction, config["prompt_style"]),
                                 max_new_tokens=config["rubric_tokens"], temperature=0)[0]
        with LocalLM(**lm_kwargs(config, args.judge_adapter)) as lm:
            prediction = lm.generate(judge_prompt(args.instruction, rubric["text"], args.a, args.b,
                                                  config["prompt_style"]),
                                     max_new_tokens=config["judge_tokens"], temperature=0)[0]
        result = {"instruction": args.instruction, "rubric": rubric, "judgment": prediction,
                  "winner": parse_winner(prediction["text"])}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            write_json(args.output, result)
    elif args.command == "demo":
        from .demo import serve
        serve(args.run, args.port)
