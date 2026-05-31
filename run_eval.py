#!/usr/bin/env python3
"""Run the agentic eval study.

Examples
--------
  # Local only, 3 repeats:
  python run_eval.py --ollama qwen2.5:3b llama3.2:3b --k 3

  # Local + cloud, 5 repeats, with an Anthropic judge for path quality:
  export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...
  python run_eval.py --ollama qwen2.5:3b \
      --anthropic claude-opus-4-7 --openai gpt-4o \
      --judge anthropic:claude-opus-4-7 --k 5

  # Filter to specific tiers:
  python run_eval.py --ollama qwen2.5:3b --tiers 1 2

Outputs results.csv (raw rows) and report.md (aggregated) to --out.
"""
from __future__ import annotations

import argparse
import os
import sys

from harness.adapters import (AnthropicAdapter, ClaudeCliAdapter,
                              OllamaAdapter, OpenAIAdapter)
from harness.report import aggregate, write_csv, write_markdown_report
from harness.runner import run_study
from harness.summary import build_summary, write_summary
from tasks.suite import TASKS


def build_models(args) -> dict:
    models = {}
    for m in args.ollama or []:
        models[f"ollama:{m}"] = OllamaAdapter(m, host=args.ollama_host,
                                              use_native_tools=args.ollama_native)
    for m in args.anthropic or []:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set")
        models[f"anthropic:{m}"] = AnthropicAdapter(m, key)
    for m in args.claude_cli or []:
        models[f"claude-cli:{m}"] = ClaudeCliAdapter(m)
    for m in args.openai or []:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY not set")
        models[f"openai:{m}"] = OpenAIAdapter(m, key)
    if not models:
        sys.exit("No models specified. Use --ollama / --anthropic / --openai.")
    return models


def build_judge(spec):
    if not spec:
        return None
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicAdapter(model, os.environ["ANTHROPIC_API_KEY"])
    if provider == "openai":
        return OpenAIAdapter(model, os.environ["OPENAI_API_KEY"])
    if provider == "ollama":
        return OllamaAdapter(model)
    sys.exit(f"unknown judge provider: {provider}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ollama", nargs="*", help="Ollama model tags")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--ollama-native", action="store_true",
                    help="Use native tool-calls instead of ReAct fallback")
    ap.add_argument("--anthropic", nargs="*", help="Anthropic model ids (API key)")
    ap.add_argument("--claude-cli", nargs="*",
                    help="Claude model ids driven via local `claude -p` (no API key)")
    ap.add_argument("--openai", nargs="*", help="OpenAI model ids")
    ap.add_argument("--judge", help="provider:model for LLM-as-judge path scoring")
    ap.add_argument("--k", type=int, default=5, help="repeats per task (default 5)")
    ap.add_argument("--tiers", nargs="*", type=int, help="filter to these tiers")
    ap.add_argument("--tasks", nargs="*", help="filter to these task_ids")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed for procedural task generation (default 0)")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--dump-trajectories", action="store_true",
                    help="write each run's trajectory text to <out>/trajectories/")
    args = ap.parse_args()

    tasks = TASKS
    if args.tiers:
        tasks = [t for t in tasks if t.tier in args.tiers]
    if args.tasks:
        tasks = [t for t in tasks if t.task_id in args.tasks]
    models = build_models(args)
    judge = build_judge(args.judge)

    sink = None
    if args.dump_trajectories:
        traj_dir = os.path.join(args.out, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        def sink(model, task_id, run_index, traj):
            safe = model.replace(":", "-").replace("/", "-")
            fp = os.path.join(traj_dir, f"{safe}__{task_id}__k{run_index}.txt")
            with open(fp, "w") as fh:
                fh.write(traj.as_text())

    print(f"Running {len(tasks)} tasks x {len(models)} models x k={args.k} "
          f"= {len(tasks)*len(models)*args.k} runs")
    rows = run_study(tasks, models, k=args.k, judge_adapter=judge,
                     progress=lambda s: print("  ", s),
                     trajectory_sink=sink, base_seed=args.seed)

    csv_path = os.path.join(args.out, "results.csv")
    md_path = os.path.join(args.out, "report.md")
    write_csv(rows, csv_path)
    agg = aggregate(rows)
    write_markdown_report(rows, agg, md_path)
    summary_path = os.path.join(args.out, "eval-summary.json")
    write_summary(build_summary(rows, agg, args.k), summary_path)
    print(f"\nWrote {csv_path}, {md_path}, and {summary_path}")


if __name__ == "__main__":
    main()
