"""Aggregate RunResult rows into the metrics that make this a study.

Reports, per (model, tier) and per (model, task):
  * pass@1   : mean success over all runs (the expected single-shot rate)
  * pass@k   : did ANY of the k runs succeed (capability ceiling)
  * mean steps, mean invalid-call rate, mean tokens (cost/efficiency)
  * mean judge score where applicable (path quality)

pass@1 vs pass@k is the key contrast: a large gap means the model CAN do the
task but isn't reliable -- exactly the signal you want when comparing a flaky
4B local model against a steady frontier model.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from typing import Any

from harness.core import RunResult


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def aggregate(rows: list[RunResult]) -> dict[str, Any]:
    by_task = defaultdict(list)
    by_tier = defaultdict(list)
    by_category = defaultdict(list)
    for r in rows:
        by_task[(r.model, r.task_id)].append(r)
        by_tier[(r.model, r.tier)].append(r)
        by_category[(r.model, r.category)].append(r)

    # per-task pass@k = did ANY of that task's runs succeed
    task_passk = {key: (1 if any(r.success for r in g) else 0)
                  for key, g in by_task.items()}

    def summarize(group, pass_at_k):
        succ = [1 if r.success else 0 for r in group]
        n = len(group)
        return {
            "n_runs": n,
            "pass_at_1": _mean(succ),
            "pass_at_k": pass_at_k,
            "mean_steps": _mean([r.n_steps for r in group]),
            "mean_invalid_rate": _mean([r.invalid_rate for r in group]),
            "mean_tokens": _mean([r.tokens_used for r in group]),
            "mean_input_tokens": _mean([r.input_tokens for r in group]),
            "mean_output_tokens": _mean([r.output_tokens for r in group]),
            "done_rate": round(sum(1 for r in group
                                   if r.halt_reason == "done") / n, 3) if n else None,
            "crash_rate": round(sum(1 for r in group
                                    if str(r.halt_reason).startswith("crash")) / n, 3)
                          if n else None,
            "mean_judge": _mean([r.judge_score for r in group]),
        }

    def group_passk(group):
        # mean of per-task pass@k over the distinct tasks in this group
        tasks = {(r.model, r.task_id) for r in group}
        vals = [task_passk[t] for t in tasks]
        return round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "by_task": {f"{m}|{t}": summarize(g, task_passk[(m, t)])
                    for (m, t), g in by_task.items()},
        "by_tier": {f"{m}|tier{t}": summarize(g, group_passk(g))
                    for (m, t), g in by_tier.items()},
        "by_category": {f"{m}|{c}": summarize(g, group_passk(g))
                        for (m, c), g in by_category.items()},
    }


def write_csv(rows: list[RunResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].to_row().keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.to_row())


def write_markdown_report(rows: list[RunResult], agg: dict, path: str) -> None:
    lines = ["# Agentic Eval — Results\n",
             f"Total runs: {len(rows)}\n",
             "_Procedurally-generated tasks: each repeat is a distinct seeded instance, "
             "so pass@k = solved ≥ 1 of k distinct instances._\n",
             "## Pass rates by tier\n",
             "| Model | Tier | pass@1 | pass@k | steps | invalid% | done% | in_tok | out_tok | judge |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(agg["by_tier"]):
        s = agg["by_tier"][key]
        model, tier = key.split("|")
        inv = f"{s['mean_invalid_rate']*100:.0f}%" if s['mean_invalid_rate'] is not None else "-"
        done = f"{s['done_rate']*100:.0f}%" if s['done_rate'] is not None else "-"
        judge = s['mean_judge'] if s['mean_judge'] is not None else "-"
        lines.append(f"| {model} | {tier} | {s['pass_at_1']} | {s['pass_at_k']} "
                     f"| {s['mean_steps']} | {inv} | {done} | {s['mean_input_tokens']} "
                     f"| {s['mean_output_tokens']} | {judge} |")
    lines.append("\n## Pass rates by capability\n")
    lines.append("| Model | Category | pass@1 | pass@k | steps | invalid% | done% | in_tok | out_tok | judge |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(agg["by_category"]):
        s = agg["by_category"][key]
        model, cat = key.split("|")
        inv = f"{s['mean_invalid_rate']*100:.0f}%" if s['mean_invalid_rate'] is not None else "-"
        done = f"{s['done_rate']*100:.0f}%" if s['done_rate'] is not None else "-"
        judge = s['mean_judge'] if s['mean_judge'] is not None else "-"
        lines.append(f"| {model} | {cat} | {s['pass_at_1']} | {s['pass_at_k']} "
                     f"| {s['mean_steps']} | {inv} | {done} | {s['mean_input_tokens']} "
                     f"| {s['mean_output_tokens']} | {judge} |")
    lines.append("\n## Per-task detail\n")
    lines.append("| Model | Task | pass@1 | pass@k | mean steps |")
    lines.append("|---|---|---|---|---|")
    for key in sorted(agg["by_task"]):
        s = agg["by_task"][key]
        model, task = key.split("|")
        lines.append(f"| {model} | {task} | {s['pass_at_1']} | {s['pass_at_k']} "
                     f"| {s['mean_steps']} |")
    open(path, "w").write("\n".join(lines))
