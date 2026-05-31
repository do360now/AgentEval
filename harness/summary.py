"""Build the machine-readable eval-summary.json artifact (consumed by helloai).

agenteval stays pricing-agnostic: this emits raw token counts + a token_source tag;
the consumer computes cost from its own pricing table. The helloai_model_id is a
SUGGESTED foreign key the consumer may override.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from harness.core import RunResult
from harness.report import aggregate


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() \
            or "unknown"
    except Exception:
        return "unknown"


# (vendor substring -> suggested helloai FK). Order matters: first match wins.
_VENDOR = [("opus", "claude"), ("sonnet", "claude"), ("haiku", "claude"),
           ("claude", "claude"), ("gpt", "gpt"), ("o1", "gpt"), ("o3", "gpt"),
           ("gemini", "gemini"), ("grok", "grok")]


def _suggest_helloai_id(eval_model_id: str) -> str:
    low = eval_model_id.lower()
    for needle, fk in _VENDOR:
        if needle in low:
            return fk
    return eval_model_id.split(":")[0]          # fall back to provider prefix


def _display_model(eval_model_id: str) -> str:
    # "claude-cli:claude-opus-4-8" -> "Claude Opus 4.8"
    name = eval_model_id.split(":")[-1]
    parts = name.replace("claude-", "claude ").replace("-", " ").split()
    out = " ".join(p.capitalize() if not any(c.isdigit() for c in p) else p
                   for p in parts)
    return out.replace("Claude Claude", "Claude")


def _summ_metrics(s: dict) -> dict:
    return {"pass_at_1": s["pass_at_1"], "pass_at_k": s["pass_at_k"],
            "mean_steps": s["mean_steps"],
            "mean_input_tokens": s["mean_input_tokens"],
            "mean_output_tokens": s["mean_output_tokens"]}


def build_summary(rows: list[RunResult], agg: dict,
                  runs_per_task: int) -> dict[str, Any]:
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)

    models = []
    for model in sorted(by_model):
        grp = by_model[model]
        one = aggregate(grp)            # single-model aggregate -> reuse summarize
        n = len(grp)
        all_succ = [1 if r.success else 0 for r in grp]
        src = next((r.token_source for r in grp
                    if not str(r.halt_reason).startswith("crash")), "measured")

        by_task = [dict(task_id=k.split("|")[1], **_summ_metrics(v),
                        mean_judge=v["mean_judge"])
                   for k, v in sorted(one["by_task"].items())]
        by_cat = [dict(category=k.split("|")[1], pass_at_1=v["pass_at_1"],
                       pass_at_k=v["pass_at_k"],
                       mean_input_tokens=v["mean_input_tokens"],
                       mean_output_tokens=v["mean_output_tokens"])
                  for k, v in sorted(one["by_category"].items())]
        by_tier = [dict(tier=int(k.split("|tier")[1]), pass_at_1=v["pass_at_1"],
                        pass_at_k=v["pass_at_k"])
                   for k, v in sorted(one["by_tier"].items())]

        models.append({
            "eval_model_id": model,
            "helloai_model_id": _suggest_helloai_id(model),
            "display_model": _display_model(model),
            "token_source": src,
            "overall": {
                "pass_at_1": round(sum(all_succ) / n, 3) if n else None,
                "pass_at_k": round(sum(v["pass_at_k"] for v in one["by_task"].values())
                                   / len(one["by_task"]), 3) if one["by_task"] else 0,
                "mean_steps": round(sum(r.n_steps for r in grp) / n, 3) if n else None,
                "mean_input_tokens": round(sum(r.input_tokens for r in grp) / n, 3) if n else None,
                "mean_output_tokens": round(sum(r.output_tokens for r in grp) / n, 3) if n else None,
            },
            "reliability": {
                "mean_invalid_rate": round(sum(r.invalid_rate for r in grp) / n, 3) if n else None,
                "done_rate": round(sum(1 for r in grp if r.halt_reason == "done") / n, 3) if n else None,
                "crash_rate": round(sum(1 for r in grp
                                        if str(r.halt_reason).startswith("crash")) / n, 3) if n else None,
            },
            "by_category": by_cat,
            "by_tier": by_tier,
            "by_task": by_task,
        })

    return {
        "schema_version": "1.0",
        "suite_version": f"agenteval@{_git_sha()}",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_per_task": runs_per_task,
        "models": models,
    }


def write_summary(summary: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
