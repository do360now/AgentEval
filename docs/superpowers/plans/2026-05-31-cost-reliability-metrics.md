# Cost + Reliability Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split tokens into input/output with a `token_source` tag, surface reliability
(invalid-rate, halt outcomes), fix the tier/category `pass@k` aggregation, and emit
`eval-summary.json` for the helloai leaderboard.

**Architecture:** Add an input/output token split through `ModelAction → Trajectory →
RunResult` (the total still drives the budget guard). Adapters fill the split from their real
usage (`measured`) or the CLI estimate (`estimated`). `report.py` gains reliability/token
columns and a corrected group `pass@k`. A new `harness/summary.py` builds the JSON artifact.

**Tech Stack:** Python 3.12 stdlib only. Tests via `pytest`, fully offline. Use `python3`.

---

## Background the implementer needs

- Read `CLAUDE.md` and the spec `docs/superpowers/specs/2026-05-31-cost-reliability-metrics-design.md`.
- The budget guard in `harness/runner.py` uses `action.tokens` / `traj.tokens_used` — keep
  `tokens = input + output` so it is unaffected.
- `RunResult.to_row()` returns `self.__dict__.copy()`, so new fields appear in `results.csv`
  automatically; `write_csv` uses `rows[0].to_row().keys()`.
- Run the suite with `python3 -m pytest -q`. Commit after each task with the exact message.
- Work on the current branch (`master`).

---

## Task 1: Token-split fields on ModelAction, Trajectory, RunResult

**Files:**
- Modify: `harness/adapters.py` (ModelAction), `harness/core.py` (Trajectory, RunResult)
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_core.py`:

```python
from harness.core import Trajectory, RunResult
from harness.adapters import ModelAction


def test_modelaction_has_token_split_defaults():
    a = ModelAction(kind="final")
    assert a.input_tokens == 0 and a.output_tokens == 0
    assert a.token_source == "measured"


def test_trajectory_accumulates_token_split():
    t = Trajectory()
    assert t.input_tokens_used == 0 and t.output_tokens_used == 0


def test_runresult_carries_token_split_in_row():
    r = RunResult(task_id="x", tier=1, category="c", model="m", run_index=0,
                  success=True, n_steps=1, invalid_rate=0.0, tokens_used=30,
                  halt_reason="done", wall_seconds=0.1,
                  input_tokens=20, output_tokens=10, token_source="estimated")
    row = r.to_row()
    assert row["input_tokens"] == 20
    assert row["output_tokens"] == 10
    assert row["token_source"] == "estimated"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_core.py -k "token_split or token_source" -q`
Expected: FAIL (unknown kwargs / attributes).

- [ ] **Step 3: Implement**

In `harness/adapters.py`, extend the `ModelAction` dataclass (add the three fields after
`tokens`):

```python
@dataclass
class ModelAction:
    """Normalized output of one model turn."""
    kind: str                       # "tool_call" | "final"
    tool: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    final_text: str = ""
    raw: str = ""
    tokens: int = 0                 # = input_tokens + output_tokens (drives budget guard)
    input_tokens: int = 0
    output_tokens: int = 0
    token_source: str = "measured"  # "measured" | "estimated"
```

In `harness/core.py`, in the `Trajectory` dataclass add (after `tokens_used`):

```python
    tokens_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
```

In `harness/core.py`, in the `RunResult` dataclass add (after `seed`):

```python
    seed: int = 0                         # the seed this instance was generated from
    input_tokens: int = 0
    output_tokens: int = 0
    token_source: str = "measured"
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_core.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness/adapters.py harness/core.py tests/test_core.py
git commit -m "feat(core): input/output token split + token_source on the data model"
```

---

## Task 2: Adapters populate the token split

**Files:**
- Modify: `harness/adapters.py` (Ollama, Anthropic, OpenAI, Claude-CLI)
- Test: `tests/test_adapters.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_adapters.py` (these mirror how existing adapter tests mock `requests.post`
/ `subprocess.run` — reuse the existing mocking helpers in this file; the snippets below show
the asserts, adapt the mock setup to match the file's existing pattern):

```python
from harness.adapters import (OllamaAdapter, AnthropicAdapter, OpenAIAdapter,
                              ClaudeCliAdapter)


def test_ollama_reports_measured_split(monkeypatch):
    import harness.adapters as A

    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": "Action: list_dir\nArgs: {}"},
                    "prompt_eval_count": 12, "eval_count": 7}
    monkeypatch.setattr(A.requests, "post", lambda *a, **k: _R())
    act = OllamaAdapter("m").act([{"role": "user", "content": "x"}], [], 64)
    assert act.input_tokens == 12 and act.output_tokens == 7
    assert act.tokens == 19 and act.token_source == "measured"


def test_anthropic_reports_measured_split(monkeypatch):
    import harness.adapters as A

    class _R:
        def raise_for_status(self): pass
        def json(self):
            return {"content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 100, "output_tokens": 25}}
    monkeypatch.setattr(A.requests, "post", lambda *a, **k: _R())
    act = AnthropicAdapter("m", "key").act([{"role": "user", "content": "x"}], [], 64)
    assert act.input_tokens == 100 and act.output_tokens == 25
    assert act.tokens == 125 and act.token_source == "measured"


def test_openai_reports_measured_split(monkeypatch):
    import harness.adapters as A

    class _R:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 8,
                              "total_tokens": 48}}
    monkeypatch.setattr(A.requests, "post", lambda *a, **k: _R())
    act = OpenAIAdapter("m", "key").act([{"role": "user", "content": "x"}], [], 64)
    assert act.input_tokens == 40 and act.output_tokens == 8
    assert act.token_source == "measured"


def test_claude_cli_reports_estimated_split(monkeypatch):
    import harness.adapters as A

    class _P:
        stdout = '{"result": "Action: list_dir\\nArgs: {}"}'
        stderr = ""
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: _P())
    act = ClaudeCliAdapter("m").act([{"role": "user", "content": "x"}], [], 64)
    assert act.token_source == "estimated"
    assert act.input_tokens > 0 and act.output_tokens > 0
    assert act.tokens == act.input_tokens + act.output_tokens
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_adapters.py -k "measured or estimated" -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `harness/adapters.py`:

**Ollama** — replace the token line + return in `OllamaAdapter.act`:

```python
        data = r.json()
        text = data.get("message", {}).get("content", "")
        inp = data.get("prompt_eval_count", 0)
        out = data.get("eval_count", 0)
        action = parse_react(text) if self.react else _parse_native(text)
        action.input_tokens, action.output_tokens = inp, out
        action.tokens = inp + out
        action.token_source = "measured"
        return action
```

**Anthropic** — in `AnthropicAdapter.act`, compute the split and set it on every returned
action. Replace the body after `data = r.json()`:

```python
        data = r.json()
        usage = data.get("usage", {})
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)

        def _finish(a: ModelAction) -> ModelAction:
            a.input_tokens, a.output_tokens = inp, out
            a.tokens = inp + out
            a.token_source = "measured"
            return a

        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                if block["name"] == "finish":
                    return _finish(ModelAction(kind="final", raw=json.dumps(data)))
                return _finish(ModelAction(kind="tool_call", tool=block["name"],
                                           args=block.get("input", {}),
                                           raw=json.dumps(data)))
        txt = " ".join(b.get("text", "") for b in data.get("content", []))
        return _finish(ModelAction(kind="final", final_text=txt, raw=txt))
```

**OpenAI** — in `OpenAIAdapter.act`, replace the usage line + returns:

```python
        data = r.json()
        usage = data.get("usage", {})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)

        def _finish(a: ModelAction) -> ModelAction:
            a.input_tokens, a.output_tokens = inp, out
            a.tokens = inp + out
            a.token_source = "measured"
            return a

        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls:
            call = calls[0]["function"]
            if call["name"] == "finish":
                return _finish(ModelAction(kind="final", raw=json.dumps(msg)))
            try:
                args = json.loads(call.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            return _finish(ModelAction(kind="tool_call", tool=call["name"], args=args,
                                       raw=json.dumps(msg)))
        return _finish(ModelAction(kind="final", final_text=msg.get("content", ""),
                                   raw=json.dumps(msg)))
```

**Claude-CLI** — in `ClaudeCliAdapter.act`, replace the final token line:

```python
        action = parse_react(text)
        action.input_tokens = len(prompt) // 4    # see class docstring: estimate
        action.output_tokens = len(text) // 4
        action.tokens = action.input_tokens + action.output_tokens
        action.token_source = "estimated"
        return action
```

(Also update the timeout-path return in `ClaudeCliAdapter.act` to set
`token_source="estimated"` for consistency: add `token_source="estimated"` to that
`ModelAction(...)`.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_adapters.py -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add harness/adapters.py tests/test_adapters.py
git commit -m "feat(adapters): populate input/output token split + token_source"
```

---

## Task 3: Runner accumulates and records the split

**Files:**
- Modify: `harness/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_runner.py` (use a small inline adapter so the token values are explicit):

```python
def test_run_task_records_token_split_and_source():
    from harness.runner import run_task
    from harness.adapters import ModelAction
    from tasks.suite import T1B

    class _Adapter:
        def __init__(self):
            self.calls = 0
        def act(self, messages, specs, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ModelAction(kind="tool_call", tool="write_file",
                                   args={"path": "answer.txt", "content": "1"},
                                   raw="x", tokens=30, input_tokens=20,
                                   output_tokens=10, token_source="estimated")
            return ModelAction(kind="final", raw="done", tokens=15,
                               input_tokens=10, output_tokens=5,
                               token_source="estimated")

    r = run_task(T1B, _Adapter(), "m", 0)
    assert r.input_tokens == 30 and r.output_tokens == 15
    assert r.tokens_used == 45
    assert r.token_source == "estimated"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_runner.py -k token_split -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `harness/runner.py`, inside `run_task`, find the budget-accumulation line
`traj.tokens_used += action.tokens` and expand it, and track the source:

```python
            action = adapter.act(messages, task.tool_specs(), task.max_tokens)
            traj.tokens_used += action.tokens
            traj.input_tokens_used += action.input_tokens
            traj.output_tokens_used += action.output_tokens
            token_source = action.token_source
```

Initialize `token_source = "measured"` just before the `for _ in range(task.max_steps):`
loop (so a zero-step run still has a value):

```python
    token_source = "measured"
    try:
        for _ in range(task.max_steps):
```

In the final `return RunResult(...)`, add the three fields (after `seed=seed,`):

```python
        seed=seed,
        input_tokens=traj.input_tokens_used,
        output_tokens=traj.output_tokens_used,
        token_source=token_source,
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness/runner.py tests/test_runner.py
git commit -m "feat(runner): accumulate + record input/output tokens and token_source"
```

---

## Task 4: report.py — fix group pass@k, add reliability + token columns

**Files:**
- Modify: `harness/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_report.py`:

```python
from harness.core import RunResult
from harness.report import aggregate


def _r(model, task, tier, cat, success, **kw):
    base = dict(task_id=task, tier=tier, category=cat, model=model, run_index=0,
                success=success, n_steps=2, invalid_rate=0.0, tokens_used=30,
                halt_reason="done", wall_seconds=0.1, input_tokens=20,
                output_tokens=10, token_source="measured")
    base.update(kw)
    return RunResult(**base)


def test_tier_passk_is_mean_of_per_task_passk():
    # tier 5 has task A (0/2) and task B (2/2) -> tier pass@k should be 0.5
    rows = [
        _r("m", "A", 5, "coding", False), _r("m", "A", 5, "coding", False),
        _r("m", "B", 5, "agentic", True), _r("m", "B", 5, "agentic", True),
    ]
    agg = aggregate(rows)
    assert agg["by_task"]["m|A"]["pass_at_k"] == 0
    assert agg["by_task"]["m|B"]["pass_at_k"] == 1
    assert agg["by_tier"]["m|tier5"]["pass_at_k"] == 0.5


def test_summarize_has_reliability_and_token_split():
    rows = [_r("m", "A", 1, "data", True, halt_reason="done"),
            _r("m", "A", 1, "data", False, halt_reason="max_steps")]
    s = aggregate(rows)["by_task"]["m|A"]
    assert s["mean_input_tokens"] == 20
    assert s["mean_output_tokens"] == 10
    assert s["done_rate"] == 0.5
    assert s["crash_rate"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_report.py -k "passk or reliability" -q`
Expected: FAIL.

- [ ] **Step 3: Rewrite `aggregate`**

Replace the `aggregate` function in `harness/report.py` with:

```python
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
```

- [ ] **Step 4: Update the markdown tables**

In `write_markdown_report`, replace the by-tier and by-category table blocks so they show the
token split + done%. Replace from the `"## Pass rates by tier\n"` line through the end of the
by-category loop with:

```python
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
```

(Leave the header lines above `## Pass rates by tier` — the title, total-runs, and the
procedural note — unchanged. Leave the per-task detail block unchanged.)

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_report.py -q`
Expected: PASS.

```bash
git add harness/report.py tests/test_report.py
git commit -m "feat(report): fix group pass@k (mean of per-task) + reliability + token split"
```

---

## Task 5: `harness/summary.py` — the eval-summary.json artifact

**Files:**
- Create: `harness/summary.py`
- Test: `tests/test_summary.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/test_summary.py`:

```python
import json
from harness.core import RunResult
from harness.report import aggregate
from harness.summary import build_summary, write_summary


def _r(model, task, tier, cat, success, src="estimated"):
    return RunResult(task_id=task, tier=tier, category=cat, model=model, run_index=0,
                     success=success, n_steps=2, invalid_rate=0.0, tokens_used=30,
                     halt_reason="done", wall_seconds=0.1, input_tokens=20,
                     output_tokens=10, token_source=src)


def _rows():
    return [_r("claude-cli:claude-opus-4-8", "h_merge_intervals", 5, "coding", True),
            _r("claude-cli:claude-opus-4-8", "r_logic_grid", 3, "reasoning", True)]


def test_build_summary_schema_and_mapping():
    rows = _rows()
    summ = build_summary(rows, aggregate(rows), runs_per_task=3)
    assert summ["schema_version"] == "1.0"
    assert summ["runs_per_task"] == 3
    assert summ["suite_version"].startswith("agenteval@")
    m = summ["models"][0]
    assert m["eval_model_id"] == "claude-cli:claude-opus-4-8"
    assert m["helloai_model_id"] == "claude"          # suggested FK
    assert "Opus" in m["display_model"]
    assert m["token_source"] == "estimated"
    assert m["overall"]["pass_at_1"] == 1.0
    assert "mean_input_tokens" in m["overall"]
    assert m["reliability"]["done_rate"] == 1.0
    assert any(t["task_id"] == "h_merge_intervals" for t in m["by_task"])
    assert {b["category"] for b in m["by_category"]} == {"coding", "reasoning"}


def test_write_summary_roundtrips(tmp_path):
    rows = _rows()
    p = tmp_path / "eval-summary.json"
    write_summary(build_summary(rows, aggregate(rows), 3), str(p))
    obj = json.loads(p.read_text())
    assert obj["models"][0]["helloai_model_id"] == "claude"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_summary.py -q`
Expected: FAIL (no module `harness.summary`).

- [ ] **Step 3: Implement `harness/summary.py`**

Create `harness/summary.py`:

```python
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
    return " ".join(p.capitalize() if not any(c.isdigit() for c in p) else p
                    for p in parts).replace("Claude Claude", "Claude")


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
        overall = aggregate(grp)            # single-model aggregate -> reuse summarize
        # the single-model aggregate has one key per tier/task/cat; fold to one overall
        all_succ = [1 if r.success else 0 for r in grp]
        n = len(grp)
        src = next((r.token_source for r in grp
                    if not str(r.halt_reason).startswith("crash")), "measured")

        by_task = [dict(task_id=k.split("|")[1], **_summ_metrics(v),
                        mean_judge=v["mean_judge"])
                   for k, v in sorted(overall["by_task"].items())]
        by_cat = [dict(category=k.split("|")[1], pass_at_1=v["pass_at_1"],
                       pass_at_k=v["pass_at_k"],
                       mean_input_tokens=v["mean_input_tokens"],
                       mean_output_tokens=v["mean_output_tokens"])
                  for k, v in sorted(overall["by_category"].items())]
        by_tier = [dict(tier=int(k.split("|tier")[1]), pass_at_1=v["pass_at_1"],
                        pass_at_k=v["pass_at_k"])
                   for k, v in sorted(overall["by_tier"].items())]

        models.append({
            "eval_model_id": model,
            "helloai_model_id": _suggest_helloai_id(model),
            "display_model": _display_model(model),
            "token_source": src,
            "overall": {
                "pass_at_1": round(sum(all_succ) / n, 3) if n else None,
                "pass_at_k": round(sum(v["pass_at_k"] for v in overall["by_task"].values())
                                   / len(overall["by_task"]), 3) if overall["by_task"] else 0,
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
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_summary.py -q`
Expected: PASS.

> If `_display_model` produces something other than expected for the test
> ("Claude Opus 4.8"), adjust the helper until `test_build_summary_schema_and_mapping`
> passes — the asserted contract (contains "Opus") is what matters.

- [ ] **Step 5: Commit**

```bash
git add harness/summary.py tests/test_summary.py
git commit -m "feat(summary): eval-summary.json emitter for the helloai contract"
```

---

## Task 6: Wire into run_eval.py + docs + green gate

**Files:**
- Modify: `run_eval.py`, `CLAUDE.md`

- [ ] **Step 1: Write the artifact from the CLI**

In `run_eval.py`, add the import near the others:

```python
from harness.summary import build_summary, write_summary
```

After the existing `write_markdown_report(rows, agg, md_path)` call, add:

```python
    summary_path = os.path.join(args.out, "eval-summary.json")
    write_summary(build_summary(rows, agg, args.k), summary_path)
    print(f"Wrote {summary_path}")
```

- [ ] **Step 2: Smoke the artifact offline (no model call)**

Run:
```bash
python3 -c "
from harness.core import RunResult
from harness.report import aggregate
from harness.summary import build_summary
import json
rows=[RunResult(task_id='h_merge_intervals',tier=5,category='coding',model='claude-cli:claude-haiku-4-5',run_index=0,success=True,n_steps=2,invalid_rate=0.5,tokens_used=30,halt_reason='done',wall_seconds=0.1,input_tokens=20,output_tokens=10,token_source='estimated')]
print(json.dumps(build_summary(rows, aggregate(rows), 3), indent=2))
"
```
Expected: a JSON object with `models[0].helloai_model_id == "claude"`, `token_source ==
"estimated"`, and a `reliability.mean_invalid_rate` of 0.5.

- [ ] **Step 3: Update `CLAUDE.md`**

1. Under "Scoring model" or "Methodology guardrails", add:
   "- **Tokens are split input/output** with a `token_source` tag (`measured` for API/Ollama,
   `estimated` for the `chars//4` CLI proxy). The sum still drives the budget guard. agenteval
   emits raw counts only — cost math lives downstream."
2. Add: "- **Tier/category `pass@k` = mean of per-task `pass@k`** (fraction of the group's
   tasks solved ≥1 time in k), not `any()` over pooled runs."
3. Add: "- **`eval-summary.json`** is written to `--out` each run: a typed artifact
   (`harness/summary.py`) for downstream consumers, with overall/by_tier/by_category/by_task
   metrics, reliability, token split, `token_source`, `git_sha`, and a *suggested*
   `helloai_model_id`."

- [ ] **Step 4: Full green gate**

Run: `python3 -m pytest -q`
Expected: ALL PASS. Then confirm CLI wiring: `python3 run_eval.py --help` runs without error.

- [ ] **Step 5: Commit**

```bash
git add run_eval.py CLAUDE.md
git commit -m "feat(cli): emit eval-summary.json; document cost/reliability metrics"
```

---

## Out of scope (do NOT do)

- No cost/dollar math (helloai owns pricing).
- No new model adapters (Gemini/Grok).
- Do NOT run the live `claude` CLI or any paid model — the measured run is operational, done
  separately after this lands.
- Do not change the budget-guard logic (it must keep using the token *total*).

## Self-review notes (author)

- Spec coverage: token split on the data model (T1), adapters populate it (T2), runner records
  it (T3), report pass@k fix + reliability + token columns (T4), eval-summary.json (T5), CLI
  wiring + docs (T6). All spec sections mapped.
- Type consistency: `input_tokens`/`output_tokens`/`token_source` names identical across
  ModelAction, Trajectory(`*_used`), RunResult, report keys, and summary fields. `tokens` =
  input + output everywhere (budget guard unaffected). `done_rate`/`crash_rate`/
  `mean_invalid_rate` names consistent between report.summarize and summary.reliability.
- The pass@k fix has an explicit discriminating test (tier with a 0/k and a k/k task → 0.5).
