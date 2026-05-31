# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state — read `HANDOFF.md` first

In-progress run status and the latest results live in **`HANDOFF.md`**; start there to pick
up work. As of 2026-05-24: the `--claude-cli` adapter is in place, and both models the
`agents/` team runs on (Opus 4.7, Sonnet 4.6) have been floor-checked through it. Both
saturate tiers 1/2/4; the one open item is `t3a_error_counts` (the sole task separating the
two models). This section and `HANDOFF.md` are transient; everything below is durable.

## What this is

A reproducible harness for scoring LLMs on **multi-step agentic task completion**.
Local Ollama models and cloud APIs (Anthropic, OpenAI) run through *one identical
scoring path* so results are comparable. The README frames the design as a real
comparative *study*, not a vibe check — the methodology choices below are load-bearing,
not incidental.

## Commands

```bash
pip install requests           # the ONLY runtime dependency

# Local Ollama, 3 repeats:
python run_eval.py --ollama qwen2.5:3b llama3.2:3b --k 3

# Local + cloud + LLM judge:
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...
python run_eval.py --ollama qwen2.5:3b --anthropic claude-opus-4-7 \
    --openai gpt-4o --judge anthropic:claude-opus-4-7 --k 5

# Run a single tier (fast iteration while developing):
python run_eval.py --ollama qwen2.5:3b --tiers 1

# Score Claude WITHOUT an API key, via the local `claude -p` CLI:
python run_eval.py --claude-cli claude-opus-4-7 --k 3
```

Every run writes `results.csv` (raw per-run rows) and `report.md` (aggregated) to
`--out` (default `.`). These filenames are fixed — the `*-sonnet.*` files in the repo
are manually-renamed snapshots of prior runs, not produced by a flag.

### Tests

```bash
python3 -m pytest            # full offline suite (~0.3s, no API key / CLI / Ollama)
python3 -m pytest tests/test_checkers.py -k t4a   # one area
```

The suite under `tests/` is **fully offline**: model providers, the `claude` CLI
(`subprocess.run`), and the network (`requests.post`) are all mocked, and the agent loop
is driven by scripted fake adapters (`tests/conftest.py`). It covers the deterministic
checkers (incl. the t4a error-recovery regression guard), the `Environment` sandbox, the
`run_task`/`run_study` loop and its halt conditions, judge gating, every adapter's parsing,
and report aggregation. There is no linter or build step. Python 3.12 is developed-against;
use `python3` (no bare `python` on this box).

## Architecture

The whole thing is built on strict module boundaries (the code cites Ousterhout):
**the agent loop never sees inside a tool; a tool never sees the grader; the grader
sees only the final environment snapshot + trajectory.** Preserve these boundaries.

```
harness/core.py      Task / Tool / Environment / Trajectory / RunResult — the data model
harness/adapters.py  Ollama / Anthropic / OpenAI / Claude-CLI, all behind one .act()
harness/runner.py    the agent loop (run_task), study orchestration (run_study), LLM judge
harness/report.py    aggregation -> pass@1, pass@k, per-tier/per-task, CSV + markdown
tasks/suite.py       the task suite (16 tasks across 4 tiers; categories: retrieval/data/agentic/coding/reasoning; several tasks are procedurally generated via Task.parametrize) + shared tools
run_eval.py          CLI: builds adapters, filters tiers, runs the matrix, writes output
```

### The two extension points

- **Add a model:** implement a class with one method, `act(messages, tool_specs, max_tokens)`,
  returning a `ModelAction` (`kind="tool_call"|"final"`, plus `tool`, `args`, `tokens`,
  `raw`). Wire it into `build_models` in `run_eval.py`. Nothing else changes — the loop
  is protocol-agnostic by design.
- **Add a task:** append a `Task` to `TASKS` in `tasks/suite.py` with a `setup(env)` that
  populates the sandbox and a `check(env, traj) -> bool` deterministic grader. Pin
  `max_steps`/`max_tokens` per tier; set `judge_path=True` only if you also want path
  quality scored. Coding tasks use `CODE_TOOLS` (adds `run_python`) and grade by executing the model's code against held-out inputs; reasoning tasks use `BASE_TOOLS` and grade an exact answer file. For **procedural** tasks set `parametrize(rng)->dict`; the dict is stashed in `env.scratch['params']` before `setup` and read by both `setup` (to render `problem.txt` / input files) and `check` (for the ground-truth answer). The generator must guarantee a unique, well-formed instance and is guarded by a per-family property test over 200 seeds.

### Scoring model (the core invariant)

- **Outcome is deterministic and always the headline metric.** `check()` runs against
  the FINAL filesystem snapshot, never the model's self-report ("models lie about having
  finished"). Most checkers read a specific output file (e.g. `answer.txt`, `result.txt`)
  and compare its stripped contents to an exact expected value.
- **LLM-judge is secondary and optional.** It scores only *path quality* (efficiency/
  soundness), runs only on tasks with `judge_path=True` AND only when `--judge` is passed,
  and is explicitly told not to re-judge correctness. Never let judge output influence the
  success boolean.

### Methodology guardrails (don't quietly break these)

- **Pinned compute budget per task.** `max_steps` and `max_tokens` live on the `Task`,
  never chosen by the model. Results are only comparable *at a fixed budget* — changing
  these values changes the experiment. The loop also enforces a cumulative ceiling:
  it halts with `halt_reason="token_budget"` once `tokens_used > max_tokens * max_steps`.
- **k repeats; report pass@1 AND pass@k.** Single runs are noisy on small models. pass@1
  is the expected single-shot rate; pass@k (any of k succeeded) is the capability ceiling.
  A large gap = the model *can* do it but isn't reliable. Keep both in any reporting.
- **Graduated tiers (1–4)** so you see *where* a model falls off. Tier 4 ("error recovery,"
  a tool deliberately fails) is the most diagnostic.
- **Procedural generation (contamination resistance).** Seeded tasks instantiate from
  `--seed` + task_id + run_index, so the k repeats are k *distinct* instances and pass@k =
  'solved ≥1 of k distinct instances'. The seed is recorded per row in `results.csv` to
  regenerate any instance.

## Things that will bite you

- **Adapters are stateless; the loop re-sends the full message history every turn.** Don't
  add per-adapter conversation state.
- **`valid` ≠ success.** A well-formed call to a real tool that then raises is still
  `valid=True` — the exception text becomes the observation (this is how tier-4 error
  recovery is tested). Only a nonexistent or unparseable tool call is `valid=False`.
- **A `finish` tool is auto-appended** to cloud adapters' tool lists to signal completion;
  the ReAct text protocol signals done with `Final: done` instead.
- **Ollama defaults to a ReAct text protocol, not native tool-calls.** Many ~4B models
  can't reliably emit structured calls, and scoring a *formatting* failure as a reasoning
  failure would be unfair vs. cloud models. `--ollama-native` forces structured calls.
- **`ClaudeCliAdapter` is a deliberate proxy, not identical to the API path:** tools are
  disabled CLI-side (it runs as a pure ReAct text model over the harness's own sandbox
  tools), and its token counts are *estimated* as `chars//4` — reading the CLI's real usage
  would include Claude Code's multi-thousand-token base prompt and blow the budget guard on
  step 1. Treat its tokens as approximate. See the class docstring before touching it.
- **A crashed run is recorded as a failed `RunResult`** (`halt_reason="crash:..."`), not
  dropped — `run_study` never lets one bad run abort the matrix.
- Models are keyed `provider:model` throughout (e.g. `ollama:qwen2.5:3b`,
  `anthropic:claude-opus-4-7`); `report.py` splits on `|` when building table keys.
- Not a git repository; the two README files (`README.md`, `agenteval_README.md`) are
  byte-identical duplicates.
- **`run_python` executes untrusted generated code on the host** — sandboxed only by temp-dir cwd + `python3 -I` + a 10s timeout, NOT containerized. Fine for trusted models; never point it at adversarial input.
- **`--dump-trajectories`** writes each run's path text to `<out>/trajectories/<model>__<task>__k<i>.txt` for post-hoc diagnosis (trajectories are otherwise not persisted).
- **Tokens are split input/output** with a `token_source` tag (`measured` for API/Ollama from `usage`/`eval_count`, `estimated` `chars//4` for the CLI proxy). `tokens = input + output` still drives the budget guard — the split is additive. agenteval emits raw counts only; cost math lives downstream (helloai).
- **Tier/category `pass@k` = mean of per-task `pass@k`** (fraction of the group's tasks solved ≥1 time in k), NOT `any()` over pooled runs. Per-task `pass@k` is still `any()` over that task's k runs.
- **`eval-summary.json`** is written to `--out` each run (`harness/summary.py`): a typed artifact for downstream consumers with overall/by_tier/by_category/by_task metrics, reliability (invalid/done/crash rates), the token split + `token_source`, a `git_sha` in `suite_version`, and a *suggested* (overridable) `helloai_model_id`.
- **`--anthropic` (raw API) needs prepaid API credits**, which are SEPARATE from a Claude Code subscription — a key can auth fine (`/v1/models` 200) yet every `/v1/messages` call returns 400 "credit balance too low". That billing gap is the whole reason the `--claude-cli` proxy exists. Also: on the API, **Haiku 4.5 is only the dated id `claude-haiku-4-5-20251001`** (the bare `claude-haiku-4-5` alias works for the CLI proxy but 404s on the API); Opus `claude-opus-4-8` and Sonnet `claude-sonnet-4-6` aliases do resolve.