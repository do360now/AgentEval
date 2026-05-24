# Eval Run Handoff

**Date:** 2026-05-24
**Goal:** Evaluate the `agents/` defender team. That team runs on exactly two Claude
models — **Opus 4.7** and **Sonnet 4.6** (Haiku 4.5 is unused) — so "evaluating the
team" at the model layer means floor-checking those two. Both were run through this
harness via the new `--claude-cli` adapter (no API key required).

This file is the pick-up point for a fresh instance. The durable architecture/gotchas
live in `CLAUDE.md`; this file is the transient *run status*.

---

## What was added to the harness

- **`ClaudeCliAdapter`** (`harness/adapters.py`) + **`--claude-cli`** flag (`run_eval.py`).
  Scores a Claude model through the local `claude -p` CLI using your existing Claude Code
  subscription auth — **no `ANTHROPIC_API_KEY`**. Two deliberate proxy choices (full
  rationale in the class docstring and `CLAUDE.md` → "Things that will bite you"):
  1. Tools are disabled CLI-side (`--allowed-tools ""`); Claude runs as a pure ReAct text
     model over the harness's own sandbox tools, so it can never touch the real filesystem.
  2. Tokens are **estimated `chars//4`**, not read from the CLI — the CLI's real `usage`
     includes Claude Code's ~30k-token base prompt (visible as `cache_creation_input_tokens`),
     which would trip the budget guard on step 1 and fail every task.
- Verified end-to-end: tier-1 smoke pass, then the full 9-task suite for both models.

## Reproduce

```bash
python3 run_eval.py --claude-cli claude-sonnet-4-6 --k 3
python3 run_eval.py --claude-cli claude-opus-4-7   --k 3
```

~13–20 min per model. Each CLI step cold-starts (`claude -p`) and carries ~30k input
tokens of Claude Code base prompt — invisible to scoring, but it **consumes subscription
quota** (~$10–15-equivalent per full run). Use `python3` (no bare `python` on this box).

---

## Results — k=3, all 4 tiers, CLI proxy path

### pass@1 by tier

| Tier | Shape               | Opus 4.7 | Sonnet 4.6 |
|------|---------------------|:--------:|:----------:|
| 1    | single tool call    | 1.00     | 1.00       |
| 2    | linear pipeline     | 1.00     | 1.00       |
| 3    | conditional/looping | **0.67** | 0.50       |
| 4    | error recovery      | 1.00     | 1.00       |

Both models score a perfect 3/3 on **8 of the 9 tasks** with zero malformed calls. The
entire Opus-vs-Sonnet difference is one task.

### The only discriminating task: `t3a_error_counts`

Goal: count `ERROR` lines in each `logs/*.log`, write the **exact** dict
`{"a.log":2,"b.log":0,"c.log":1}` to `errors.json` (drop the `logs/` prefix, ignore
`readme.txt`). Budget: `max_steps=12`.

| run | Opus 4.7                | Sonnet 4.6              |
|-----|-------------------------|-------------------------|
| 0   | ✓ done (8 steps)        | ✗ max_steps (12)        |
| 1   | ✗ max_steps (12, 1 inv) | ✗ max_steps (12)        |
| 2   | ✗ done, wrong (9)       | ✗ done, wrong (7)       |
| **pass@1 / pass@3** | **0.33 / ✓ (solved once)** | **0.00 / ✗ (never)** |

Classic pass@1-vs-pass@3 signal: **Opus *can* do it (1/3); Sonnet couldn't crack it in 3
tries.** Both flail on the same thing — assembling an exact multi-file dict within the step
budget. The sibling tier-3 task `t3b_sum_files` was a clean 3/3 at 7 steps for **both**, so
tier 3 isn't a wall — `t3a` specifically is.

---

## Where the artifacts are

| File | Holds |
|------|-------|
| `report.md` / `results.csv` | **Opus 4.7** (most recent run — filenames are fixed by the harness) |
| `report-sonnet.md` / `results-sonnet.csv` | **Sonnet 4.6** (manual `cp` snapshot) |

Re-running **overwrites** `report.md`/`results.csv`. Snapshot first (`cp report.md
report-<model>.md`) if you want to keep a run.

---

## Open finding to chase

Is `t3a` failure a real capability gap or an artifact? Two suspects, untangle them:
- **Strict checker** — `check()` requires `== {"a.log":2,"b.log":0,"c.log":1}`. Any
  deviation (kept `logs/` prefix, included `readme.txt`, wrong count) is a hard fail.
- **Proxy inefficiency** — over the stateless ReAct-CLI path, building the dict took 9–12
  steps and hit the budget.

Trajectories are **not** persisted (only `RunResult` rows reach the CSV). To see what
happened, either re-run one `t3a` instance with a `print(traj.as_text())` patched into
`run_task`, or add a `--dump-trajectories` option. Compare an Opus pass vs. a Sonnet fail.

## Caveats on these numbers

- **CLI proxy ≠ native API.** ReAct text + tools-disabled is fair for a floor check but may
  understate the models on `t3a` specifically; native tool-calling tends to be more reliable.
- **No path-quality judge ran** (`t3a/t3b/t4a/t4b` support it). Add `--judge
  anthropic:claude-opus-4-7` — needs an API key — for the secondary score. Outcome scores
  above are deterministic and unaffected.
- **`mean tokens` is the `chars//4` estimate**, not real billing.

---

## Suggested next steps (pick up here)

1. **Confirm the `t3a` cause** — dump the trajectory for one Opus pass + one Sonnet fail.
2. **Native API read** — `export ANTHROPIC_API_KEY=…` then
   `python3 run_eval.py --anthropic claude-opus-4-7 claude-sonnet-4-6 --k 3` for a
   non-proxy comparison (paid path).
3. **Local Ollama spread** — what the suite is actually calibrated for:
   `python3 run_eval.py --ollama qwen2.5:3b llama3.2:3b qwen3:8b --k 3`.
4. **Housekeeping noticed but not done** (didn't act without asking):
   - Not a git repo yet — `git init` if you want history.
   - Stray empty dir `{tasks,harness}/` from a botched brace-expansion `mkdir` — safe to remove.
   - `README.md` and `agenteval_README.md` are byte-identical duplicates — dedup one.
