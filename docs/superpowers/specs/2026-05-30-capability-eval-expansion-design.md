# Capability-Eval Expansion — Design

**Date:** 2026-05-30
**Status:** Approved, ready for implementation plan
**Goal:** Expand `agenteval` from a 9-task filesystem suite into a harness that scores
models on **agentic**, **coding**, and **reasoning** capability, reported per capability
category, then run the full Claude ladder (Opus 4.8 → Sonnet 4.6 → Haiku 4.5) through the
CLI-proxy path.

## Context & constraints

- **Run path is the `claude -p` CLI proxy** (no API key, subscription auth). Claude runs as
  a pure ReAct *text* model over the harness's own sandbox tools; tools are disabled
  CLI-side so generated code can never touch the real filesystem. Token counts are the
  harness `chars//4` estimate. See `ClaudeCliAdapter` docstring + `CLAUDE.md`.
- **Core invariants are preserved** (do not break): deterministic `check()` against the
  FINAL sandbox snapshot; pinned `max_steps`/`max_tokens` per task; pass@1 AND pass@k
  reported; strict module boundaries (loop ⊥ tool ⊥ grader). See `CLAUDE.md`.
- Models for the final run: `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
  **k=3**, all tiers.

## 1. New capability dimensions (task suite)

Add two capability categories and deepen the agentic one. Tier (1–4 difficulty, sets the
budget) is orthogonal to category (capability type).

### Coding (`category="coding"`) — requires the new `run_python` tool

| task_id | tier | what | grading |
|---|---|---|---|
| `c_impl_function` | 3 | Implement a function to a written spec in `solution.py`. | `check()` runs `solution.py` against **held-out** inputs via subprocess and compares stdout to expected. Held-out inputs are NOT shown in the goal, defeating hardcoding. |
| `c_fix_bug` | 3 | A `buggy.py` is provided that crashes / returns wrong output. Fix it so it produces the correct result. | `check()` executes the fixed file and compares output. |
| `c_code_transform` | 2 | Solve a small data problem by writing+running code (not by hand). | `check()` reads the output file the model's code wrote. |

The intended loop: model uses `write_file` to author code, `run_python` to execute and
observe errors, iterates, then signals done. This naturally exercises the tier-4 error-
recovery muscle within coding.

### Reasoning (`category="reasoning"`) — uses only the existing read/write tools

Pure deduction, no tool-use confound. Model reads the problem and writes an exact answer.

| task_id | tier | what | grading |
|---|---|---|---|
| `r_logic_grid` | 3 | Small multi-constraint deduction puzzle. | exact answer string to `answer.txt`. |
| `r_multi_step_math` | 2 | Word problem requiring several dependent steps. | exact integer to `answer.txt`. |
| `r_constraint_plan` | 3 | Output a valid ordering satisfying given constraints. | `check()` validates the produced ordering against the constraints (a valid order, not one fixed string). |

### Agentic (deepen)

Keep all 9 existing tasks. Add **one** new multi-tool tier-4 task that chains
list→read→compute→write with a deliberately-failing step, distinct from the existing
`t4a`/`t4b`.

## 2. Harness improvements

1. **`run_python` tool** (`tasks/suite.py`): executes model-written code in the sandbox
   temp dir via `subprocess.run(["python3", "-I", path], cwd=env.root, timeout=10,
   capture_output=True, text=True)`. Returns `{stdout, stderr, returncode}` (truncated) as
   the observation. A timeout returns an error observation (a real, recoverable failure —
   `valid=True`). **Security note:** runs untrusted generated Python on the host, sandboxed
   to a temp dir + 10s timeout but NOT containerized; acceptable for trusted Claude models
   on this box. Documented in the tool docstring and `CLAUDE.md`.

2. **Robust ReAct argument parsing** (`harness/adapters.py`): the current
   `_ARGS_RE = re.compile(r"Args:\s*(\{.*?\})", re.S)` is non-greedy and truncates at the
   first `}`, breaking on multi-line code args. Replace with **balanced-brace extraction**:
   find the first `{` after `Args:`, scan counting brace depth (skipping braces inside JSON
   string literals), capture through the matching `}`, then `json.loads`. Falls back to the
   current "invalid → model retries" behavior on parse failure. This is load-bearing for
   coding-over-ReAct and strictly improves every ReAct task. Must remain covered by adapter
   tests.

3. **Per-category aggregation** (`harness/report.py`): add a `by_category` grouping
   alongside `by_task`/`by_tier`, and a "Pass rates by capability" table in the markdown
   report (category | pass@1 | pass@k | mean steps | invalid% | judge). This is the headline
   answer to "how good is each model at coding vs reasoning vs agentic".

4. **`--dump-trajectories`** (`run_eval.py` + `runner.py`): when set, write each run's
   `traj.as_text()` to `--out/trajectories/<model>__<task>__k<i>.txt`. Closes the open ask
   in `HANDOFF.md` and lets us diagnose failures without re-running with print patches.
   `run_task` gains an optional `trajectory_sink` callback so the runner stays decoupled
   from file IO (boundary-preserving).

## 3. Testing (TDD, fully offline)

Mirror the existing offline style (`tests/`, mocked subprocess/network, scripted fake
adapters in `tests/conftest.py`). New/changed coverage:

- New checkers for every new task (coding checkers use a real short-lived temp dir + real
  `python3` since execution is the point; keep them <0.5s each, or mock if the suite must
  stay network/binary-free — prefer real `python3` as it's present).
- `run_python` tool: success, nonzero exit, timeout, output-file side effects.
- Balanced-brace `Args:` parser: multi-line JSON, nested braces, braces inside strings,
  malformed → invalid.
- `by_category` aggregation in `test_report.py`.
- `--dump-trajectories` sink wiring in `test_runner.py`.

Full suite must stay green and fast. Develop test-first per task.

## 4. The run (after harness is green offline)

1. **Smoke**: one tier-1 task through the CLI for each of the three model IDs to confirm
   they resolve (`python3 run_eval.py --claude-cli claude-opus-4-8 --tiers 1 --k 1`).
2. **Full matrix in background**: `--claude-cli claude-opus-4-8 claude-sonnet-4-6
   claude-haiku-4-5 --k 3 --dump-trajectories`. Snapshot `report.md`/`results.csv` per
   model (`cp report.md report-<model>.md`) since filenames are fixed and overwritten.
3. **Deliverable**: a comparison report with pass@1/pass@k broken down **by capability
   category** across the three models, plus saved trajectories for any discriminating tasks.

## Explicitly out of scope (YAGNI)

- Unrelated cleanup (README dedup, stray dirs) unless trivial.
- Network sandboxing / containerization for `run_python` beyond temp-dir + timeout.
- Native-API (`--anthropic`) run; deferred to the CLI-proxy path per decision.
- New model adapters; the four existing providers are sufficient.

## Files touched

- `tasks/suite.py` — `run_python` tool + new coding/reasoning/agentic tasks.
- `harness/adapters.py` — balanced-brace `Args:` parser.
- `harness/report.py` — `by_category` aggregation + report table.
- `harness/runner.py` — optional `trajectory_sink`.
- `run_eval.py` — `--dump-trajectories` flag + wiring.
- `tests/*` — coverage for all of the above.
- `CLAUDE.md` — document `run_python` security note, new categories, new flag.
