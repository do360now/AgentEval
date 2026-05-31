# Cost + Reliability Metrics (+ eval-summary.json) — Design

**Date:** 2026-05-31
**Status:** Approved, ready for implementation plan
**Author:** Opus 4.8 (lead). Implementer: Sonnet 4.6.
**Goal:** Make the harness rank models on the axes that actually separate frontier models —
**cost** (input/output tokens) and **reliability** (invalid-call rate, halt outcomes) — since
the k=3 hard-set run proved outcome pass/fail is saturated. Also emit the machine-readable
`eval-summary.json` artifact the helloai leaderboard ingests.

## Why (evidence)

The k=3 hard-discriminator run: all three models scored **pass@1 = 1.00** on both tasks.
The only separation was **reliability** (Haiku ~33% invalid-call rate on the hard coding task
vs 0% for Opus/Sonnet) and **efficiency** (steps/tokens). Those signals already live in
`results.csv` but are not surfaced as headline metrics, and tokens are recorded only as an
undifferentiated total. This piece promotes them.

## Approved decisions

1. **CLI-proxy tokens: estimated split, tagged.** The `claude -p` proxy reports
   `input_tokens = len(prompt)//4`, `output_tokens = len(reply)//4`, `token_source="estimated"`.
   Real **measured** split comes only from the API adapters (`--anthropic`/`--openai`/Ollama).
   helloai computes dollars only from `measured` rows (or shows estimated with a caveat).
2. **Bundle the eval-summary.json emitter** in this piece — the cost numbers exist to feed it.
3. **agenteval stays pricing-agnostic**: emits raw counts + `token_source`; helloai owns cost
   math and the model FK (we emit a *suggested*, overridable `helloai_model_id`).

## Data model changes (the split)

Add an input/output token split alongside the existing total (which **keeps driving the budget
guard** — methodology guardrail untouched).

- **`ModelAction`** (`harness/adapters.py`): `input_tokens: int = 0`, `output_tokens: int = 0`,
  `token_source: str = "measured"`. `tokens` stays = `input + output`.
- **`Trajectory`** (`harness/core.py`): `input_tokens_used: int = 0`,
  `output_tokens_used: int = 0` (alongside `tokens_used`).
- **`RunResult`** (`harness/core.py`): `input_tokens: int = 0`, `output_tokens: int = 0`,
  `token_source: str = "measured"`. `to_row()` picks them up → new CSV columns automatically.

### Adapters populate the split from what they actually have
| adapter | input | output | token_source |
|---|---|---|---|
| Ollama | `prompt_eval_count` | `eval_count` | `measured` |
| Anthropic | `usage.input_tokens` | `usage.output_tokens` | `measured` |
| OpenAI | `usage.prompt_tokens` | `usage.completion_tokens` | `measured` |
| Claude-CLI | `len(prompt)//4` | `len(reply)//4` | `estimated` |

Each sets `tokens = input + output` so the budget guard is unchanged.

### Runner
The loop accumulates all three counters; `run_task` records `input_tokens`/`output_tokens`
on the result and captures `token_source` from the action (uniform per adapter). Crash rows
keep defaults (0 tokens).

## Reliability (surface what's already recorded)

In `harness/report.py::aggregate`, `summarize` gains:
- `mean_invalid_rate` (already present — promote to headline),
- `done_rate` = fraction of runs with `halt_reason == "done"`,
- `crash_rate` = fraction with `halt_reason` starting `"crash"`,
- `mean_input_tokens`, `mean_output_tokens`.

The markdown report shows invalid% and done% as columns (per tier and per category), and a
short **Reliability** subsection per model. Transparent components — **no opaque composite
score** (so the ranking stays interpretable).

## Fix the tier/category `pass@k` aggregation (helloai-flagged bug)

Currently tier/category `pass@k = any(success)` over the *pooled* runs (`report.py`), which
reads 1 even when a constituent task is 0/k. Change tier/category `pass@k` to the **mean of
per-task `pass@k`** (fraction of that group's tasks solved ≥1 time in k tries). Per-task
`pass@k` is unchanged (`any()` over that task's runs). helloai mirrors this definition.

## `eval-summary.json` emitter (new module `harness/summary.py`)

`build_summary(rows, agg, runs_per_task) -> dict`, `write_summary(summary, path)`. Written to
`<out>/eval-summary.json` by `run_eval.py` after the CSV/markdown.

Schema:
```json
{
  "schema_version": "1.0",
  "suite_version": "agenteval@<git_short_sha>",
  "generated_at": "<ISO8601 UTC>",
  "runs_per_task": 3,
  "models": [{
    "eval_model_id": "claude-cli:claude-opus-4-8",
    "helloai_model_id": "claude",        // SUGGESTED; helloai may override
    "display_model": "Claude Opus 4.8",
    "token_source": "estimated",
    "overall":     { "pass_at_1": 1.0, "pass_at_k": 1.0, "mean_steps": 2.4,
                     "mean_input_tokens": 0, "mean_output_tokens": 0 },
    "reliability": { "mean_invalid_rate": 0.0, "done_rate": 1.0, "crash_rate": 0.0 },
    "by_category": [{ "category": "coding", "pass_at_1": 1.0, "pass_at_k": 1.0,
                      "mean_input_tokens": 0, "mean_output_tokens": 0 }],
    "by_tier":     [{ "tier": 5, "pass_at_1": 1.0, "pass_at_k": 1.0 }],
    "by_task":     [{ "task_id": "h_merge_intervals", "pass_at_1": 1.0, "pass_at_k": 1.0,
                      "mean_steps": 1.7, "mean_input_tokens": 0, "mean_output_tokens": 0,
                      "mean_judge": null }]
  }]
}
```

Helpers in the emitter:
- `git_short_sha`: `git rev-parse --short HEAD` (fallback `"unknown"`).
- `display_model` + suggested `helloai_model_id`: a small static mapper keyed on the model id
  substring (`opus|sonnet|haiku → claude`, `gpt|o1|o3 → gpt`, `gemini → gemini`,
  `grok → grok`, else the provider prefix). Display name derived from the id.
- `token_source` per model: taken from that model's non-crash rows.

## What changes, exactly

`harness/adapters.py` (ModelAction + 4 adapters), `harness/core.py` (Trajectory, RunResult),
`harness/runner.py` (accumulate + record split/source), `harness/report.py` (pass@k fix,
reliability + token columns), `harness/summary.py` (NEW), `run_eval.py` (write the artifact),
`tests/*` (adapters, core, runner, report, new test_summary), `CLAUDE.md` (document the split,
token_source, the pass@k change, and the artifact).

## Testing (TDD, offline)

- Adapters: each sets `input_tokens`/`output_tokens`/`token_source` from a mocked response;
  `tokens == input + output`.
- Core/runner: `Trajectory` accumulates the split; `RunResult` carries it + `token_source`
  and emits them in `to_row()`/CSV.
- Report: **the pass@k fix is the key test** — a model with two tasks in a tier (one 0/k, one
  k/k) yields tier `pass@k == 0.5`, not 1. Plus reliability fields and token means present.
- Summary: `build_summary` emits the schema with `token_source`, a `git_sha`-bearing
  `suite_version`, the suggested `helloai_model_id` (`claude` for a claude id), and
  per-task/category/tier blocks.

## Out of scope (YAGNI)

- No pricing/cost-dollar math in agenteval (helloai owns it).
- No new model adapters (Gemini/Grok) — separate effort.
- The paid `--anthropic` measured run is operational, not code; run after this lands and the
  API key is configured.
- Agent-safety tasks — separate roadmap item.
