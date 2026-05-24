# Agentic Eval Harness

A small, reproducible harness for scoring LLMs on **multi-step agentic task
completion** — local Ollama models and cloud APIs through one identical scoring
path. Built to support a real comparative *study*, not a vibe check.

## What it measures

Agentic capability is a **trajectory**, not a single answer. For each
(model, task) pair the harness records the full sequence of tool calls and
observations, then scores:

- **Outcome** (deterministic): did the final sandbox state match the expected
  end state? This is the headline metric and never depends on an LLM's opinion.
- **Path quality** (optional LLM-judge): how efficient/sound was the approach?
  Only runs on tasks that opt in via `judge_path=True`, and only affects a
  secondary score — correctness stays deterministic.

Per run it logs: `success`, `n_steps`, `invalid_rate`, `tokens_used`,
`halt_reason`, `wall_seconds`.

## Methodology (the part that makes it a study)

These choices are deliberate and mirror how AISI structured its public agent
evaluations:

1. **Pinned compute budget per tier.** `max_steps` and `max_tokens` are fixed on
   each `Task`, never left to the model. Agent capability scales with
   inference-time compute with no clear plateau, so a model that "fails" at a
   small budget may "pass" at a larger one — comparison is only meaningful at a
   fixed budget.
2. **k repeats, report pass@1 AND pass@k.** Single runs (especially on ~4B
   local models) are noisy. `pass@1` is the expected single-shot rate; `pass@k`
   is the capability ceiling. A large gap = the model *can* do it but isn't
   reliable.
3. **Graduated difficulty tiers.** So you see *where* a model falls off, not
   just an aggregate score.
4. **Grader separated from task.** Deterministic outcome checks are the gold
   standard; LLM-judge is reserved for path quality where correctness can't be
   checked programmatically, and its rationale is logged for auditing.
5. **Document what you do NOT test** (see Limitations).

## Tiers

| Tier | Shape | Example |
|---|---|---|
| 1 | single tool call | read a value from a config file |
| 2 | 2–4 step linear pipeline | read CSV, compute mean, write result |
| 3 | 5–10 step conditional/looping | per-file error counts across a directory |
| 4 | multi-step with **error recovery** | a tool deliberately fails; agent must adapt |

Tier 4 is the most diagnostic: it's the safe analogue of the "recover across
many steps" capability that separates frontier from local models hardest.

## Install & run

```bash
pip install requests

# Local only, 3 repeats:
python run_eval.py --ollama qwen2.5:3b llama3.2:3b --k 3

# Local + cloud + LLM judge, 5 repeats:
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...
python run_eval.py \
    --ollama qwen2.5:3b \
    --anthropic claude-opus-4-7 \
    --openai gpt-4o \
    --judge anthropic:claude-opus-4-7 \
    --k 5

# Score Claude with NO API key, via the local `claude -p` CLI (subscription auth).
# Proxy path: ReAct text, tools disabled, tokens estimated chars//4 (see CLAUDE.md):
python run_eval.py --claude-cli claude-opus-4-7 claude-sonnet-4-6 --k 3

# Just tiers 1–2 (quick local sanity pass):
python run_eval.py --ollama qwen2.5:3b --tiers 1 2 --k 3
```

Outputs `results.csv` (raw per-run rows) and `report.md` (aggregated).

### Local model notes (GTX 1070 / ~4B)

Many small models don't reliably emit structured tool-calls, so the Ollama
adapter defaults to a **ReAct text protocol** (Thought/Action/Args). This avoids
scoring a model as zero for a *formatting* failure rather than a *reasoning*
one — an important fairness point when comparing against cloud models that have
native tool-calling. Pass `--ollama-native` to force structured calls if your
model supports them well.

## Architecture

```
harness/
  core.py      Task / Tool / Environment / Trajectory / RunResult
  adapters.py  Ollama (ReAct or native) + Anthropic + OpenAI, one interface
  runner.py    the agent loop + LLM-as-judge
  report.py    aggregation: pass@1, pass@k, per-tier, CSV + markdown
tasks/
  suite.py     9 graduated tasks with deterministic checkers
run_eval.py    CLI
```

Clean module boundaries (Ousterhout): the agent loop never sees inside a tool;
a tool never sees the grader; the grader sees only the final env snapshot +
trajectory. Adding a model = implement one `.act()` method. Adding a task =
append a `Task` with a `setup` and a deterministic `check`.

## Limitations (state these in any writeup)

- The sandbox is a clean filesystem with **no adversarial conditions** — no
  concurrent processes, no flaky I/O beyond the one deliberate tier-4 failure.
  It measures task-completion reasoning, not robustness to messy real systems.
- Tasks are **English-only** and **short-horizon** (≤14 steps). Long-horizon
  behavior (50+ steps) is not measured.
- The LLM-judge path score is **subjective and model-dependent**; treat it as a
  secondary signal, never as ground truth. The deterministic outcome is the
  real metric.
- Results are only comparable **at the pinned budget**. If you change
  `max_tokens`/`max_steps`, you've changed the experiment.

## Extending

The natural next tiers to add (all safe): tasks needing **information
synthesis** across multiple files, **stateful** tasks where order matters,
**ambiguous** goals that require the agent to ask or assume, and **distractor**
environments with irrelevant files that test focus.
