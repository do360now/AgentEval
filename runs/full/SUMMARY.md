# Claude ladder run — 2026-05-30 (static 16-task suite, CLI proxy, k=3)

**Models:** claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5 (via `claude -p` proxy).
**Matrix:** 16 tasks × 3 models × k=3 = **144 runs**. Tokens are the CLI `chars//4` estimate.

## Headline

**143 / 144 runs passed.** The entire Claude ladder — including the *smallest* model — is at
ceiling on this suite.

| Model | tasks perfect | runs passed | pass@1 (overall) |
|---|---|---|---|
| Claude Opus 4.8 | 16 / 16 | 48 / 48 | 1.00 |
| Claude Haiku 4.5 | 16 / 16 | 48 / 48 | 1.00 |
| Claude Sonnet 4.6 | 15 / 16 | 47 / 48 | ~0.98 |

The **only** non-pass in the whole matrix: Sonnet on `r_logic_grid` (2/3 — one run hit
`max_steps` with a malformed call). pass@k = 1 everywhere. Zero invalid calls except that one
Sonnet run. No judge scores (`--judge` not passed).

## Interpretation

- **The suite does not discriminate current frontier Claude models.** Coding-by-execution and
  the new reasoning tasks, which were meant to stretch the models, are also saturated — even
  Haiku 4.5 clears them 3/3. At this difficulty, capability is not the differentiating axis.
- **The real separation is cost/speed, not success.** On the heavy agentic tasks Opus spends
  noticeably more wall-time for the *same* outcome (e.g. `t3b_sum_files`: Opus ~61s vs Sonnet
  ~36s vs Haiku ~39s; `t3a_error_counts`: Opus ~49s vs ~31s/~33s). For identical pass rates,
  Sonnet/Haiku finish faster and (on a measured-token run) cheaper — the "capability per
  dollar" story.
- This **validates the roadmap**: procedural generation (fresh, un-memorizable instances —
  now being implemented) plus harder tier-4/agentic tasks are required before this suite can
  separate frontier models. The single Sonnet blip is noise, not signal.

## Caveats

- CLI-proxy path: ReAct text + tools-disabled; tokens estimated. Capability ranking is fair;
  cost numbers are not billing-grade (needs an `--anthropic` run).
- Static (pre-procedural) suite: these exact instances could in principle be in training data.
  Procedural generation addresses this next.

Snapshot preserved at `runs/2026-05-30-claude-ladder-static/`.
