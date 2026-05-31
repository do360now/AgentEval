# Agentic Eval — Results

Total runs: 2

_Procedurally-generated tasks: each repeat is a distinct seeded instance, so pass@k = solved ≥ 1 of k distinct instances._

## Pass rates by tier

| Model | Tier | pass@1 | pass@k | mean steps | invalid% | mean tokens | judge |
|---|---|---|---|---|---|---|---|
| claude-cli:claude-haiku-4-5 | tier5 | 1 | 1 | 2.5 | 25% | 1889 | - |

## Pass rates by capability

| Model | Category | pass@1 | pass@k | mean steps | invalid% | judge |
|---|---|---|---|---|---|---|
| claude-cli:claude-haiku-4-5 | agentic | 1 | 1 | 3 | 0% | - |
| claude-cli:claude-haiku-4-5 | coding | 1 | 1 | 2 | 50% | - |

## Per-task detail

| Model | Task | pass@1 | pass@k | mean steps |
|---|---|---|---|---|
| claude-cli:claude-haiku-4-5 | h_merge_intervals | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | h_revenue_by_region | 1 | 1 | 3 |