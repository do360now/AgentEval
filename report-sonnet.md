# Agentic Eval — Results

Total runs: 27

## Pass rates by tier

| Model | Tier | pass@1 | pass@k | mean steps | invalid% | mean tokens | judge |
|---|---|---|---|---|---|---|---|
| claude-cli:claude-sonnet-4-6 | tier1 | 1 | 1 | 2 | 0% | 847.333 | - |
| claude-cli:claude-sonnet-4-6 | tier2 | 1 | 1 | 2 | 0% | 872.111 | - |
| claude-cli:claude-sonnet-4-6 | tier3 | 0.5 | 1 | 8.667 | 0% | 4760 | - |
| claude-cli:claude-sonnet-4-6 | tier4 | 1 | 1 | 2.5 | 0% | 1149.5 | - |

## Per-task detail

| Model | Task | pass@1 | pass@k | mean steps |
|---|---|---|---|---|
| claude-cli:claude-sonnet-4-6 | t1a_read_value | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t1b_count_logs | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t2a_csv_mean | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t2b_sort_lines | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t2c_transform | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t3a_error_counts | 0 | 0 | 10.333 |
| claude-cli:claude-sonnet-4-6 | t3b_sum_files | 1 | 1 | 7 |
| claude-cli:claude-sonnet-4-6 | t4a_recover_fallback | 1 | 1 | 3 |
| claude-cli:claude-sonnet-4-6 | t4b_skip_bad_rows | 1 | 1 | 2 |