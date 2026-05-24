# Agentic Eval — Results

Total runs: 27

## Pass rates by tier

| Model | Tier | pass@1 | pass@k | mean steps | invalid% | mean tokens | judge |
|---|---|---|---|---|---|---|---|
| claude-cli:claude-opus-4-7 | tier1 | 1 | 1 | 2 | 0% | 906.5 | - |
| claude-cli:claude-opus-4-7 | tier2 | 1 | 1 | 2.111 | 0% | 983.444 | - |
| claude-cli:claude-opus-4-7 | tier3 | 0.667 | 1 | 8.333 | 1% | 6026.667 | - |
| claude-cli:claude-opus-4-7 | tier4 | 1 | 1 | 2.5 | 0% | 1290.667 | - |

## Per-task detail

| Model | Task | pass@1 | pass@k | mean steps |
|---|---|---|---|---|
| claude-cli:claude-opus-4-7 | t1a_read_value | 1 | 1 | 2 |
| claude-cli:claude-opus-4-7 | t1b_count_logs | 1 | 1 | 2 |
| claude-cli:claude-opus-4-7 | t2a_csv_mean | 1 | 1 | 2 |
| claude-cli:claude-opus-4-7 | t2b_sort_lines | 1 | 1 | 2 |
| claude-cli:claude-opus-4-7 | t2c_transform | 1 | 1 | 2.333 |
| claude-cli:claude-opus-4-7 | t3a_error_counts | 0.333 | 1 | 9.667 |
| claude-cli:claude-opus-4-7 | t3b_sum_files | 1 | 1 | 7 |
| claude-cli:claude-opus-4-7 | t4a_recover_fallback | 1 | 1 | 3 |
| claude-cli:claude-opus-4-7 | t4b_skip_bad_rows | 1 | 1 | 2 |