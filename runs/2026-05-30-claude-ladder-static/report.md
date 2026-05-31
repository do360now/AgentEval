# Agentic Eval — Results

Total runs: 144

## Pass rates by tier

| Model | Tier | pass@1 | pass@k | mean steps | invalid% | mean tokens | judge |
|---|---|---|---|---|---|---|---|
| claude-cli:claude-haiku-4-5 | tier1 | 1 | 1 | 2.333 | 0% | 968.333 | - |
| claude-cli:claude-haiku-4-5 | tier2 | 1 | 1 | 2.2 | 0% | 1136.933 | - |
| claude-cli:claude-haiku-4-5 | tier3 | 1 | 1 | 3.333 | 0% | 1794 | - |
| claude-cli:claude-haiku-4-5 | tier4 | 1 | 1 | 3.333 | 0% | 1673.667 | - |
| claude-cli:claude-opus-4-8 | tier1 | 1 | 1 | 2 | 0% | 899.5 | - |
| claude-cli:claude-opus-4-8 | tier2 | 1 | 1 | 2.2 | 0% | 1064 | - |
| claude-cli:claude-opus-4-8 | tier3 | 1 | 1 | 3.333 | 0% | 1679.833 | - |
| claude-cli:claude-opus-4-8 | tier4 | 1 | 1 | 3.333 | 0% | 1591.222 | - |
| claude-cli:claude-sonnet-4-6 | tier1 | 1 | 1 | 2 | 0% | 849 | - |
| claude-cli:claude-sonnet-4-6 | tier2 | 1 | 1 | 2.067 | 0% | 973.933 | - |
| claude-cli:claude-sonnet-4-6 | tier3 | 0.944 | 1 | 3.611 | 6% | 1894.778 | - |
| claude-cli:claude-sonnet-4-6 | tier4 | 1 | 1 | 3.333 | 0% | 1500.444 | - |

## Pass rates by capability

| Model | Category | pass@1 | pass@k | mean steps | invalid% | judge |
|---|---|---|---|---|---|---|
| claude-cli:claude-haiku-4-5 | agentic | 1 | 1 | 4.4 | 0% | - |
| claude-cli:claude-haiku-4-5 | coding | 1 | 1 | 2.333 | 0% | - |
| claude-cli:claude-haiku-4-5 | data | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-haiku-4-5 | reasoning | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-haiku-4-5 | retrieval | 1 | 1 | 2.333 | 0% | - |
| claude-cli:claude-opus-4-8 | agentic | 1 | 1 | 4.4 | 0% | - |
| claude-cli:claude-opus-4-8 | coding | 1 | 1 | 2.333 | 0% | - |
| claude-cli:claude-opus-4-8 | data | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-opus-4-8 | reasoning | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-opus-4-8 | retrieval | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-sonnet-4-6 | agentic | 1 | 1 | 4.4 | 0% | - |
| claude-cli:claude-sonnet-4-6 | coding | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-sonnet-4-6 | data | 1 | 1 | 2 | 0% | - |
| claude-cli:claude-sonnet-4-6 | reasoning | 0.889 | 1 | 2.667 | 11% | - |
| claude-cli:claude-sonnet-4-6 | retrieval | 1 | 1 | 2 | 0% | - |

## Per-task detail

| Model | Task | pass@1 | pass@k | mean steps |
|---|---|---|---|---|
| claude-cli:claude-haiku-4-5 | c_code_transform | 1 | 1 | 3 |
| claude-cli:claude-haiku-4-5 | c_fix_bug | 1 | 1 | 3 |
| claude-cli:claude-haiku-4-5 | c_impl_function | 1 | 1 | 1 |
| claude-cli:claude-haiku-4-5 | r_constraint_plan | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | r_logic_grid | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | r_multi_step_math | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | t1a_read_value | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | t1b_count_logs | 1 | 1 | 2.667 |
| claude-cli:claude-haiku-4-5 | t2a_csv_mean | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | t2b_sort_lines | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | t2c_transform | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | t3a_error_counts | 1 | 1 | 5 |
| claude-cli:claude-haiku-4-5 | t3b_sum_files | 1 | 1 | 7 |
| claude-cli:claude-haiku-4-5 | t4a_recover_fallback | 1 | 1 | 3 |
| claude-cli:claude-haiku-4-5 | t4b_skip_bad_rows | 1 | 1 | 2 |
| claude-cli:claude-haiku-4-5 | t4c_sum_skip_corrupt | 1 | 1 | 5 |
| claude-cli:claude-opus-4-8 | c_code_transform | 1 | 1 | 3 |
| claude-cli:claude-opus-4-8 | c_fix_bug | 1 | 1 | 3 |
| claude-cli:claude-opus-4-8 | c_impl_function | 1 | 1 | 1 |
| claude-cli:claude-opus-4-8 | r_constraint_plan | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | r_logic_grid | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | r_multi_step_math | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t1a_read_value | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t1b_count_logs | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t2a_csv_mean | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t2b_sort_lines | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t2c_transform | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t3a_error_counts | 1 | 1 | 5 |
| claude-cli:claude-opus-4-8 | t3b_sum_files | 1 | 1 | 7 |
| claude-cli:claude-opus-4-8 | t4a_recover_fallback | 1 | 1 | 3 |
| claude-cli:claude-opus-4-8 | t4b_skip_bad_rows | 1 | 1 | 2 |
| claude-cli:claude-opus-4-8 | t4c_sum_skip_corrupt | 1 | 1 | 5 |
| claude-cli:claude-sonnet-4-6 | c_code_transform | 1 | 1 | 2.333 |
| claude-cli:claude-sonnet-4-6 | c_fix_bug | 1 | 1 | 2.667 |
| claude-cli:claude-sonnet-4-6 | c_impl_function | 1 | 1 | 1 |
| claude-cli:claude-sonnet-4-6 | r_constraint_plan | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | r_logic_grid | 0.667 | 1 | 4 |
| claude-cli:claude-sonnet-4-6 | r_multi_step_math | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t1a_read_value | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t1b_count_logs | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t2a_csv_mean | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t2b_sort_lines | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t2c_transform | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t3a_error_counts | 1 | 1 | 5 |
| claude-cli:claude-sonnet-4-6 | t3b_sum_files | 1 | 1 | 7 |
| claude-cli:claude-sonnet-4-6 | t4a_recover_fallback | 1 | 1 | 3 |
| claude-cli:claude-sonnet-4-6 | t4b_skip_bad_rows | 1 | 1 | 2 |
| claude-cli:claude-sonnet-4-6 | t4c_sum_skip_corrupt | 1 | 1 | 5 |