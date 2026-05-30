"""Tests for aggregation and output writing (pass@1, pass@k, CSV, markdown)."""
from __future__ import annotations

import csv

from harness.core import RunResult
from harness.report import aggregate, write_csv, write_markdown_report


def _result(model="m", task="t1", tier=1, run_index=0, success=True,
            n_steps=3, invalid_rate=0.0, tokens=100, judge=None):
    return RunResult(
        task_id=task, tier=tier, category="cat", model=model,
        run_index=run_index, success=success, n_steps=n_steps,
        invalid_rate=invalid_rate, tokens_used=tokens, halt_reason="done",
        wall_seconds=0.1, judge_score=judge)


def test_pass_at_1_is_mean_success():
    rows = [_result(success=True, run_index=0),
            _result(success=True, run_index=1),
            _result(success=False, run_index=2)]
    agg = aggregate(rows)
    summary = agg["by_task"]["m|t1"]
    assert summary["n_runs"] == 3
    assert summary["pass_at_1"] == round(2 / 3, 3)
    assert summary["pass_at_k"] == 1          # at least one success


def test_pass_at_k_zero_when_all_fail():
    rows = [_result(success=False, run_index=i) for i in range(3)]
    summary = aggregate(rows)["by_task"]["m|t1"]
    assert summary["pass_at_1"] == 0.0
    assert summary["pass_at_k"] == 0


def test_aggregate_groups_by_tier():
    rows = [_result(task="t1", tier=1, success=True),
            _result(task="t2", tier=1, success=False),
            _result(task="t3", tier=2, success=True)]
    agg = aggregate(rows)
    assert "m|tier1" in agg["by_tier"]
    assert "m|tier2" in agg["by_tier"]
    # tier 1 mixes a pass and a fail -> pass@1 = 0.5
    assert agg["by_tier"]["m|tier1"]["pass_at_1"] == 0.5


def test_mean_judge_ignores_none():
    rows = [_result(judge=None), _result(judge=0.4), _result(judge=0.6)]
    summary = aggregate(rows)["by_task"]["m|t1"]
    assert summary["mean_judge"] == 0.5       # the None is dropped


def test_mean_judge_none_when_all_none():
    rows = [_result(judge=None), _result(judge=None)]
    assert aggregate(rows)["by_task"]["m|t1"]["mean_judge"] is None


def test_separate_models_are_distinct_keys():
    rows = [_result(model="a", success=True), _result(model="b", success=False)]
    agg = aggregate(rows)
    assert agg["by_task"]["a|t1"]["pass_at_1"] == 1.0
    assert agg["by_task"]["b|t1"]["pass_at_1"] == 0.0


def test_write_csv_roundtrip(tmp_path):
    rows = [_result(success=True), _result(success=False, run_index=1)]
    path = tmp_path / "results.csv"
    write_csv(rows, str(path))

    with open(path, newline="") as f:
        read_back = list(csv.DictReader(f))
    assert len(read_back) == 2
    assert read_back[0]["model"] == "m"
    assert read_back[0]["success"] == "True"
    assert "tokens_used" in read_back[0]


def test_write_markdown_report(tmp_path):
    rows = [_result(success=True), _result(success=False, run_index=1)]
    path = tmp_path / "report.md"
    write_markdown_report(rows, aggregate(rows), str(path))

    text = path.read_text()
    assert "Pass rates by tier" in text
    assert "Per-task detail" in text
    assert "Total runs: 2" in text


# --------------------------------------------------------------------------- #
# by_category aggregation
# --------------------------------------------------------------------------- #
from harness.core import RunResult
from harness.report import aggregate


def _row(model, task_id, category, tier, success):
    return RunResult(task_id=task_id, tier=tier, category=category, model=model,
                     run_index=0, success=success, n_steps=3, invalid_rate=0.0,
                     tokens_used=100, halt_reason="done", wall_seconds=0.1)


def test_aggregate_has_by_category():
    rows = [
        _row("m", "c_impl_function", "coding", 3, True),
        _row("m", "c_fix_bug", "coding", 3, False),
        _row("m", "r_logic_grid", "reasoning", 3, True),
    ]
    agg = aggregate(rows)
    assert "by_category" in agg
    assert agg["by_category"]["m|coding"]["pass_at_1"] == 0.5
    assert agg["by_category"]["m|coding"]["pass_at_k"] == 1
    assert agg["by_category"]["m|reasoning"]["pass_at_1"] == 1.0
