import json

from harness.core import RunResult
from harness.report import aggregate
from harness.summary import build_summary, write_summary


def _r(model, task, tier, cat, success, src="estimated"):
    return RunResult(task_id=task, tier=tier, category=cat, model=model, run_index=0,
                     success=success, n_steps=2, invalid_rate=0.0, tokens_used=30,
                     halt_reason="done", wall_seconds=0.1, input_tokens=20,
                     output_tokens=10, token_source=src)


def _rows():
    return [_r("claude-cli:claude-opus-4-8", "h_merge_intervals", 5, "coding", True),
            _r("claude-cli:claude-opus-4-8", "r_logic_grid", 3, "reasoning", True)]


def test_build_summary_schema_and_mapping():
    rows = _rows()
    summ = build_summary(rows, aggregate(rows), runs_per_task=3)
    assert summ["schema_version"] == "1.0"
    assert summ["runs_per_task"] == 3
    assert summ["suite_version"].startswith("agenteval@")
    m = summ["models"][0]
    assert m["eval_model_id"] == "claude-cli:claude-opus-4-8"
    assert m["helloai_model_id"] == "claude"          # suggested FK
    assert "Opus" in m["display_model"]
    assert m["token_source"] == "estimated"
    assert m["overall"]["pass_at_1"] == 1.0
    assert "mean_input_tokens" in m["overall"]
    assert m["reliability"]["done_rate"] == 1.0
    assert any(t["task_id"] == "h_merge_intervals" for t in m["by_task"])
    assert {b["category"] for b in m["by_category"]} == {"coding", "reasoning"}


def test_write_summary_roundtrips(tmp_path):
    rows = _rows()
    p = tmp_path / "eval-summary.json"
    write_summary(build_summary(rows, aggregate(rows), 3), str(p))
    obj = json.loads(p.read_text())
    assert obj["models"][0]["helloai_model_id"] == "claude"
