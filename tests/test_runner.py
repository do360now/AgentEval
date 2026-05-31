"""Tests for the agent loop (run_task), study orchestration (run_study), and the
LLM-as-judge gating. All model turns are scripted via fake adapters."""
from __future__ import annotations

from harness.adapters import ModelAction
from harness.core import Task, Tool
from harness.runner import judge_path_quality, run_study, run_task
from tasks.suite import T4A

from tests.conftest import AlwaysAdapter, CrashAdapter, ProgrammedAdapter


# --------------------------------------------------------------------------- #
# A tiny self-contained task: one tool that writes a fixed file.
# --------------------------------------------------------------------------- #
def _touch(env, name):
    env.write(name, "x")
    return f"created {name}"


TOUCH = Tool("touch", "Create a file.",
             {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"]}, _touch)


def _make_task(check=lambda env, traj: env.path("done.txt").exists(),
               max_steps=4, max_tokens=512, judge_path=False):
    return Task("test_task", 1, "test", "create done.txt", [TOUCH],
                setup=lambda env: None, check=check,
                max_steps=max_steps, max_tokens=max_tokens, judge_path=judge_path)


def _call(tool, **args):
    return ModelAction(kind="tool_call", tool=tool, args=args)


FINISH = ModelAction(kind="final")


# --------------------------------------------------------------------------- #
# run_task: happy path
# --------------------------------------------------------------------------- #
def test_run_task_success_and_metrics():
    adapter = ProgrammedAdapter([_call("touch", name="done.txt"), FINISH])
    res = run_task(_make_task(), adapter, "fake", run_index=0)

    assert res.success is True
    assert res.halt_reason == "done"
    assert res.n_steps == 1            # the finish turn is not a step
    assert res.invalid_rate == 0.0
    assert res.model == "fake"
    assert res.wall_seconds >= 0.0


def test_run_task_failure_when_goal_unmet():
    # Finishes immediately without ever creating done.txt.
    res = run_task(_make_task(), ProgrammedAdapter([FINISH]), "fake", 0)
    assert res.success is False
    assert res.halt_reason == "done"
    assert res.n_steps == 0


# --------------------------------------------------------------------------- #
# run_task: halting conditions
# --------------------------------------------------------------------------- #
def test_run_task_hits_max_steps():
    # Never finishes: keeps calling a valid tool until the step budget runs out.
    adapter = AlwaysAdapter(_call("touch", name="loop.txt"))
    res = run_task(_make_task(max_steps=3), adapter, "fake", 0)
    assert res.halt_reason == "max_steps"
    assert res.n_steps == 3


def test_run_task_token_budget_guard():
    # budget = max_tokens * max_steps = 10 * 2 = 20; one 100-token turn blows it.
    big = ModelAction(kind="tool_call", tool="touch", args={"name": "x"},
                      tokens=100)
    res = run_task(_make_task(max_steps=2, max_tokens=10),
                   AlwaysAdapter(big), "fake", 0)
    assert res.halt_reason == "token_budget"
    assert res.n_steps == 0            # cut before the tool ran
    assert res.tokens_used == 100


# --------------------------------------------------------------------------- #
# run_task: invalid / unknown tool handling
# --------------------------------------------------------------------------- #
def test_run_task_unknown_tool_is_invalid_then_recovers():
    adapter = ProgrammedAdapter([
        _call("ghost", foo=1),                 # not a real tool -> invalid
        _call("touch", name="done.txt"),       # real tool -> valid
        FINISH,
    ])
    res = run_task(_make_task(), adapter, "fake", 0)
    assert res.success is True
    assert res.n_steps == 2
    assert res.invalid_rate == 0.5


def test_run_task_tool_exception_is_valid_observation():
    # A real tool that raises is a *valid* call with an ERROR observation,
    # not an invalid step. We assert via a check that inspects the trajectory.
    def boom(env):
        raise RuntimeError("kaboom")

    boom_tool = Tool("boom", "always fails",
                     {"type": "object", "properties": {}}, boom)

    captured = {}

    def check(env, traj):
        captured["steps"] = list(traj.steps)
        return True

    task = Task("boomtask", 1, "test", "go", [boom_tool],
                lambda env: None, check, max_steps=3, max_tokens=128)
    res = run_task(task, ProgrammedAdapter([_call("boom"), FINISH]), "fake", 0)

    step = captured["steps"][0]
    assert step.valid is True
    assert "ERROR" in str(step.observation)
    assert res.invalid_rate == 0.0


# --------------------------------------------------------------------------- #
# run_task: tier-4 error recovery, end to end through the loop
# --------------------------------------------------------------------------- #
def test_t4a_full_recovery_through_loop():
    adapter = ProgrammedAdapter([
        _call("read_file", path="primary.json"),   # missing -> ERROR obs
        _call("read_file", path="backup.json"),    # exists -> {"port": 8080}
        _call("write_file", path="port.txt", content="8080"),
        FINISH,
    ])
    res = run_task(T4A, adapter, "fake", 0)
    assert res.success is True


def test_t4a_guess_without_reading_fails_through_loop():
    # Writes the right value but never attempts primary.json -> no recovery.
    adapter = ProgrammedAdapter([
        _call("write_file", path="port.txt", content="8080"),
        FINISH,
    ])
    res = run_task(T4A, adapter, "fake", 0)
    assert res.success is False


# --------------------------------------------------------------------------- #
# LLM-as-judge gating (path quality only; never touches correctness)
# --------------------------------------------------------------------------- #
def _judge(score=0.8, rationale="clean"):
    payload = f'{{"score": {score}, "rationale": "{rationale}"}}'
    return ProgrammedAdapter([ModelAction(kind="final", final_text=payload)])


def test_judge_runs_when_opted_in():
    task = _make_task(judge_path=True)
    adapter = ProgrammedAdapter([_call("touch", name="done.txt"), FINISH])
    res = run_task(task, adapter, "fake", 0, judge_adapter=_judge(0.9))
    assert res.judge_score == 0.9
    assert res.judge_rationale == "clean"


def test_judge_skipped_when_not_opted_in():
    task = _make_task(judge_path=False)
    judge = _judge()
    adapter = ProgrammedAdapter([_call("touch", name="done.txt"), FINISH])
    res = run_task(task, adapter, "fake", 0, judge_adapter=judge)
    assert res.judge_score is None
    assert judge.calls == []           # judge was never invoked


def test_judge_skipped_when_no_judge_adapter():
    task = _make_task(judge_path=True)
    res = run_task(task, ProgrammedAdapter([FINISH]), "fake", 0, judge_adapter=None)
    assert res.judge_score is None


def test_judge_parse_error_scores_zero():
    task = _make_task(judge_path=True)
    bad = ProgrammedAdapter([ModelAction(kind="final", final_text="not json")])
    adapter = ProgrammedAdapter([_call("touch", name="done.txt"), FINISH])
    res = run_task(task, adapter, "fake", 0, judge_adapter=bad)
    assert res.judge_score == 0.0
    assert "parse error" in res.judge_rationale


def test_judge_strips_markdown_fences():
    payload = '```json\n{"score": 0.5, "rationale": "ok"}\n```'
    judge = ProgrammedAdapter([ModelAction(kind="final", final_text=payload)])
    from harness.core import Trajectory
    score, rationale = judge_path_quality(_make_task(), Trajectory(), judge)
    assert score == 0.5
    assert rationale == "ok"


# --------------------------------------------------------------------------- #
# run_study: the k x model x task matrix
# --------------------------------------------------------------------------- #
def test_run_study_produces_k_rows_per_pair():
    task = _make_task()
    models = {"m1": AlwaysAdapter(FINISH)}
    rows = run_study([task], models, k=3)
    assert len(rows) == 3
    assert {r.run_index for r in rows} == {0, 1, 2}
    assert all(r.model == "m1" for r in rows)


def test_run_study_records_crash_as_failure():
    rows = run_study([_make_task()], {"boom": CrashAdapter()}, k=2)
    assert len(rows) == 2
    assert all(r.success is False for r in rows)
    assert all(r.halt_reason.startswith("crash:") for r in rows)


def test_run_study_progress_callback():
    seen = []
    run_study([_make_task()], {"m": AlwaysAdapter(FINISH)}, k=2,
              progress=seen.append)
    assert len(seen) == 2
    assert "run 1/2" in seen[0]


def test_run_task_records_seed_and_differs_by_run_index(scripted_adapter_factory):
    from tasks.suite import T1B
    from harness.runner import run_task
    a0 = scripted_adapter_factory([("write_file", {"path": "answer.txt", "content": "1"}),
                                   ("__final__", {})])
    a1 = scripted_adapter_factory([("write_file", {"path": "answer.txt", "content": "1"}),
                                   ("__final__", {})])
    r0 = run_task(T1B, a0, "m", 0, base_seed=99)
    r1 = run_task(T1B, a1, "m", 1, base_seed=99)
    assert isinstance(r0.seed, int)
    assert r0.seed != r1.seed          # different run_index -> different instance seed
    # reproducible
    a0b = scripted_adapter_factory([("write_file", {"path": "answer.txt", "content": "1"}),
                                    ("__final__", {})])
    assert run_task(T1B, a0b, "m", 0, base_seed=99).seed == r0.seed


# --------------------------------------------------------------------------- #
# trajectory_sink
# --------------------------------------------------------------------------- #
def test_run_task_invokes_trajectory_sink(scripted_adapter_factory):
    """A trajectory_sink receives (model, task_id, run_index, trajectory)."""
    from tasks.suite import T1B
    captured = {}

    def sink(model, task_id, run_index, traj):
        captured["model"] = model
        captured["task_id"] = task_id
        captured["text"] = traj.as_text()

    adapter = scripted_adapter_factory([
        ("write_file", {"path": "answer.txt", "content": "1"}),
        ("__final__", {}),
    ])
    run_task(T1B, adapter, "fake:model", 0, trajectory_sink=sink)
    assert captured["model"] == "fake:model"
    assert captured["task_id"] == "t1b_count_logs"
    assert "halt:" in captured["text"]


def test_run_task_records_token_split_and_source():
    from harness.runner import run_task
    from harness.adapters import ModelAction
    from tasks.suite import T1B

    class _Adapter:
        def __init__(self):
            self.calls = 0
        def act(self, messages, specs, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ModelAction(kind="tool_call", tool="write_file",
                                   args={"path": "answer.txt", "content": "1"},
                                   raw="x", tokens=30, input_tokens=20,
                                   output_tokens=10, token_source="estimated")
            return ModelAction(kind="final", raw="done", tokens=15,
                               input_tokens=10, output_tokens=5,
                               token_source="estimated")

    r = run_task(T1B, _Adapter(), "m", 0)
    assert r.input_tokens == 30 and r.output_tokens == 15
    assert r.tokens_used == 45
    assert r.token_source == "estimated"
