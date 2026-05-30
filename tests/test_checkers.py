"""Tests for every task's deterministic outcome checker.

These are the headline metric of the whole harness, so each checker is tested
for at least one passing and one failing final state. The t4a checker gets extra
attention: it must require *genuine error recovery*, not a lucky guess.
"""
from __future__ import annotations

import json

import pytest

from harness.core import Step, Trajectory
from tasks import suite


# --------------------------------------------------------------------------- #
# Tier 1
# --------------------------------------------------------------------------- #
def test_t1a_plain_value(make_env):
    env = make_env(suite._setup_t1a)
    env.write("answer.txt", "30")
    assert suite._check_t1a(env, Trajectory()) is True


def test_t1a_accepts_json_form(make_env):
    env = make_env(suite._setup_t1a)
    env.write("answer.txt", json.dumps({"timeout": 30}))
    assert suite._check_t1a(env, Trajectory()) is True


def test_t1a_wrong_value_fails(make_env):
    env = make_env(suite._setup_t1a)
    env.write("answer.txt", "31")
    assert suite._check_t1a(env, Trajectory()) is False


def test_t1a_missing_file_fails(make_env):
    env = make_env(suite._setup_t1a)
    assert suite._check_t1a(env, Trajectory()) is False


def test_t1b_counts_logs(make_env):
    env = make_env(suite._setup_t1b)
    env.write("answer.txt", "1")
    assert suite._check_t1b(env, Trajectory()) is True


def test_t1b_wrong_count_fails(make_env):
    env = make_env(suite._setup_t1b)
    env.write("answer.txt", "3")
    assert suite._check_t1b(env, Trajectory()) is False


# --------------------------------------------------------------------------- #
# Tier 2
# --------------------------------------------------------------------------- #
def test_t2a_mean(make_env):
    env = make_env(suite._setup_t2a)
    env.write("result.txt", "200")
    assert suite._check_t2a(env, Trajectory()) is True


def test_t2a_wrong_mean_fails(make_env):
    env = make_env(suite._setup_t2a)
    env.write("result.txt", "199")
    assert suite._check_t2a(env, Trajectory()) is False


def test_t2b_sorted(make_env):
    env = make_env(suite._setup_t2b)
    env.write("sorted.txt", "apple\nbanana\ncherry\n")
    assert suite._check_t2b(env, Trajectory()) is True


def test_t2b_unsorted_fails(make_env):
    env = make_env(suite._setup_t2b)
    env.write("sorted.txt", "banana\napple\ncherry\n")
    assert suite._check_t2b(env, Trajectory()) is False


def test_t2c_uppercase(make_env):
    env = make_env(suite._setup_t2c)
    env.write("upper.txt", "THE QUICK BROWN FOX")
    assert suite._check_t2c(env, Trajectory()) is True


def test_t2c_not_upper_fails(make_env):
    env = make_env(suite._setup_t2c)
    env.write("upper.txt", "the quick brown fox")
    assert suite._check_t2c(env, Trajectory()) is False


# --------------------------------------------------------------------------- #
# Tier 3
# --------------------------------------------------------------------------- #
def test_t3a_exact_dict(make_env):
    env = make_env(suite._setup_t3a)
    env.write("errors.json", json.dumps({"a.log": 2, "b.log": 0, "c.log": 1}))
    assert suite._check_t3a(env, Trajectory()) is True


def test_t3a_with_logs_prefix_fails(make_env):
    env = make_env(suite._setup_t3a)
    env.write("errors.json",
              json.dumps({"logs/a.log": 2, "logs/b.log": 0, "logs/c.log": 1}))
    assert suite._check_t3a(env, Trajectory()) is False


def test_t3a_including_non_log_fails(make_env):
    env = make_env(suite._setup_t3a)
    env.write("errors.json", json.dumps(
        {"a.log": 2, "b.log": 0, "c.log": 1, "readme.txt": 0}))
    assert suite._check_t3a(env, Trajectory()) is False


def test_t3b_sum_of_squares(make_env):
    env = make_env(suite._setup_t3b)
    env.write("total.txt", "55")
    assert suite._check_t3b(env, Trajectory()) is True


def test_t3b_wrong_total_fails(make_env):
    env = make_env(suite._setup_t3b)
    env.write("total.txt", "54")
    assert suite._check_t3b(env, Trajectory()) is False


# --------------------------------------------------------------------------- #
# Tier 4 — error recovery (the checker we fixed)
# --------------------------------------------------------------------------- #
def _error_step():
    """A well-formed (valid) call whose observation reports an error."""
    return Step(tool="read_file", args={"path": "primary.json"},
                observation="ERROR: FileNotFoundError: primary.json", valid=True)


def test_t4a_recovery_then_correct_passes(make_env):
    env = make_env(suite._setup_t4a)
    env.write("port.txt", "8080")
    traj = Trajectory(steps=[_error_step()])
    assert suite._check_t4a(env, traj) is True


def test_t4a_correct_value_without_error_fails(make_env):
    """A lucky guess (right answer, never hit the missing file) must NOT pass.

    This is the regression guard for the original bug where `recovered` was
    computed but never used.
    """
    env = make_env(suite._setup_t4a)
    env.write("port.txt", "8080")
    traj = Trajectory(steps=[
        Step(tool="read_file", args={"path": "backup.json"},
             observation='{"port": 8080}', valid=True)])
    assert suite._check_t4a(env, traj) is False


def test_t4a_error_seen_but_no_output_fails(make_env):
    env = make_env(suite._setup_t4a)
    traj = Trajectory(steps=[_error_step()])
    assert suite._check_t4a(env, traj) is False


def test_t4a_invalid_call_does_not_count_as_recovery(make_env):
    """An ERROR observation on an *invalid* call (e.g. 'no such tool') is not a
    real environment error and must not satisfy the recovery requirement."""
    env = make_env(suite._setup_t4a)
    env.write("port.txt", "8080")
    traj = Trajectory(steps=[
        Step(tool="bogus", args={}, observation="ERROR: no such tool",
             valid=False)])
    assert suite._check_t4a(env, traj) is False


def test_t4b_skips_bad_rows(make_env):
    env = make_env(suite._setup_t4b)
    env.write("revenue.txt", "25")
    assert suite._check_t4b(env, Trajectory()) is True


def test_t4b_counting_bad_row_fails(make_env):
    env = make_env(suite._setup_t4b)
    # If the malformed row had been (wrongly) included it wouldn't equal 25.
    env.write("revenue.txt", "26")
    assert suite._check_t4b(env, Trajectory()) is False


# --------------------------------------------------------------------------- #
# Suite-level invariants
# --------------------------------------------------------------------------- #
def test_task_ids_unique():
    ids = [t.task_id for t in suite.TASKS]
    assert len(ids) == len(set(ids))


def test_every_task_has_pinned_budget():
    for t in suite.TASKS:
        assert t.max_steps > 0 and t.max_tokens > 0


def test_tiers_are_in_range():
    assert all(1 <= t.tier <= 4 for t in suite.TASKS)


# --------------------------------------------------------------------------- #
# run_python tool
# --------------------------------------------------------------------------- #
import tempfile
from pathlib import Path
from harness.core import Environment
from tasks.suite import _run_python


def _env():
    return Environment(Path(tempfile.mkdtemp(prefix="agenteval_test_")))


def test_run_python_captures_stdout():
    env = _env()
    env.write("hi.py", "print('hello')")
    obs = _run_python(env, "hi.py")
    assert obs["returncode"] == 0
    assert obs["stdout"].strip() == "hello"


def test_run_python_reports_nonzero_exit_and_stderr():
    env = _env()
    env.write("boom.py", "raise ValueError('nope')")
    obs = _run_python(env, "boom.py")
    assert obs["returncode"] != 0
    assert "ValueError" in obs["stderr"]


def test_run_python_missing_file():
    env = _env()
    obs = _run_python(env, "nope.py")
    assert "error" in obs


def test_run_python_can_write_sandbox_files():
    env = _env()
    env.write("w.py", "open('out.txt','w').write('42')")
    _run_python(env, "w.py")
    assert env.read("out.txt") == "42"


# --------------------------------------------------------------------------- #
# Coding tasks
# --------------------------------------------------------------------------- #
from tasks.suite import (_check_c_impl, _check_c_fix, _setup_c_fix,
                         _check_c_transform, _setup_c_transform)


def test_c_impl_passes_correct_solution():
    env = _env()
    env.write("solution.py", "def solve(nums):\n    return sum(n for n in nums if n % 2 == 0)\n")
    assert _check_c_impl(env, None) is True


def test_c_impl_fails_wrong_solution():
    env = _env()
    env.write("solution.py", "def solve(nums):\n    return 6\n")  # hardcoded
    assert _check_c_impl(env, None) is False


def test_c_fix_passes_when_factorial_correct():
    env = _env()
    _setup_c_fix(env)
    env.write("buggy.py", "def fact(n):\n    r = 1\n    for i in range(1, n+1):\n        r *= i\n    return r\nprint(fact(5))\n")
    assert _check_c_fix(env, None) is True


def test_c_fix_fails_on_original_bug():
    env = _env()
    _setup_c_fix(env)
    assert _check_c_fix(env, None) is False


def test_c_transform_checks_output_file():
    env = _env()
    _setup_c_transform(env)
    env.write("sum.txt", "5050")
    assert _check_c_transform(env, None) is True
    env.write("sum.txt", "1")
    assert _check_c_transform(env, None) is False
