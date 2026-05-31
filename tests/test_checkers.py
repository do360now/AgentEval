"""Tests for every task's deterministic outcome checker.

These are the headline metric of the whole harness, so each checker is tested
for at least one passing and one failing final state. The t4a checker gets extra
attention: it must require *genuine error recovery*, not a lucky guess.
"""
from __future__ import annotations

import json
import random

import pytest

from harness.core import Step, Trajectory, make_environment
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
from tasks.suite import T3A, _gen_t3a
import json as _json


def test_t3a_answer_matches_error_lines():
    for s in range(200):
        p = _gen_t3a(random.Random(s))
        for name, count in p["answer"].items():
            content = p["files"]["logs/" + name]
            assert sum(1 for ln in content.splitlines() if "ERROR" in ln) == count
        assert any(not k.endswith(".log") for k in p["files"])  # has a decoy


def test_t3a_check_uses_params():
    env = make_environment(T3A, seed=8)
    env.write("errors.json", _json.dumps(env.scratch["params"]["answer"]))
    assert T3A.check(env, None) is True
    env.write("errors.json", _json.dumps({"zzz.log": 99}))
    assert T3A.check(env, None) is False
    env.destroy()


from tasks.suite import T3B, _gen_t3b


def test_t3b_answer_is_sum():
    for s in range(200):
        p = _gen_t3b(random.Random(s))
        total = sum(int(v.strip()) for v in p["files"].values())
        assert p["answer"] == str(total)


def test_t3b_check_uses_params():
    env = make_environment(T3B, seed=2)
    env.write("total.txt", env.scratch["params"]["answer"])
    assert T3B.check(env, None) is True
    env.write("total.txt", "0")
    assert T3B.check(env, None) is False
    env.destroy()


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
    # 1-4 = graduated difficulty; 5 = hard discriminators (separate frontier models).
    assert all(1 <= t.tier <= 5 for t in suite.TASKS)


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
from tasks.suite import (_check_c_fix, _setup_c_fix,
                         _check_c_transform, _setup_c_transform)
from tasks.suite import C_IMPL, _gen_c_impl


def test_c_impl_cases_match_reference():
    for s in range(200):
        p = _gen_c_impl(random.Random(s))
        for inp, exp in p["cases"]:
            assert exp == sum(x for x in inp if x % 2 == 0)


def test_c_impl_check_runs_against_seed_cases():
    env = make_environment(C_IMPL, seed=4)
    env.write("solution.py",
              "def solve(nums):\n    return sum(n for n in nums if n % 2 == 0)\n")
    assert C_IMPL.check(env, None) is True
    env.write("solution.py", "def solve(nums):\n    return 6\n")  # hardcoded -> fails
    assert C_IMPL.check(env, None) is False
    env.destroy()


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


# --------------------------------------------------------------------------- #
# Reasoning tasks
# --------------------------------------------------------------------------- #
from tasks.suite import R_MATH, _gen_r_math


def test_r_math_generator_answer_is_correct():
    for s in range(200):
        p = _gen_r_math(random.Random(s))
        cost = (p["num_pens"] // p["group_size"]) * p["group_price"]
        assert p["answer"] == str(p["bill"] - cost)
        assert p["bill"] > cost          # change is non-negative
        assert "problem_text" in p


def test_r_math_check_uses_params():
    env = make_environment(R_MATH, seed=5)
    env.write("answer.txt", env.scratch["params"]["answer"])
    assert R_MATH.check(env, None) is True
    env.write("answer.txt", "-1")
    assert R_MATH.check(env, None) is False
    env.destroy()


from tasks.suite import R_LOGIC, _gen_r_logic, _logic_solutions


def test_r_logic_instances_are_unique_and_answer_correct():
    for s in range(200):
        p = _gen_r_logic(random.Random(s))
        sols = _logic_solutions(p["people"], p["pets"], p["clues"])
        assert len(sols) == 1, f"seed {s} not unique"
        owner = next(per for per in p["people"]
                     if sols[0][per] == p["queried_pet"])
        assert p["answer"] == owner


def test_r_logic_check_uses_params():
    env = make_environment(R_LOGIC, seed=3)
    env.write("answer.txt", env.scratch["params"]["answer"])
    assert R_LOGIC.check(env, None) is True
    env.write("answer.txt", "Nobody")
    assert R_LOGIC.check(env, None) is False
    env.destroy()


from tasks.suite import R_PLAN, _gen_r_plan


def test_r_plan_ground_truth_order_satisfies_constraints():
    for s in range(200):
        p = _gen_r_plan(random.Random(s))
        order = p["valid_order"]
        pos = {x: i for i, x in enumerate(order)}
        for c in p["constraints"]:
            if c[0] == "before":
                assert pos[c[1]] < pos[c[2]]
            else:
                assert pos[c[1]] == len(order) - 1


def test_r_plan_check_accepts_valid_rejects_invalid():
    env = make_environment(R_PLAN, seed=11)
    p = env.scratch["params"]
    env.write("plan.txt", " ".join(p["valid_order"]))
    assert R_PLAN.check(env, None) is True
    env.write("plan.txt", " ".join(reversed(p["valid_order"])))
    # reversed almost always violates a 'before'/'last' constraint
    assert R_PLAN.check(env, None) is False
    env.destroy()


# --------------------------------------------------------------------------- #
# Agentic t4c + suite integrity
# --------------------------------------------------------------------------- #
from tasks.suite import _setup_t4c, _check_t4c, TASKS


def test_t4c_skips_corrupt_file():
    env = _env()
    _setup_t4c(env)
    env.write("total.txt", "42")
    assert _check_t4c(env, None) is True
    env.write("total.txt", "84")
    assert _check_t4c(env, None) is False


def test_suite_has_unique_task_ids_and_new_categories():
    ids = [t.task_id for t in TASKS]
    assert len(ids) == len(set(ids)), "duplicate task_id"
    cats = {t.category for t in TASKS}
    assert {"coding", "reasoning", "agentic"} <= cats
    # every task has positive budgets
    assert all(t.max_steps > 0 and t.max_tokens > 0 for t in TASKS)


# --- Tier 5 hard discriminators ------------------------------------------- #
import random as _random
import csv as _csvmod
import io as _io
import json as _jsonmod
from harness.core import make_environment as _make_env
from tasks.suite import (H_MERGE, _gen_h_merge, _merge_ref,
                         H_JOIN, _gen_h_join, TASKS as _TASKS)


def test_h_merge_reference_cases_are_sorted_nonoverlapping():
    for s in range(200):
        p = _gen_h_merge(_random.Random(s))
        for inp, exp in p["cases"]:
            assert exp == _merge_ref(inp)
            for (a1, b1), (a2, b2) in zip(exp, exp[1:]):
                assert b1 < a2          # strictly separated => merged correctly


def test_h_merge_check_passes_reference_fails_identity():
    env = _make_env(H_MERGE, seed=3)
    env.write("solution.py",
              "def merge(intervals):\n"
              "    if not intervals: return []\n"
              "    s=sorted([list(x) for x in intervals])\n"
              "    out=[list(s[0])]\n"
              "    for a,b in s[1:]:\n"
              "        if a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n"
              "        else: out.append([a,b])\n"
              "    return out\n")
    assert H_MERGE.check(env, None) is True
    env.write("solution.py", "def merge(iv):\n    return iv\n")
    assert H_MERGE.check(env, None) is False
    env.destroy()


def test_h_join_answer_matches_independent_recompute():
    for s in range(200):
        p = _gen_h_join(_random.Random(s))
        cust = {r["customer_id"]: r["region"]
                for r in _csvmod.DictReader(_io.StringIO(p["customers_csv"]))}
        tot: dict = {}
        for r in _csvmod.DictReader(_io.StringIO(p["orders_csv"])):
            reg = cust[r["customer_id"]]
            tot[reg] = tot.get(reg, 0) + int(r["amount"])
        assert tot == p["answer"]


def test_h_join_check_uses_params():
    env = _make_env(H_JOIN, seed=7)
    env.write("region_totals.json", _jsonmod.dumps(env.scratch["params"]["answer"]))
    assert H_JOIN.check(env, None) is True
    env.write("region_totals.json", _jsonmod.dumps({"Nowhere": 1}))
    assert H_JOIN.check(env, None) is False
    env.destroy()


def test_hard_tasks_registered_as_tier5():
    hard = {t.task_id for t in _TASKS if t.tier == 5}
    assert hard == {"h_merge_intervals", "h_revenue_by_region"}


# --- run_python sandbox containment (bwrap) ------------------------------- #
import os as _os
import pytest as _pytest
from tasks.suite import _bwrap_bin as _bwrap, _run_python as _rp


@_pytest.mark.skipif(not _bwrap(), reason="bwrap not available; run_python falls back unsandboxed")
def test_run_python_cannot_escape_sandbox():
    env = _env()
    sentinel = _os.path.join(_os.path.expanduser("~"), "AGENTEVAL_PWN_TEST")
    env.write("esc.py", f"open({sentinel!r}, 'w').write('x')")
    r = _rp(env, "esc.py")
    assert r["sandboxed"] is True
    assert r["returncode"] != 0                 # the out-of-jail write raised
    assert not _os.path.exists(sentinel)        # nothing escaped to $HOME
