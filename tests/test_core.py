"""Tests for the core data model: Environment sandbox + Trajectory metrics."""
from __future__ import annotations

import pytest

from harness.core import Environment, Step, Trajectory, make_environment
from tasks.suite import T1A


# --------------------------------------------------------------------------- #
# Environment — the throwaway filesystem sandbox
# --------------------------------------------------------------------------- #
def test_write_read_roundtrip(make_env):
    env = make_env()
    env.write("a/b.txt", "hello")
    assert env.read("a/b.txt") == "hello"


def test_write_creates_parent_dirs(make_env):
    env = make_env()
    env.write("deep/nested/path.txt", "x")
    assert env.path("deep/nested/path.txt").exists()


def test_listdir_is_sorted(make_env):
    env = make_env()
    for name in ["c.txt", "a.txt", "b.txt"]:
        env.write(name, "x")
    assert env.listdir() == ["a.txt", "b.txt", "c.txt"]


def test_path_rejects_sandbox_escape(make_env):
    env = make_env()
    with pytest.raises(ValueError, match="escapes sandbox"):
        env.path("../outside.txt")


def test_path_allows_root_itself(make_env):
    env = make_env()
    # "." resolves to the root and must not be treated as an escape.
    assert env.path(".") == env.root


def test_read_missing_file_raises(make_env):
    env = make_env()
    with pytest.raises(FileNotFoundError):
        env.read("nope.txt")


def test_snapshot_is_flat_relpath_map(make_env):
    env = make_env()
    env.write("top.txt", "1")
    env.write("sub/inner.txt", "2")
    snap = env.snapshot()
    assert snap == {"top.txt": "1", "sub/inner.txt": "2"}


def test_snapshot_marks_binary(make_env):
    env = make_env()
    (env.root / "blob.bin").write_bytes(b"\xff\xfe\x00")
    assert env.snapshot()["blob.bin"] == "<binary>"


def test_destroy_removes_root(make_env):
    env = make_env()
    env.write("f.txt", "x")
    env.destroy()
    assert not env.root.exists()


def test_make_environment_runs_task_setup():
    env = make_environment(T1A)
    try:
        # T1A's setup writes config.json with a 'timeout' key.
        assert "config.json" in env.listdir()
        assert "timeout" in env.read("config.json")
    finally:
        env.destroy()


# --------------------------------------------------------------------------- #
# Trajectory — derived metrics the report depends on
# --------------------------------------------------------------------------- #
def _step(tool="read_file", valid=True, obs="ok"):
    return Step(tool=tool, args={}, observation=obs, valid=valid)


def test_n_steps_counts_steps():
    traj = Trajectory(steps=[_step(), _step()])
    assert traj.n_steps == 2


def test_invalid_rate_empty_is_zero():
    assert Trajectory().invalid_rate == 0.0


def test_invalid_rate_mixed():
    traj = Trajectory(steps=[_step(valid=True), _step(valid=False),
                             _step(valid=False), _step(valid=True)])
    assert traj.invalid_rate == 0.5


def test_as_text_flags_invalid_and_halt():
    traj = Trajectory(steps=[_step(tool="ghost", valid=False, obs="ERROR")],
                      halt_reason="max_steps")
    text = traj.as_text()
    assert "[INVALID]" in text
    assert "ghost" in text
    assert "halt: max_steps" in text
