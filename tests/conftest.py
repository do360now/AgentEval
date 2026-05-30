"""Shared fixtures and fake adapters for the offline test suite.

Nothing here touches a network, an API key, the `claude` CLI, or Ollama. The
agent loop talks to adapters only through `act(messages, tool_specs, max_tokens)
-> ModelAction`, so a scripted adapter is enough to exercise the whole runner.
"""
from __future__ import annotations

import pytest

from harness.adapters import ModelAction
from harness.core import Environment


@pytest.fixture
def make_env(tmp_path):
    """Build an Environment in a fresh subdir and run a task's setup over it.

    Returns a factory so a single test can spin up several independent sandboxes.
    """
    counter = {"n": 0}

    def _make(setup=None):
        counter["n"] += 1
        root = tmp_path / f"env{counter['n']}"
        root.mkdir()
        env = Environment(root)
        if setup is not None:
            setup(env)
        return env

    return _make


class ProgrammedAdapter:
    """Replays a fixed list of ModelActions, one per `act()` call.

    Once the program is exhausted it returns `default` (a `finish` by default),
    so a loop that runs longer than expected terminates instead of hanging.
    """

    def __init__(self, program, default=None):
        self.program = list(program)
        self.default = default or ModelAction(kind="final")
        self.i = 0
        self.calls = []  # records (messages, tool_specs, max_tokens) per turn

    def act(self, messages, tool_specs, max_tokens):
        self.calls.append((messages, tool_specs, max_tokens))
        if self.i < len(self.program):
            action = self.program[self.i]
            self.i += 1
            return action
        return self.default


class AlwaysAdapter:
    """Returns the same action on every call (handy across k repeats)."""

    def __init__(self, action):
        self.action = action
        self.n_calls = 0

    def act(self, messages, tool_specs, max_tokens):
        self.n_calls += 1
        return self.action


class CrashAdapter:
    """Raises on every call — used to prove run_study records crashes as data."""

    def act(self, messages, tool_specs, max_tokens):
        raise RuntimeError("boom")


@pytest.fixture
def scripted_adapter_factory():
    """Return a factory that builds a ProgrammedAdapter from a list of steps.

    Each step is a (tool_name, args_dict) tuple or ("__final__", {}) to signal
    completion. This mirrors ProgrammedAdapter but with a simpler step spec.
    """
    from harness.adapters import ModelAction

    def factory(steps):
        program = []
        for tool, args in steps:
            if tool == "__final__":
                program.append(ModelAction(kind="final"))
            else:
                program.append(ModelAction(kind="tool_call", tool=tool, args=args))
        return ProgrammedAdapter(program)

    return factory
