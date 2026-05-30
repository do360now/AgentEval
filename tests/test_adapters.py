"""Tests for model adapters and the shared protocol parsing.

The point of the adapter layer is that every provider — local ReAct text, the
`claude -p` CLI, and the cloud tool-calling APIs — normalizes to one ModelAction.
Network calls (`requests.post`) and the CLI (`subprocess.run`) are monkeypatched,
so nothing here needs an API key, the `claude` binary, or Ollama.
"""
from __future__ import annotations

import json
import subprocess
import types

import pytest

from harness import adapters
from harness.adapters import (AnthropicAdapter, ClaudeCliAdapter, OpenAIAdapter,
                              _parse_native, parse_react, system_preamble)
from tasks.suite import READ


SPECS = [READ.spec()]


# --------------------------------------------------------------------------- #
# system_preamble
# --------------------------------------------------------------------------- #
def test_preamble_react_lists_format_and_tools():
    p = system_preamble(SPECS, react=True)
    assert "Thought:" in p and "Action:" in p and "Final: done" in p
    assert "read_file" in p


def test_preamble_native_mentions_finish_tool():
    p = system_preamble(SPECS, react=False)
    assert "finish" in p
    assert "read_file" in p
    assert "Thought:" not in p


# --------------------------------------------------------------------------- #
# ReAct text parsing
# --------------------------------------------------------------------------- #
def test_parse_react_tool_call():
    text = 'Thought: read it\nAction: read_file\nArgs: {"path": "a.txt"}'
    a = parse_react(text)
    assert a.kind == "tool_call"
    assert a.tool == "read_file"
    assert a.args == {"path": "a.txt"}


def test_parse_react_final():
    a = parse_react("Thought: all set\nFinal: done")
    assert a.kind == "final"


def test_parse_react_unparseable_has_no_tool():
    a = parse_react("I'm not sure what to do here.")
    assert a.kind == "tool_call"
    assert a.tool is None          # loop will mark this step invalid


def test_parse_react_bad_args_json_defaults_empty():
    a = parse_react("Action: read_file\nArgs: {not valid json}")
    assert a.tool == "read_file"
    assert a.args == {}


# --------------------------------------------------------------------------- #
# Native JSON parsing (Ollama "native" fallback)
# --------------------------------------------------------------------------- #
def test_parse_native_tool_call():
    a = _parse_native(json.dumps({"name": "read_file",
                                  "arguments": {"path": "x"}}))
    assert a.tool == "read_file"
    assert a.args == {"path": "x"}


def test_parse_native_finish():
    a = _parse_native(json.dumps({"tool": "finish"}))
    assert a.kind == "final"


def test_parse_native_falls_back_to_react():
    a = _parse_native("Action: read_file\nArgs: {}")
    assert a.tool == "read_file"


# --------------------------------------------------------------------------- #
# Helpers for mocking the network
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_post(monkeypatch, payload, capture=None):
    def fake_post(url, headers=None, json=None, timeout=None):
        if capture is not None:
            capture.update(url=url, headers=headers, body=json)
        return FakeResponse(payload)
    monkeypatch.setattr(adapters.requests, "post", fake_post)


# --------------------------------------------------------------------------- #
# Anthropic adapter
# --------------------------------------------------------------------------- #
def test_anthropic_tool_use(monkeypatch):
    payload = {
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [{"type": "tool_use", "name": "read_file",
                     "input": {"path": "a.txt"}}],
    }
    captured = {}
    _patch_post(monkeypatch, payload, captured)
    a = AnthropicAdapter("claude-x", "key").act([], SPECS, 256)

    assert a.kind == "tool_call"
    assert a.tool == "read_file"
    assert a.args == {"path": "a.txt"}
    assert a.tokens == 15
    # A synthetic `finish` tool is advertised to the model.
    assert any(t["name"] == "finish" for t in captured["body"]["tools"])


def test_anthropic_finish_block_is_final(monkeypatch):
    payload = {"usage": {"input_tokens": 3, "output_tokens": 1},
               "content": [{"type": "tool_use", "name": "finish", "input": {}}]}
    _patch_post(monkeypatch, payload)
    a = AnthropicAdapter("claude-x", "key").act([], SPECS, 256)
    assert a.kind == "final"
    assert a.tokens == 4


def test_anthropic_text_only_is_final(monkeypatch):
    payload = {"usage": {}, "content": [{"type": "text", "text": "all done"}]}
    _patch_post(monkeypatch, payload)
    a = AnthropicAdapter("claude-x", "key").act([], SPECS, 256)
    assert a.kind == "final"
    assert "all done" in a.final_text


# --------------------------------------------------------------------------- #
# OpenAI adapter
# --------------------------------------------------------------------------- #
def test_openai_tool_call(monkeypatch):
    payload = {
        "usage": {"total_tokens": 42},
        "choices": [{"message": {"tool_calls": [
            {"function": {"name": "read_file",
                          "arguments": '{"path": "a.txt"}'}}]}}],
    }
    _patch_post(monkeypatch, payload)
    a = OpenAIAdapter("gpt-x", "key").act([], SPECS, 256)
    assert a.kind == "tool_call"
    assert a.tool == "read_file"
    assert a.args == {"path": "a.txt"}
    assert a.tokens == 42


def test_openai_finish_is_final(monkeypatch):
    payload = {"usage": {"total_tokens": 5},
               "choices": [{"message": {"tool_calls": [
                   {"function": {"name": "finish", "arguments": "{}"}}]}}]}
    _patch_post(monkeypatch, payload)
    a = OpenAIAdapter("gpt-x", "key").act([], SPECS, 256)
    assert a.kind == "final"


def test_openai_plain_message_is_final(monkeypatch):
    payload = {"usage": {"total_tokens": 7},
               "choices": [{"message": {"content": "no tools here"}}]}
    _patch_post(monkeypatch, payload)
    a = OpenAIAdapter("gpt-x", "key").act([], SPECS, 256)
    assert a.kind == "final"
    assert a.final_text == "no tools here"


# --------------------------------------------------------------------------- #
# Claude CLI adapter — the path used in THIS environment (no API key)
# --------------------------------------------------------------------------- #
def _patch_cli(monkeypatch, result_text, capture=None):
    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        if capture is not None:
            capture.update(cmd=cmd, input=input)
        return types.SimpleNamespace(
            stdout=json.dumps({"result": result_text}), stderr="", returncode=0)
    monkeypatch.setattr(adapters.subprocess, "run", fake_run)


def test_claude_cli_parses_react_tool_call(monkeypatch):
    captured = {}
    _patch_cli(monkeypatch,
               'Thought: read\nAction: read_file\nArgs: {"path": "a.txt"}',
               captured)
    a = ClaudeCliAdapter("claude-opus-4-7").act(
        [{"role": "user", "content": "TASK: go"}], SPECS, 256)

    assert a.kind == "tool_call"
    assert a.tool == "read_file"
    assert a.args == {"path": "a.txt"}
    # Tokens are ESTIMATED (chars//4), not read from the CLI, and must be > 0.
    assert a.tokens > 0
    # Tools are disabled CLI-side so Claude can't touch the real filesystem.
    assert "--allowed-tools" in captured["cmd"]
    assert "" in captured["cmd"]
    assert "--model" in captured["cmd"] and "claude-opus-4-7" in captured["cmd"]


def test_claude_cli_final(monkeypatch):
    _patch_cli(monkeypatch, "Thought: done\nFinal: done")
    a = ClaudeCliAdapter("claude-opus-4-7").act(
        [{"role": "user", "content": "x"}], SPECS, 256)
    assert a.kind == "final"


def test_claude_cli_timeout_is_invalid_step(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)
    monkeypatch.setattr(adapters.subprocess, "run", fake_run)

    a = ClaudeCliAdapter("claude-opus-4-7", timeout=1).act(
        [{"role": "user", "content": "x"}], SPECS, 256)
    assert a.tool is None           # loop marks it invalid
    assert "timeout" in a.raw.lower()


def test_claude_cli_non_json_stdout_falls_back(monkeypatch):
    # If stdout isn't JSON, the adapter should treat it as raw ReAct text.
    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(
            stdout="Action: read_file\nArgs: {}", stderr="", returncode=0)
    monkeypatch.setattr(adapters.subprocess, "run", fake_run)
    a = ClaudeCliAdapter("claude-opus-4-7").act(
        [{"role": "user", "content": "x"}], SPECS, 256)
    assert a.tool == "read_file"


# --------------------------------------------------------------------------- #
# Balanced-brace Args parser
# --------------------------------------------------------------------------- #
def test_parse_react_multiline_braces_in_args():
    text = (
        "Thought: write a script\n"
        "Action: write_file\n"
        'Args: {"path": "solution.py", "content": "def f():\\n    return {1: 2}\\n"}\n'
    )
    action = parse_react(text)
    assert action.kind == "tool_call"
    assert action.tool == "write_file"
    assert action.args["path"] == "solution.py"
    assert "return {1: 2}" in action.args["content"]


def test_parse_react_args_object_spanning_newlines():
    text = (
        "Action: write_file\n"
        'Args: {\n  "path": "a.txt",\n  "content": "hello"\n}\n'
    )
    action = parse_react(text)
    assert action.args == {"path": "a.txt", "content": "hello"}


def test_parse_react_brace_inside_string_not_counted():
    text = 'Action: write_file\nArgs: {"content": "a } b { c"}\n'
    action = parse_react(text)
    assert action.args == {"content": "a } b { c"}


def test_parse_react_malformed_args_yields_empty_dict():
    text = "Action: list_dir\nArgs: {not json}\n"
    action = parse_react(text)
    assert action.tool == "list_dir"
    assert action.args == {}
