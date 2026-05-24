"""Model adapters.

Two protocols, one interface. The agent loop calls `adapter.act(messages,
tool_specs)` and gets back a normalized `ModelAction` regardless of whether the
underlying model speaks structured tool-calls or plain ReAct text. This is the
information-hiding boundary: the loop never knows or cares which protocol a
given model uses, so local 4B models and frontier cloud models run through the
exact same scoring path.

Token accounting: each adapter reports tokens it consumed so the harness can
enforce the task's pinned budget. For local Ollama we read eval_count /
prompt_eval_count from the response; for cloud we read usage.

Providers wired:
  * OllamaAdapter      -> http://localhost:11434  (local, GTX 1070, ~4B models)
  * AnthropicAdapter   -> Claude API
  * OpenAIAdapter      -> GPT API
Add others by implementing .act() with the same return contract.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class ModelAction:
    """Normalized output of one model turn."""
    kind: str                       # "tool_call" | "final"
    tool: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    final_text: str = ""
    raw: str = ""
    tokens: int = 0


# --------------------------------------------------------------------------- #
# Prompt construction (shared)
# --------------------------------------------------------------------------- #
def system_preamble(tool_specs: list[dict[str, Any]], react: bool) -> str:
    tools_desc = "\n".join(
        f"- {t['name']}({', '.join(t['parameters'].get('properties', {}).keys())}): "
        f"{t['description']}"
        for t in tool_specs
    )
    if react:
        return (
            "You are an agent that completes a task by calling tools one at a "
            "time. Available tools:\n" + tools_desc + "\n\n"
            "Respond in EXACTLY this format each turn:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Args: <single-line JSON object of arguments>\n\n"
            "When the task is fully complete, respond instead with:\n"
            "Thought: <why you are done>\n"
            "Final: done\n"
        )
    return (
        "You are an agent that completes a task by calling tools. "
        "Call one tool per turn. When the task is fully complete, call the "
        "special tool `finish` with no arguments. Available tools:\n" + tools_desc
    )


# --------------------------------------------------------------------------- #
# ReAct text parsing (fallback for models without structured tool-calls)
# --------------------------------------------------------------------------- #
_ACTION_RE = re.compile(r"Action:\s*(\w+)", re.I)
_ARGS_RE = re.compile(r"Args:\s*(\{.*?\})", re.I | re.S)
_FINAL_RE = re.compile(r"Final:\s*done", re.I)


def parse_react(text: str) -> ModelAction:
    if _FINAL_RE.search(text):
        return ModelAction(kind="final", final_text=text, raw=text)
    m_action = _ACTION_RE.search(text)
    if not m_action:
        # Unparseable -> caller will mark the step invalid.
        return ModelAction(kind="tool_call", tool=None, args=None, raw=text)
    tool = m_action.group(1)
    args: dict[str, Any] = {}
    m_args = _ARGS_RE.search(text)
    if m_args:
        try:
            args = json.loads(m_args.group(1))
        except json.JSONDecodeError:
            args = {}
    return ModelAction(kind="tool_call", tool=tool, args=args, raw=text)


# --------------------------------------------------------------------------- #
# Ollama (local)
# --------------------------------------------------------------------------- #
class OllamaAdapter:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 use_native_tools: bool = False):
        self.model = model
        self.host = host.rstrip("/")
        # Many ~4B models are unreliable at native tool-calls; default to ReAct.
        self.react = not use_native_tools

    def act(self, messages: list[dict[str, str]],
            tool_specs: list[dict[str, Any]], max_tokens: int) -> ModelAction:
        sys = system_preamble(tool_specs, react=self.react)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys}] + messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        action = parse_react(text) if self.react else _parse_native(text)
        action.tokens = tokens
        return action


def _parse_native(text: str) -> ModelAction:
    """Best-effort parse when a local model claims native tool support but
    actually emits a JSON blob in content."""
    try:
        obj = json.loads(text)
        if obj.get("tool") == "finish" or obj.get("name") == "finish":
            return ModelAction(kind="final", final_text=text, raw=text)
        name = obj.get("tool") or obj.get("name")
        args = obj.get("args") or obj.get("arguments") or {}
        return ModelAction(kind="tool_call", tool=name, args=args, raw=text)
    except json.JSONDecodeError:
        return parse_react(text)


# --------------------------------------------------------------------------- #
# Anthropic (cloud) — uses native tool-calling
# --------------------------------------------------------------------------- #
class AnthropicAdapter:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.key = api_key

    def _tools(self, tool_specs):
        out = [{"name": t["name"], "description": t["description"],
                "input_schema": t["parameters"]} for t in tool_specs]
        out.append({"name": "finish", "description": "Call when task is done.",
                    "input_schema": {"type": "object", "properties": {}}})
        return out

    def act(self, messages, tool_specs, max_tokens) -> ModelAction:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.model, "max_tokens": max_tokens,
                  "tools": self._tools(tool_specs), "messages": messages},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                if block["name"] == "finish":
                    return ModelAction(kind="final", raw=json.dumps(data),
                                       tokens=tokens)
                return ModelAction(kind="tool_call", tool=block["name"],
                                   args=block.get("input", {}),
                                   raw=json.dumps(data), tokens=tokens)
        # No tool call -> treat text as final.
        txt = " ".join(b.get("text", "") for b in data.get("content", []))
        return ModelAction(kind="final", final_text=txt, raw=txt, tokens=tokens)


# --------------------------------------------------------------------------- #
# OpenAI (cloud) — chat completions w/ tools
# --------------------------------------------------------------------------- #
class OpenAIAdapter:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.key = api_key

    def _tools(self, tool_specs):
        out = [{"type": "function",
                "function": {"name": t["name"], "description": t["description"],
                             "parameters": t["parameters"]}}
               for t in tool_specs]
        out.append({"type": "function",
                    "function": {"name": "finish",
                                 "description": "Call when task is done.",
                                 "parameters": {"type": "object",
                                                "properties": {}}}})
        return out

    def act(self, messages, tool_specs, max_tokens) -> ModelAction:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "max_tokens": max_tokens,
                  "tools": self._tools(tool_specs), "messages": messages},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls:
            call = calls[0]["function"]
            if call["name"] == "finish":
                return ModelAction(kind="final", raw=json.dumps(msg),
                                   tokens=tokens)
            try:
                args = json.loads(call.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            return ModelAction(kind="tool_call", tool=call["name"], args=args,
                               raw=json.dumps(msg), tokens=tokens)
        return ModelAction(kind="final", final_text=msg.get("content", ""),
                           raw=json.dumps(msg), tokens=tokens)


# --------------------------------------------------------------------------- #
# Claude via the local `claude -p` CLI (subscription auth, no API key)
# --------------------------------------------------------------------------- #
class ClaudeCliAdapter:
    """Drive a Claude model through the local `claude -p` CLI.

    Why this exists: the harness's AnthropicAdapter calls api.anthropic.com with a
    raw ANTHROPIC_API_KEY. When only the Claude Code CLI is authenticated (no API
    key in env), this adapter scores the same model through `claude -p` instead.

    Two deliberate choices, both flagged because they make this a *proxy* for the
    raw-API path, not an identical one:

    1. Tools are disabled on the CLI side (`--allowed-tools ""`). Claude Code acts
       as a pure text model and parses through the ReAct path, exactly like a local
       Ollama model — so it can never touch the real filesystem. The harness's own
       sandbox tools do all the work.
    2. Token counts are ESTIMATED as (prompt+reply chars)//4, NOT read from the
       CLI's `usage`. Claude Code injects its own multi-thousand-token base system
       prompt; counting that would blow the harness's per-task budget guard on
       step 1 and fail every task. The estimate reflects the harness's own token
       economy (preamble + conversation), which is the fair, comparable measure.
       Treat tokens from this adapter as approximate.

    The CLI is stateless per call, so each `act()` resends the full conversation —
    which matches how the API adapters re-send history each turn.
    """

    def __init__(self, model: str, claude_bin: str = "claude",
                 timeout: int = 300):
        self.model = model
        self.bin = claude_bin
        self.timeout = timeout
        self.react = True  # always the ReAct text protocol over the CLI

    def act(self, messages: list[dict[str, str]],
            tool_specs: list[dict[str, Any]], max_tokens: int) -> ModelAction:
        sys = system_preamble(tool_specs, react=True)
        convo = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        prompt = (f"{sys}\n\n=== CONVERSATION SO FAR ===\n{convo}\n\n"
                  "Respond with the NEXT single step only, in the required "
                  "format. Do not use any tools; emit only the text format.")
        try:
            proc = subprocess.run(
                [self.bin, "-p", "--model", self.model,
                 "--output-format", "json",
                 "--permission-mode", "default",
                 "--allowed-tools", "",
                 "--no-session-persistence"],
                input=prompt, capture_output=True, text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return ModelAction(kind="tool_call", tool=None, args=None,
                               raw="ERROR: claude CLI timeout", tokens=0)

        text = ""
        out = (proc.stdout or "").strip()
        if out:
            try:
                obj = json.loads(out)
                text = obj.get("result", "") or ""
            except json.JSONDecodeError:
                text = out  # fall back to raw text if not JSON
        action = parse_react(text)
        action.tokens = (len(prompt) + len(text)) // 4  # see class docstring
        return action
