"""Core abstractions for the agentic eval harness.

Design notes
------------
A *task* is a goal + a sandboxed environment + a set of tools the agent may
call + a deterministic outcome checker. The harness runs an agent loop and
records a *trajectory* (the full sequence of tool calls and observations), then
scores both the final outcome (auto) and, optionally, the path quality
(LLM-judge).

Methodology guardrails baked in here:
  * Per-tier compute budget (max_steps, max_tokens) is pinned on the Task, never
    left to the model. Capability is only comparable at a fixed budget.
  * Every run records steps, invalid-call rate, and tokens so you can report
    more than pass/fail.
  * The outcome checker is deterministic and runs against the *final env state*,
    not the model's self-report — models lie about having finished.

Single-responsibility modules, clean interfaces, info hiding (Ousterhout):
the agent loop never sees inside a tool; a tool never sees the grader; the
grader only sees the final environment snapshot + trajectory.
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    """A single action the agent can take.

    `fn` receives the live Environment plus keyword args the model supplied,
    and returns a JSON-serializable observation. Tools must be pure w.r.t. the
    environment they're given (no global state) so runs stay reproducible.
    """
    name: str
    description: str
    # JSON-schema-ish parameter spec, surfaced to the model.
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        """The advertised contract the model sees (no implementation leak)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class Environment:
    """A throwaway filesystem sandbox for one run of one task.

    Created from a task's `setup` callable, mutated by tool calls, snapshotted
    by the grader, then destroyed. Never reused across runs.
    """

    def __init__(self, root: Path):
        self.root = root
        self.scratch: dict[str, Any] = {}  # for tasks that report via a value
        self.seed: int = 0                  # set by make_environment for procedural tasks

    # -- filesystem helpers exposed to tools ------------------------------- #
    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        # contain everything inside the sandbox root
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path escapes sandbox: {rel}")
        return p

    def read(self, rel: str) -> str:
        return self.path(rel).read_text()

    def write(self, rel: str, content: str) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def listdir(self, rel: str = ".") -> list[str]:
        return sorted(p.name for p in self.path(rel).iterdir())

    def snapshot(self) -> dict[str, str]:
        """Flat {relpath: content} of all files, for the outcome checker."""
        out = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                try:
                    out[str(p.relative_to(self.root))] = p.read_text()
                except UnicodeDecodeError:
                    out[str(p.relative_to(self.root))] = "<binary>"
        return out

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    task_id: str
    tier: int                       # 1..4, graduated difficulty
    category: str
    goal: str                       # natural-language instruction to the agent
    tools: list[Tool]
    setup: Callable[[Environment], None]          # populate the sandbox
    check: Callable[[Environment, "Trajectory"], bool]  # deterministic grader
    # Pinned per-task compute budget. Defaults scale with tier; override freely.
    max_steps: int = 10
    max_tokens: int = 2048
    # If True, path quality is *also* judged by an LLM (outcome still auto).
    judge_path: bool = False
    notes: str = ""
    # Optional: build per-instance params from a seeded RNG (procedural generation).
    # Returns a dict stashed in env.scratch["params"] before setup runs; the grader
    # reads it for the ground-truth answer. None => static task (params == {}).
    parametrize: Optional[Callable[["random.Random"], dict]] = None

    def tool_specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools]

    def tool_by_name(self, name: str) -> Optional[Tool]:
        return next((t for t in self.tools if t.name == name), None)


# --------------------------------------------------------------------------- #
# Trajectory + Result
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    observation: Any
    valid: bool          # was this a well-formed call to a real tool?
    raw_model_output: str = ""


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    finished: bool = False           # did the model signal done itself?
    halt_reason: str = ""            # "done" | "max_steps" | "error"
    tokens_used: int = 0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def invalid_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(0 if s.valid else 1 for s in self.steps) / len(self.steps)

    def as_text(self) -> str:
        """Human/LLM-readable rendering of the path, for the judge."""
        lines = []
        for i, s in enumerate(self.steps, 1):
            tag = "" if s.valid else " [INVALID]"
            lines.append(f"{i}. call {s.tool}({json.dumps(s.args)}){tag}")
            lines.append(f"   -> {json.dumps(s.observation)[:300]}")
        lines.append(f"halt: {self.halt_reason}")
        return "\n".join(lines)


@dataclass
class RunResult:
    task_id: str
    tier: int
    category: str
    model: str
    run_index: int
    success: bool                    # deterministic outcome check
    n_steps: int
    invalid_rate: float
    tokens_used: int
    halt_reason: str
    wall_seconds: float
    judge_score: Optional[float] = None   # 0..1, only if judge_path
    judge_rationale: str = ""
    seed: int = 0                         # the seed this instance was generated from

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Sandbox factory
# --------------------------------------------------------------------------- #
def make_environment(task: Task, seed: int = 0) -> Environment:
    root = Path(tempfile.mkdtemp(prefix=f"agenteval_{task.task_id}_"))
    env = Environment(root)
    env.seed = seed
    rng = random.Random(seed)
    env.scratch["params"] = task.parametrize(rng) if task.parametrize else {}
    task.setup(env)
    return env
