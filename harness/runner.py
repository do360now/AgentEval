"""The agent loop + scoring.

`run_task` executes one (task, model) pair once: it drives the agent loop under
the task's pinned budget, builds a Trajectory, runs the deterministic outcome
checker against the FINAL environment state, and — only if the task opts in —
runs an LLM judge over the trajectory text for path quality.

`run_study` orchestrates k repeats across the full model x task matrix and
returns a flat list of RunResult rows ready for aggregation.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Optional

from harness.core import (Environment, RunResult, Step, Task, Trajectory,
                   make_environment)


# --------------------------------------------------------------------------- #
# One run of the agent loop
# --------------------------------------------------------------------------- #
def run_task(task: Task, adapter, model_name: str, run_index: int,
             judge_adapter=None, trajectory_sink=None, base_seed: int = 0) -> RunResult:
    seed = int(hashlib.sha256(
        f"{base_seed}:{task.task_id}:{run_index}".encode()).hexdigest()[:8], 16)
    env = make_environment(task, seed)
    traj = Trajectory()
    t0 = time.time()

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"TASK: {task.goal}"}
    ]

    token_source = "measured"
    try:
        for _ in range(task.max_steps):
            action = adapter.act(messages, task.tool_specs(), task.max_tokens)
            traj.tokens_used += action.tokens
            traj.input_tokens_used += action.input_tokens
            traj.output_tokens_used += action.output_tokens
            token_source = action.token_source

            # Budget guard: stop if we've blown the pinned token ceiling.
            if traj.tokens_used > task.max_tokens * task.max_steps:
                traj.halt_reason = "token_budget"
                break

            if action.kind == "final":
                traj.finished = True
                traj.halt_reason = "done"
                break

            tool = task.tool_by_name(action.tool) if action.tool else None
            if tool is None:
                # Invalid: nonexistent or unparseable tool call.
                traj.steps.append(Step(tool=str(action.tool), args=action.args or {},
                                       observation="ERROR: no such tool",
                                       valid=False, raw_model_output=action.raw))
                messages.append({"role": "assistant", "content": action.raw})
                messages.append({"role": "user",
                                 "content": "Observation: ERROR: no such tool. "
                                            "Use one of the available tools."})
                continue

            # Execute the tool against the live environment.
            try:
                obs = tool.fn(env, **(action.args or {}))
                valid = True
            except Exception as e:  # tool-level failure is a real observation
                obs = f"ERROR: {type(e).__name__}: {e}"
                valid = True  # well-formed call, environment said no
            traj.steps.append(Step(tool=tool.name, args=action.args or {},
                                   observation=obs, valid=valid,
                                   raw_model_output=action.raw))
            messages.append({"role": "assistant", "content": action.raw})
            messages.append({"role": "user",
                             "content": f"Observation: {json.dumps(obs)[:1000]}"})
        else:
            traj.halt_reason = "max_steps"

        # ---- deterministic outcome check against FINAL state -------------- #
        success = bool(task.check(env, traj))

        # ---- optional LLM-judge over the path ----------------------------- #
        judge_score: Optional[float] = None
        judge_rationale = ""
        if task.judge_path and judge_adapter is not None:
            judge_score, judge_rationale = judge_path_quality(
                task, traj, judge_adapter)

        if trajectory_sink is not None:
            trajectory_sink(model_name, task.task_id, run_index, traj)

    finally:
        snap_seconds = time.time() - t0
        env.destroy()

    return RunResult(
        task_id=task.task_id, tier=task.tier, category=task.category,
        model=model_name, run_index=run_index, success=success,
        n_steps=traj.n_steps, invalid_rate=round(traj.invalid_rate, 3),
        tokens_used=traj.tokens_used, halt_reason=traj.halt_reason,
        wall_seconds=round(snap_seconds, 2),
        judge_score=judge_score, judge_rationale=judge_rationale,
        seed=seed,
        input_tokens=traj.input_tokens_used,
        output_tokens=traj.output_tokens_used,
        token_source=token_source,
    )


# --------------------------------------------------------------------------- #
# LLM-as-judge (path quality only; correctness stays deterministic)
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """You are scoring the EFFICIENCY and SOUNDNESS of an agent's \
approach to a task. The task's success/failure is already determined separately \
-- do NOT re-judge correctness. Score only HOW the agent worked.

TASK GOAL:
{goal}

AGENT TRAJECTORY:
{trajectory}

Rate the path on a 0.0-1.0 scale where:
  1.0 = direct, no wasted or redundant steps, clean error recovery
  0.5 = reached the goal but with avoidable detours or repeated calls
  0.0 = flailing, many invalid/redundant actions, no coherent plan

Respond ONLY with a JSON object, no prose, no markdown:
{{"score": <float>, "rationale": "<one sentence>"}}"""


def judge_path_quality(task: Task, traj: Trajectory,
                       judge_adapter) -> tuple[float, str]:
    prompt = JUDGE_PROMPT.format(goal=task.goal, trajectory=traj.as_text())
    # The judge is just a model turn with no tools; reuse .act with empty specs.
    action = judge_adapter.act([{"role": "user", "content": prompt}], [], 512)
    text = action.final_text or action.raw or ""
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
        return float(obj.get("score", 0.0)), str(obj.get("rationale", ""))[:200]
    except (json.JSONDecodeError, ValueError):
        return 0.0, "judge parse error"


# --------------------------------------------------------------------------- #
# Study orchestration: k repeats over the model x task matrix
# --------------------------------------------------------------------------- #
def run_study(tasks: list[Task], models: dict[str, Any], k: int = 5,
              judge_adapter=None,
              progress: Optional[Callable[[str], None]] = None,
              trajectory_sink=None, base_seed: int = 0) -> list[RunResult]:
    rows: list[RunResult] = []
    for model_name, adapter in models.items():
        for task in tasks:
            for i in range(k):
                if progress:
                    progress(f"{model_name} | {task.task_id} | run {i+1}/{k}")
                try:
                    rows.append(run_task(task, adapter, model_name, i,
                                         judge_adapter,
                                         trajectory_sink=trajectory_sink,
                                         base_seed=base_seed))
                except Exception as e:
                    # A crashed run is data too: record it as a failure.
                    rows.append(RunResult(
                        task_id=task.task_id, tier=task.tier,
                        category=task.category, model=model_name, run_index=i,
                        success=False, n_steps=0, invalid_rate=0.0,
                        tokens_used=0, halt_reason=f"crash:{type(e).__name__}",
                        wall_seconds=0.0))
    return rows
