# Capability-Eval Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `agenteval` to score models on agentic, coding, and reasoning capability (reported per category), driven by a `run_python` execution tool, robust multi-line ReAct parsing, and trajectory dumping.

**Architecture:** Preserve the strict module boundaries (loop ⊥ tool ⊥ grader) and the deterministic-`check()` invariant. Add a sandboxed `run_python` tool, new coding/reasoning/agentic tasks with deterministic graders, a balanced-brace `Args:` parser so multi-line code args survive the ReAct text protocol, a `by_category` aggregation, and an optional trajectory sink.

**Tech Stack:** Python 3.12 stdlib only (runtime dep is `requests`). Tests via `pytest`, fully offline. Use `python3` (no bare `python` on this box).

---

## Background the implementer needs

- Read `CLAUDE.md` and the spec `docs/superpowers/specs/2026-05-30-capability-eval-expansion-design.md` first.
- The suite lives in `tasks/suite.py`: each `Task` has a `setup(env)` that populates a temp-dir sandbox and a `check(env, traj) -> bool` deterministic grader. Tasks are collected in the `TASKS` list at the bottom.
- `Environment` (`harness/core.py`) exposes `env.read/write/listdir/path` and `env.root` (a `Path` to the temp dir). `env.path(rel)` resolves a sandbox-contained path.
- The agent loop (`harness/runner.py::run_task`) re-sends full history each turn; adapters are stateless.
- Run the full offline suite with `python3 -m pytest -q` (must stay green, ~0.3–1s).
- Commit after every task. Branch: work on the current branch (`master`).

---

## Task 1: Balanced-brace `Args:` parser (multi-line tool args)

**Why:** `harness/adapters.py` parses ReAct text with `_ARGS_RE = re.compile(r"Args:\s*(\{.*?\})", re.S)`. The non-greedy `.*?` stops at the FIRST `}`, so a multi-line JSON arg containing code with braces is truncated and fails to parse. Coding tasks over the CLI-proxy ReAct path need balanced-brace extraction.

**Files:**
- Modify: `harness/adapters.py` (replace `_ARGS_RE` usage in `parse_react`, add `_extract_args_object`)
- Test: `tests/test_adapters.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_adapters.py`:

```python
from harness.adapters import parse_react


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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_adapters.py -k "multiline_braces or spanning_newlines or brace_inside_string" -q`
Expected: FAIL (truncated/incorrect parse on the multi-line/brace cases).

- [ ] **Step 3: Implement balanced-brace extraction**

In `harness/adapters.py`, add this helper above `parse_react`:

```python
def _extract_args_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring after an ``Args:`` label.

    Brace-counts, ignoring braces that appear inside JSON string literals, so a
    multi-line code argument containing ``{`` / ``}`` survives. Returns None if no
    object is found.
    """
    m = re.search(r"Args:\s*", text, re.I)
    if not m:
        return None
    start = text.find("{", m.end())
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return None
```

Then replace the `_ARGS_RE` block inside `parse_react`. The function becomes:

```python
def parse_react(text: str) -> ModelAction:
    if _FINAL_RE.search(text):
        return ModelAction(kind="final", final_text=text, raw=text)
    m_action = _ACTION_RE.search(text)
    if not m_action:
        # Unparseable -> caller will mark the step invalid.
        return ModelAction(kind="tool_call", tool=None, args=None, raw=text)
    tool = m_action.group(1)
    args: dict[str, Any] = {}
    raw_args = _extract_args_object(text)
    if raw_args:
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    return ModelAction(kind="tool_call", tool=tool, args=args, raw=text)
```

Delete the now-unused `_ARGS_RE = re.compile(...)` line.

- [ ] **Step 4: Run all adapter tests**

Run: `python3 -m pytest tests/test_adapters.py -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add harness/adapters.py tests/test_adapters.py
git commit -m "feat(adapters): balanced-brace Args parser for multi-line tool args"
```

---

## Task 2: `run_python` sandboxed execution tool

**Why:** Coding tasks must grade on real execution. Add a tool that runs a Python file inside the sandbox temp dir.

**Files:**
- Modify: `tasks/suite.py` (add tool + `CODE_TOOLS` list, imports)
- Test: `tests/test_checkers.py` (a `run_python` section)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_checkers.py` (it already constructs Environments — match the existing helper; if there is a fixture/factory for a temp `Environment`, reuse it; otherwise use this self-contained helper):

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k run_python -q`
Expected: FAIL (`_run_python` not defined).

- [ ] **Step 3: Implement the tool**

In `tasks/suite.py`, add to the imports at top:

```python
import subprocess
import sys
```

Add after the existing shared-tool definitions (after `BASE_TOOLS = [...]`):

```python
def _run_python(env: Environment, path: str):
    """Execute a Python file inside the sandbox and return its output.

    SECURITY: runs untrusted, model-generated Python on the host. It is sandboxed
    only by (a) running inside the throwaway temp dir (cwd=env.root), (b) Python's
    isolated mode (-I), and (c) a 10s hard timeout. It is NOT containerized and the
    process can reach the network/host. Acceptable for trusted models on a dev box;
    do not point this at adversarial input. A timeout or crash is a real, recoverable
    observation (valid=True) -- this is how coding error-recovery is exercised.
    """
    target = env.path(path)
    if not target.exists():
        return {"error": f"no such file: {path}"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(target)],
            cwd=str(env.root), capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout after 10s", "returncode": None}
    return {"stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "returncode": proc.returncode}


RUN_PYTHON = Tool(
    "run_python",
    "Execute a Python file in the sandbox; returns its stdout, stderr, exit code.",
    {"type": "object", "properties": {"path": {"type": "string"}},
     "required": ["path"]}, _run_python)

CODE_TOOLS = BASE_TOOLS + [RUN_PYTHON]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_checkers.py -k run_python -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): sandboxed run_python execution tool"
```

---

## Task 3: Coding tasks (impl / fix-bug / transform)

**Files:**
- Modify: `tasks/suite.py` (add 3 tasks + their setup/check, append to `TASKS`)
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing checker tests**

Add to `tests/test_checkers.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k "c_impl or c_fix or c_transform" -q`
Expected: FAIL (names not defined).

- [ ] **Step 3: Implement the tasks**

In `tasks/suite.py`, add a "Coding" section before the final `TASKS = [...]` line:

```python
# --------------------------------------------------------------------------- #
# Coding — write/execute code, graded on real run output
# --------------------------------------------------------------------------- #
def _run_file(env, name):
    """Run a sandbox file with the test interpreter; return CompletedProcess|None."""
    try:
        return subprocess.run([sys.executable, "-I", str(env.path(name))],
                              cwd=str(env.root), capture_output=True,
                              text=True, timeout=10)
    except Exception:
        return None


def _setup_c_impl(env):
    pass  # spec is in the goal; the model authors solution.py

def _check_c_impl(env, traj):
    # Held-out cases (NOT shown in the goal) defeat hardcoding.
    grader = (
        "import solution\n"
        "cases = [([1,2,3,4],6), ([10,11,12],22), ([],0), ([7],0), ([2,4,6],12)]\n"
        "print('PASS' if all(solution.solve(i)==e for i,e in cases) else 'FAIL')\n"
    )
    try:
        env.write("_grader.py", grader)
    except Exception:
        return False
    proc = _run_file(env, "_grader.py")
    return bool(proc) and proc.stdout.strip().endswith("PASS")

C_IMPL = Task("c_impl_function", 3, "coding",
              "Write a function solve(nums) in solution.py that returns the sum of "
              "the EVEN integers in the list nums. Example: solve([1,2,3,4]) returns "
              "6. Do not call the function yourself; just define it.",
              CODE_TOOLS, _setup_c_impl, _check_c_impl,
              max_steps=14, max_tokens=1536, judge_path=True)


def _setup_c_fix(env):
    # r initialised to 0 -> always prints 0; correct factorial(5) is 120.
    env.write("buggy.py",
              "def fact(n):\n    r = 0\n    for i in range(1, n+1):\n"
              "        r *= i\n    return r\nprint(fact(5))\n")

def _check_c_fix(env, traj):
    proc = _run_file(env, "buggy.py")
    return bool(proc) and proc.stdout.strip() == "120"

C_FIX = Task("c_fix_bug", 3, "coding",
             "buggy.py is supposed to print the factorial of 5 (which is 120) but it "
             "prints the wrong number. Find and fix the bug in buggy.py so that "
             "running it prints exactly 120.",
             CODE_TOOLS, _setup_c_fix, _check_c_fix,
             max_steps=14, max_tokens=1536, judge_path=True,
             notes="r is initialised to 0 instead of 1.")


def _setup_c_transform(env):
    env.write("nums.txt", "\n".join(str(i) for i in range(1, 101)) + "\n")

def _check_c_transform(env, traj):
    try:
        return env.read("sum.txt").strip() == "5050"
    except Exception:
        return False

C_TRANSFORM = Task("c_code_transform", 2, "coding",
                   "nums.txt contains 100 integers, one per line. Write and run a "
                   "Python script that reads them and writes ONLY their total sum to "
                   "sum.txt.",
                   CODE_TOOLS, _setup_c_transform, _check_c_transform,
                   max_steps=10, max_tokens=1280)
```

- [ ] **Step 4: Append to `TASKS`** (done together with Tasks 4–5 in Task 5's TASKS edit; for now run the checker tests)

Run: `python3 -m pytest tests/test_checkers.py -k "c_impl or c_fix or c_transform" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): coding tasks (impl, fix-bug, code transform)"
```

---

## Task 4: Reasoning tasks (math / logic-grid / constraint-plan)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing checker tests**

Add to `tests/test_checkers.py`:

```python
from tasks.suite import (_check_r_math, _check_r_logic, _check_r_plan)


def test_r_math_check():
    env = _env()
    env.write("answer.txt", "12")
    assert _check_r_math(env, None) is True
    env.write("answer.txt", "8")
    assert _check_r_math(env, None) is False


def test_r_logic_check():
    env = _env()
    env.write("answer.txt", "Cara")
    assert _check_r_logic(env, None) is True
    env.write("answer.txt", "Ana")
    assert _check_r_logic(env, None) is False


def test_r_plan_accepts_any_valid_order():
    env = _env()
    env.write("plan.txt", "B A C D")
    assert _check_r_plan(env, None) is True


def test_r_plan_accepts_comma_separated():
    env = _env()
    env.write("plan.txt", "B,A,C,D")
    assert _check_r_plan(env, None) is True


def test_r_plan_rejects_constraint_violation():
    env = _env()
    env.write("plan.txt", "A B C D")  # B must precede A
    assert _check_r_plan(env, None) is False
    env.write("plan.txt", "B A D C")  # D must be last
    assert _check_r_plan(env, None) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k "r_math or r_logic or r_plan" -q`
Expected: FAIL (names not defined).

- [ ] **Step 3: Implement the tasks**

In `tasks/suite.py`, add a "Reasoning" section before `TASKS = [...]`:

```python
# --------------------------------------------------------------------------- #
# Reasoning — pure deduction, no tool-use confound (read/write only)
# --------------------------------------------------------------------------- #
def _setup_r_math(env):
    env.write("problem.txt",
              "A store sells pens at 3 for $2. John buys 12 pens and pays with a "
              "$20 bill. How many whole dollars of change does he get?")

def _check_r_math(env, traj):
    try:
        return env.read("answer.txt").strip() == "12"   # 12 pens=4*$2=$8; 20-8=12
    except Exception:
        return False

R_MATH = Task("r_multi_step_math", 2, "reasoning",
              "Read problem.txt and solve it. Write ONLY the final integer answer "
              "(number of dollars) to answer.txt.",
              BASE_TOOLS, _setup_r_math, _check_r_math,
              max_steps=6, max_tokens=1024)


def _setup_r_logic(env):
    env.write("problem.txt",
              "Three friends -- Ana, Ben, Cara -- each own a different pet: a cat, a "
              "dog, and a fish. Ana does not own the cat. Ben owns the fish. Who owns "
              "the cat?")

def _check_r_logic(env, traj):
    try:
        return env.read("answer.txt").strip() == "Cara"
    except Exception:
        return False

R_LOGIC = Task("r_logic_grid", 3, "reasoning",
               "Read problem.txt, solve the logic puzzle, and write ONLY the name of "
               "the person who owns the cat to answer.txt.",
               BASE_TOOLS, _setup_r_logic, _check_r_logic,
               max_steps=8, max_tokens=1024, judge_path=True)


def _setup_r_plan(env):
    env.write("problem.txt",
              "Order four tasks A, B, C, D, one per position. Constraints: A must "
              "come before C; B must come before A; D must be last.")

def _check_r_plan(env, traj):
    try:
        order = env.read("plan.txt").strip().replace(",", " ").split()
        if sorted(order) != ["A", "B", "C", "D"]:
            return False
        pos = {x: i for i, x in enumerate(order)}
        return pos["A"] < pos["C"] and pos["B"] < pos["A"] and pos["D"] == 3
    except Exception:
        return False

R_PLAN = Task("r_constraint_plan", 3, "reasoning",
              "Read problem.txt. Output a valid ordering of the four tasks (the four "
              "letters separated by spaces) to plan.txt.",
              BASE_TOOLS, _setup_r_plan, _check_r_plan,
              max_steps=8, max_tokens=1024, judge_path=True)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_checkers.py -k "r_math or r_logic or r_plan" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): reasoning tasks (math, logic grid, constraint plan)"
```

---

## Task 5: New agentic tier-4 task + register all new tasks in `TASKS`

**Files:**
- Modify: `tasks/suite.py` (add `t4c`, update module docstring, extend `TASKS`)
- Test: `tests/test_checkers.py` + `tests/test_runner.py` (suite-integrity check)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_checkers.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k "t4c or suite_has_unique" -q`
Expected: FAIL.

- [ ] **Step 3: Implement `t4c` and extend `TASKS`**

In `tasks/suite.py`, in the Tier-4 section (after `T4B`), add:

```python
def _setup_t4c(env):
    env.write("data/1.txt", "10")
    env.write("data/2.txt", "oops")   # corrupt: not an integer -> must be skipped
    env.write("data/3.txt", "32")

def _check_t4c(env, traj):
    try:
        return env.read("total.txt").strip() == "42"   # 10 + 32, skip "oops"
    except Exception:
        return False

T4C = Task("t4c_sum_skip_corrupt", 4, "agentic",
           "Each file in data/ should contain a single integer. Read every file, sum "
           "the integers, but SKIP any file whose contents are not a valid integer. "
           "Write only the total to total.txt.",
           BASE_TOOLS, _setup_t4c, _check_t4c, max_steps=14, max_tokens=1280,
           judge_path=True, notes="data/2.txt is non-numeric; tests error recovery.")
```

Replace the final `TASKS = [...]` line with:

```python
TASKS = [T1A, T1B, T2A, T2B, T2C, T3A, T3B, T4A, T4B, T4C,
         C_IMPL, C_FIX, C_TRANSFORM, R_MATH, R_LOGIC, R_PLAN]
```

Update the module docstring's first line from "12 tasks" / "9 tasks" wording to:

```python
"""Agentic + coding + reasoning task suite — 16 tasks across 4 difficulty tiers.
```

And in the Tiers docstring block, add after the tier list:

```python
Categories (capability axis, orthogonal to tier): retrieval, data, agentic,
coding (write+run code via run_python), reasoning (pure deduction).
```

- [ ] **Step 4: Run the full checker + suite tests**

Run: `python3 -m pytest tests/test_checkers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): agentic t4c + register coding/reasoning tasks (16 total)"
```

---

## Task 6: `by_category` aggregation + report table

**Files:**
- Modify: `harness/report.py` (add `by_category` to `aggregate`, add table to `write_markdown_report`)
- Test: `tests/test_report.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_report.py` (reuse the existing `RunResult` construction style in that file; this builds minimal rows):

```python
from harness.core import RunResult
from harness.report import aggregate


def _row(model, task_id, category, tier, success):
    return RunResult(task_id=task_id, tier=tier, category=category, model=model,
                     run_index=0, success=success, n_steps=3, invalid_rate=0.0,
                     tokens_used=100, halt_reason="done", wall_seconds=0.1)


def test_aggregate_has_by_category():
    rows = [
        _row("m", "c_impl_function", "coding", 3, True),
        _row("m", "c_fix_bug", "coding", 3, False),
        _row("m", "r_logic_grid", "reasoning", 3, True),
    ]
    agg = aggregate(rows)
    assert "by_category" in agg
    assert agg["by_category"]["m|coding"]["pass_at_1"] == 0.5
    assert agg["by_category"]["m|coding"]["pass_at_k"] == 1
    assert agg["by_category"]["m|reasoning"]["pass_at_1"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_report.py -k by_category -q`
Expected: FAIL (`by_category` missing).

- [ ] **Step 3: Implement aggregation**

In `harness/report.py::aggregate`, add a third group. Change the body to:

```python
def aggregate(rows: list[RunResult]) -> dict[str, Any]:
    by_task = defaultdict(list)
    by_tier = defaultdict(list)
    by_category = defaultdict(list)
    for r in rows:
        by_task[(r.model, r.task_id)].append(r)
        by_tier[(r.model, r.tier)].append(r)
        by_category[(r.model, r.category)].append(r)

    def summarize(group):
        succ = [1 if r.success else 0 for r in group]
        return {
            "n_runs": len(group),
            "pass_at_1": _mean(succ),
            "pass_at_k": 1 if any(succ) else 0,
            "mean_steps": _mean([r.n_steps for r in group]),
            "mean_invalid_rate": _mean([r.invalid_rate for r in group]),
            "mean_tokens": _mean([r.tokens_used for r in group]),
            "mean_judge": _mean([r.judge_score for r in group]),
        }

    return {
        "by_task": {f"{m}|{t}": summarize(g) for (m, t), g in by_task.items()},
        "by_tier": {f"{m}|tier{t}": summarize(g) for (m, t), g in by_tier.items()},
        "by_category": {f"{m}|{c}": summarize(g)
                        for (m, c), g in by_category.items()},
    }
```

- [ ] **Step 4: Add the report table**

In `write_markdown_report`, after the by-tier table loop and before `lines.append("\n## Per-task detail\n")`, insert:

```python
    lines.append("\n## Pass rates by capability\n")
    lines.append("| Model | Category | pass@1 | pass@k | mean steps | invalid% | judge |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(agg["by_category"]):
        s = agg["by_category"][key]
        model, cat = key.split("|")
        inv = f"{s['mean_invalid_rate']*100:.0f}%" if s['mean_invalid_rate'] is not None else "-"
        judge = s['mean_judge'] if s['mean_judge'] is not None else "-"
        lines.append(f"| {model} | {cat} | {s['pass_at_1']} | {s['pass_at_k']} "
                     f"| {s['mean_steps']} | {inv} | {judge} |")
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_report.py -q`
Expected: PASS.

```bash
git add harness/report.py tests/test_report.py
git commit -m "feat(report): per-capability (by_category) aggregation + table"
```

---

## Task 7: `--dump-trajectories` (runner sink + CLI flag)

**Files:**
- Modify: `harness/runner.py` (`run_task` + `run_study` gain `trajectory_sink`)
- Modify: `run_eval.py` (flag + sink wiring)
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_runner.py` (it already has fake adapters in `tests/conftest.py`; reuse whatever fixture drives `run_task` there — match the existing pattern for building a task + adapter). Minimal sink test:

```python
def test_run_task_invokes_trajectory_sink(scripted_adapter_factory):
    """A trajectory_sink receives (model, task_id, run_index, trajectory)."""
    # Build a 1-step task + adapter the same way other tests in this file do.
    # See existing tests for the scripted_adapter_factory / simple task helpers.
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
    from harness.runner import run_task
    run_task(T1B, adapter, "fake:model", 0, trajectory_sink=sink)
    assert captured["model"] == "fake:model"
    assert captured["task_id"] == "t1b_count_logs"
    assert "halt:" in captured["text"]
```

> NOTE TO IMPLEMENTER: the exact fake-adapter constructor name may differ. Open `tests/conftest.py` and `tests/test_runner.py`, find how existing tests build a scripted adapter that emits tool calls then finishes, and mirror that exactly. The assertion content (sink receives model/task_id/text) is what matters.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_runner.py -k trajectory_sink -q`
Expected: FAIL (`run_task` has no `trajectory_sink` parameter).

- [ ] **Step 3: Thread the sink through the runner**

In `harness/runner.py`, change `run_task`'s signature:

```python
def run_task(task: Task, adapter, model_name: str, run_index: int,
             judge_adapter=None, trajectory_sink=None) -> RunResult:
```

Inside `run_task`, after the judge block and before `finally:` (i.e. right after `judge_score, judge_rationale = ...` / the `if task.judge_path` block), add:

```python
        if trajectory_sink is not None:
            trajectory_sink(model_name, task.task_id, run_index, traj)
```

Change `run_study`'s signature and call site:

```python
def run_study(tasks: list[Task], models: dict[str, Any], k: int = 5,
              judge_adapter=None,
              progress: Optional[Callable[[str], None]] = None,
              trajectory_sink=None) -> list[RunResult]:
```

and update the `run_task(...)` call inside it to pass `trajectory_sink=trajectory_sink`:

```python
                    rows.append(run_task(task, adapter, model_name, i,
                                         judge_adapter,
                                         trajectory_sink=trajectory_sink))
```

- [ ] **Step 4: Wire the CLI flag**

In `run_eval.py`, add the argument (near the other `ap.add_argument` calls):

```python
    ap.add_argument("--dump-trajectories", action="store_true",
                    help="write each run's trajectory text to <out>/trajectories/")
```

In `main()`, after `judge = build_judge(args.judge)` and before the `run_study` call, build the sink:

```python
    sink = None
    if args.dump_trajectories:
        traj_dir = os.path.join(args.out, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        def sink(model, task_id, run_index, traj):
            safe = model.replace(":", "-").replace("/", "-")
            fp = os.path.join(traj_dir, f"{safe}__{task_id}__k{run_index}.txt")
            with open(fp, "w") as fh:
                fh.write(traj.as_text())
```

Change the `run_study(...)` call to pass it:

```python
    rows = run_study(tasks, models, k=args.k, judge_adapter=judge,
                     progress=lambda s: print("  ", s),
                     trajectory_sink=sink)
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_runner.py -q`
Expected: PASS.

```bash
git add harness/runner.py run_eval.py tests/test_runner.py
git commit -m "feat(runner): --dump-trajectories sink for per-run path dumps"
```

---

## Task 8: Docs — `CLAUDE.md` update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the architecture/gotchas**

Make these edits to `CLAUDE.md`:

1. In the `tasks/suite.py` line of the architecture tree, change "(currently 9 tasks across 4 tiers)" to "(16 tasks across 4 tiers; categories: retrieval/data/agentic/coding/reasoning)".

2. Under "The two extension points" → "Add a task", append a sentence:
   "Coding tasks use `CODE_TOOLS` (adds `run_python`) and grade by executing the model's code against held-out inputs; reasoning tasks use `BASE_TOOLS` and grade an exact answer file."

3. Add a new bullet under "Things that will bite you":
   "- **`run_python` executes untrusted generated code on the host** — sandboxed only by temp-dir cwd + `python3 -I` + a 10s timeout, NOT containerized. Fine for trusted models; never point it at adversarial input."

4. Add a bullet documenting the flag:
   "- **`--dump-trajectories`** writes each run's path text to `<out>/trajectories/<model>__<task>__k<i>.txt` for post-hoc diagnosis (trajectories are otherwise not persisted)."

- [ ] **Step 2: Verify the full suite still passes**

Run: `python3 -m pytest -q`
Expected: PASS (all tests).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document run_python, coding/reasoning tasks, --dump-trajectories"
```

---

## Task 9: Full-suite green gate (no code, verification only)

- [ ] **Step 1: Run the entire offline suite**

Run: `python3 -m pytest -q`
Expected: ALL PASS, no warnings about collection errors. If anything fails, fix before handing back.

- [ ] **Step 2: Smoke the CLI wiring offline (no model call)**

Run: `python3 run_eval.py --help`
Expected: help text shows `--dump-trajectories`.

- [ ] **Step 3: Report green status** to the dispatcher with the pytest summary line.

---

## Out of scope (do NOT do)

- Do not run the live `claude` CLI / model matrix — the dispatcher drives that separately.
- Do not dedup READMEs or remove stray dirs.
- Do not add network sandboxing/containers for `run_python`.
- Do not add new model adapters or the `--anthropic` run.

## Self-review notes (author)

- Spec coverage: coding (T3), reasoning (T4), agentic-deepen (T5), run_python (T2),
  balanced-brace parser (T1), by_category (T6), dump-trajectories (T7), docs (T8) — all
  spec sections mapped.
- Type consistency: `_run_python`/`_run_file` both use `sys.executable -I`, cwd=`env.root`,
  10s timeout; checkers call `_run_file`; `CODE_TOOLS = BASE_TOOLS + [RUN_PYTHON]`; sink
  signature `(model, task_id, run_index, traj)` identical in runner + CLI + test.
- The live model run is intentionally a separate phase (Section 4 of the spec), executed by
  the dispatcher after this plan is green.
```
