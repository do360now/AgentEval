# Procedural Task Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tasks instantiate from a random seed (Approach A: params via `env.scratch`) so instances can't be memorized, with each of the k repeats a distinct instance, and every procedural family guarded by a uniqueness/well-formedness property test.

**Architecture:** `Task` gains an optional `parametrize(rng)->dict`. The runner derives a per-(task,run_index) seed, builds params, stashes them in `env.scratch["params"]` before `setup`, and records the seed on `RunResult`. Procedural `setup`/`check` read `env.scratch["params"]`; the static goal text is unchanged and the per-instance problem is written to `problem.txt`/sandbox files. No signature changes to existing `setup`/`check`.

**Tech Stack:** Python 3.12 stdlib only (`random`, `hashlib`, `itertools`). Tests via `pytest`, offline. Use `python3`.

---

## Background the implementer needs

- Read `CLAUDE.md` and the spec `docs/superpowers/specs/2026-05-30-procedural-generation-design.md`.
- `Environment` (`harness/core.py`) has `self.scratch: dict` already. We add `self.seed: int = 0`.
- `make_environment(task)` (bottom of `core.py`) is the only place envs are built; `run_task` calls it.
- Existing checker tests in `tests/test_checkers.py` call helpers like `_check_r_math(env, None)` with a hand-populated env. For the 6 converted families those tests must move to building the env via `make_environment(TASK, seed)` so `env.scratch["params"]` is populated. Each task below shows the replacement test.
- Run the suite with `python3 -m pytest -q`. Commit after each task with the exact message given.
- Work on the current branch (`master`).

---

## Task 1: Core — `Task.parametrize`, seeded `make_environment`, `RunResult.seed`

**Files:**
- Modify: `harness/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_core.py`:

```python
import random
from harness.core import Task, Tool, Environment, RunResult, make_environment


def _noop_setup(env):
    pass

def _noop_check(env, traj):
    return True


def test_make_environment_populates_params_from_parametrize():
    t = Task("p_demo", 3, "reasoning", "goal", [], _noop_setup, _noop_check,
             parametrize=lambda rng: {"n": rng.randint(0, 1_000_000)})
    env_a = make_environment(t, seed=42)
    env_b = make_environment(t, seed=42)
    env_c = make_environment(t, seed=43)
    assert env_a.scratch["params"] == env_b.scratch["params"]   # same seed -> same params
    assert env_a.scratch["params"] != env_c.scratch["params"]   # different seed -> different
    assert env_a.seed == 42
    for e in (env_a, env_b, env_c):
        e.destroy()


def test_make_environment_static_task_has_empty_params():
    t = Task("s_demo", 1, "retrieval", "goal", [], _noop_setup, _noop_check)
    env = make_environment(t, seed=7)
    assert env.scratch["params"] == {}
    env.destroy()


def test_runresult_has_seed_in_row():
    r = RunResult(task_id="x", tier=1, category="c", model="m", run_index=0,
                  success=True, n_steps=1, invalid_rate=0.0, tokens_used=10,
                  halt_reason="done", wall_seconds=0.1, seed=12345)
    assert r.to_row()["seed"] == 12345
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_core.py -k "params or seed_in_row" -q`
Expected: FAIL (`parametrize` kw unknown / `seed` kw unknown).

- [ ] **Step 3: Implement core changes**

In `harness/core.py`:

Add `import random` near the top imports (with `import json`, `import shutil`, ...).

In `Environment.__init__`, add a seed attribute:

```python
    def __init__(self, root: Path):
        self.root = root
        self.scratch: dict[str, Any] = {}  # for tasks that report via a value
        self.seed: int = 0                  # set by make_environment for procedural tasks
```

In the `Task` dataclass, add the field after `notes`:

```python
    notes: str = ""
    # Optional: build per-instance params from a seeded RNG (procedural generation).
    # Returns a dict stashed in env.scratch["params"] before setup runs; the grader
    # reads it for the ground-truth answer. None => static task (params == {}).
    parametrize: Optional[Callable[["random.Random"], dict]] = None
```

In the `RunResult` dataclass, add a `seed` field after `judge_rationale` (defaults keep the
dataclass valid):

```python
    judge_score: Optional[float] = None   # 0..1, only if judge_path
    judge_rationale: str = ""
    seed: int = 0                         # the seed this instance was generated from
```

Replace `make_environment`:

```python
def make_environment(task: Task, seed: int = 0) -> Environment:
    root = Path(tempfile.mkdtemp(prefix=f"agenteval_{task.task_id}_"))
    env = Environment(root)
    env.seed = seed
    rng = random.Random(seed)
    env.scratch["params"] = task.parametrize(rng) if task.parametrize else {}
    task.setup(env)
    return env
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_core.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness/core.py tests/test_core.py
git commit -m "feat(core): Task.parametrize + seeded make_environment + RunResult.seed"
```

---

## Task 2: Runner — derive the seed, thread `base_seed`, record it

**Files:**
- Modify: `harness/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_runner.py` (mirror how existing tests build a scripted adapter + task —
reuse the `scripted_adapter_factory` fixture added previously):

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_runner.py -k records_seed -q`
Expected: FAIL (`run_task` has no `base_seed`; no `seed` on result).

- [ ] **Step 3: Implement**

In `harness/runner.py`:

Add `import hashlib` near the top imports.

Change `run_task` signature to add `base_seed`:

```python
def run_task(task: Task, adapter, model_name: str, run_index: int,
             judge_adapter=None, trajectory_sink=None, base_seed: int = 0) -> RunResult:
```

At the very start of `run_task`, derive the seed and pass it to `make_environment`:

```python
    seed = int(hashlib.sha256(
        f"{base_seed}:{task.task_id}:{run_index}".encode()).hexdigest()[:8], 16)
    env = make_environment(task, seed)
```

(Replace the existing `env = make_environment(task)` line.)

In the `return RunResult(...)` at the end of `run_task`, add `seed=seed,`:

```python
        wall_seconds=round(snap_seconds, 2),
        judge_score=judge_score, judge_rationale=judge_rationale,
        seed=seed,
    )
```

Change `run_study` signature to add `base_seed` and pass it through:

```python
def run_study(tasks: list[Task], models: dict[str, Any], k: int = 5,
              judge_adapter=None,
              progress: Optional[Callable[[str], None]] = None,
              trajectory_sink=None, base_seed: int = 0) -> list[RunResult]:
```

and the `run_task(...)` call inside it:

```python
                    rows.append(run_task(task, adapter, model_name, i,
                                         judge_adapter,
                                         trajectory_sink=trajectory_sink,
                                         base_seed=base_seed))
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness/runner.py tests/test_runner.py
git commit -m "feat(runner): derive per-instance seed, thread base_seed, record on result"
```

---

## Task 3: CLI `--seed` flag + report note + CSV seed column

**Files:**
- Modify: `run_eval.py`, `harness/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_report.py`:

```python
import csv as _csv
from harness.core import RunResult
from harness.report import write_csv


def test_write_csv_includes_seed(tmp_path):
    rows = [RunResult(task_id="t", tier=1, category="c", model="m", run_index=0,
                      success=True, n_steps=1, invalid_rate=0.0, tokens_used=5,
                      halt_reason="done", wall_seconds=0.1, seed=777)]
    p = tmp_path / "r.csv"
    write_csv(rows, str(p))
    got = list(_csv.DictReader(open(p)))
    assert got[0]["seed"] == "777"
```

- [ ] **Step 2: Run to verify failure (or confirm pass)**

Run: `python3 -m pytest tests/test_report.py -k includes_seed -q`
Expected: PASS automatically (CSV is built from `RunResult.to_row()`, which now includes
`seed`). If it passes, good — this test is a regression guard. If it FAILS, fix `write_csv`
to use all keys from `rows[0].to_row()` (it already does).

- [ ] **Step 3: Add `--seed` flag**

In `run_eval.py`, add the argument near the others:

```python
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed for procedural task generation (default 0)")
```

Pass it into `run_study` (update the existing call):

```python
    rows = run_study(tasks, models, k=args.k, judge_adapter=judge,
                     progress=lambda s: print("  ", s),
                     trajectory_sink=sink, base_seed=args.seed)
```

- [ ] **Step 4: Add a report note**

In `harness/report.py::write_markdown_report`, change the header lines so the second line
notes procedural generation:

```python
    lines = ["# Agentic Eval — Results\n",
             f"Total runs: {len(rows)}\n",
             "_Procedurally-generated tasks: each repeat is a distinct seeded instance, "
             "so pass@k = solved ≥ 1 of k distinct instances._\n",
             "## Pass rates by tier\n",
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_report.py -q`
Expected: PASS.

```bash
git add run_eval.py harness/report.py tests/test_report.py
git commit -m "feat(cli): --seed flag; report notes procedural pass@k semantics"
```

---

## Task 4: Generator helpers + `r_multi_step_math` (simplest family)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_checkers.py`, REPLACE the existing `test_r_math_check` with seed-based tests
(import `make_environment`, `random` at top of file if not present):

```python
from harness.core import make_environment
from tasks.suite import R_MATH, _gen_r_math
import random


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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k r_math -q`
Expected: FAIL (`_gen_r_math` undefined).

- [ ] **Step 3: Implement the generator + rewrite the task**

In `tasks/suite.py`, add `import random` at the top if not present. Add a reasoning-helpers
section, and replace the existing `_setup_r_math` / `_check_r_math` / `R_MATH` block with:

```python
def _gen_r_math(rng: random.Random) -> dict:
    group_size = rng.randint(2, 4)
    group_price = rng.randint(1, 5)
    num_groups = rng.randint(3, 8)
    num_pens = group_size * num_groups
    cost = num_groups * group_price
    bill = next(b for b in (20, 50, 100) if b > cost)
    answer = bill - cost
    text = (f"A store sells pens at {group_size} for ${group_price}. A customer buys "
            f"{num_pens} pens and pays with a ${bill} bill. How many whole dollars of "
            f"change do they get?")
    return {"group_size": group_size, "group_price": group_price,
            "num_pens": num_pens, "bill": bill, "answer": str(answer),
            "problem_text": text}

def _setup_r_math(env):
    env.write("problem.txt", env.scratch["params"]["problem_text"])

def _check_r_math(env, traj):
    try:
        return env.read("answer.txt").strip() == env.scratch["params"]["answer"]
    except Exception:
        return False

R_MATH = Task("r_multi_step_math", 2, "reasoning",
              "Read problem.txt and solve it. Write ONLY the final integer answer "
              "(number of dollars) to answer.txt.",
              BASE_TOOLS, _setup_r_math, _check_r_math,
              max_steps=6, max_tokens=1024, parametrize=_gen_r_math)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_checkers.py -k r_math -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): procedural r_multi_step_math + generator pattern"
```

---

## Task 5: `r_logic_grid` (brute-force uniqueness guarantee)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_checkers.py`, REPLACE the existing `test_r_logic_check` with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k r_logic -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `tasks/suite.py`, add `import itertools` at top if absent. Replace the
`_setup_r_logic`/`_check_r_logic`/`R_LOGIC` block with:

```python
_LOGIC_NAMES = ["Ana", "Ben", "Cara", "Dan", "Eve", "Finn"]
_LOGIC_PETS = ["cat", "dog", "fish", "bird", "rabbit", "hamster"]

def _logic_solutions(people, pets, clues):
    """All assignments (person->pet) consistent with the clues."""
    out = []
    for perm in itertools.permutations(pets, len(people)):
        assign = dict(zip(people, perm))
        ok = True
        for kind, person, pet in clues:
            if kind == "not" and assign[person] == pet:
                ok = False
                break
            if kind == "is" and assign[person] != pet:
                ok = False
                break
        if ok:
            out.append(assign)
    return out

def _render_logic(people, pets, clues, queried_pet):
    lines = [f"{len(people)} friends -- {', '.join(people)} -- each own a different pet: "
             f"{', '.join(pets)}. Each person owns exactly one pet."]
    for kind, person, pet in clues:
        if kind == "not":
            lines.append(f"{person} does not own the {pet}.")
        else:
            lines.append(f"{person} owns the {pet}.")
    lines.append(f"Who owns the {queried_pet}?")
    return "\n".join(lines)

def _gen_r_logic(rng: random.Random) -> dict:
    n = 3
    people = rng.sample(_LOGIC_NAMES, n)
    pets = rng.sample(_LOGIC_PETS, n)
    sol_pets = rng.sample(pets, n)
    solution = dict(zip(people, sol_pets))
    candidates = [("not", p, q) for p in people for q in pets if solution[p] != q]
    rng.shuffle(candidates)
    clues = []
    for c in candidates:
        clues.append(c)
        if len(_logic_solutions(people, pets, clues)) == 1:
            break
    if len(_logic_solutions(people, pets, clues)) != 1:
        for p in people:                      # fallback: pin positives until unique
            clues.append(("is", p, solution[p]))
            if len(_logic_solutions(people, pets, clues)) == 1:
                break
    queried_pet = rng.choice(pets)
    answer = next(p for p in people if solution[p] == queried_pet)
    return {"people": people, "pets": pets, "clues": clues,
            "queried_pet": queried_pet, "answer": answer,
            "problem_text": _render_logic(people, pets, clues, queried_pet)}

def _setup_r_logic(env):
    env.write("problem.txt", env.scratch["params"]["problem_text"])

def _check_r_logic(env, traj):
    try:
        return env.read("answer.txt").strip() == env.scratch["params"]["answer"]
    except Exception:
        return False

R_LOGIC = Task("r_logic_grid", 3, "reasoning",
               "Read problem.txt, solve the logic puzzle, and write ONLY the name of "
               "the person who owns the pet the puzzle asks about to answer.txt.",
               BASE_TOOLS, _setup_r_logic, _check_r_logic,
               max_steps=8, max_tokens=1024, judge_path=True, parametrize=_gen_r_logic)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_checkers.py -k r_logic -q`
Expected: PASS (the 200-seed uniqueness test is the key guard).

- [ ] **Step 5: Commit**

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): procedural r_logic_grid with brute-force uniqueness guarantee"
```

---

## Task 6: `r_constraint_plan` (validator-style, multiple valid orders)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_checkers.py`, REPLACE the existing `test_r_plan_*` tests with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k r_plan -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `tasks/suite.py`, replace the `_setup_r_plan`/`_check_r_plan`/`R_PLAN` block with:

```python
def _render_plan(items, constraints):
    lines = [f"Order these {len(items)} tasks, one per position: {', '.join(items)}. "
             "Constraints:"]
    for c in constraints:
        if c[0] == "before":
            lines.append(f"- {c[1]} must come before {c[2]}.")
        else:
            lines.append(f"- {c[1]} must be last.")
    return "\n".join(lines)

def _gen_r_plan(rng: random.Random) -> dict:
    n = rng.choice([4, 5])
    items = [chr(ord("A") + i) for i in range(n)]
    valid_order = items[:]
    rng.shuffle(valid_order)
    pos = {x: i for i, x in enumerate(valid_order)}
    pairs = [(a, b) for a in items for b in items if pos[a] < pos[b]]
    rng.shuffle(pairs)
    k = rng.randint(n - 1, n + 1)
    constraints = [("before", a, b) for a, b in pairs[:k]]
    if rng.random() < 0.5:
        constraints.append(("last", valid_order[-1]))
    return {"items": items, "constraints": constraints, "valid_order": valid_order,
            "problem_text": _render_plan(items, constraints)}

def _setup_r_plan(env):
    env.write("problem.txt", env.scratch["params"]["problem_text"])

def _check_r_plan(env, traj):
    try:
        params = env.scratch["params"]
        order = env.read("plan.txt").strip().replace(",", " ").split()
    except Exception:
        return False
    if sorted(order) != sorted(params["items"]):
        return False
    pos = {x: i for i, x in enumerate(order)}
    for c in params["constraints"]:
        if c[0] == "before" and not pos[c[1]] < pos[c[2]]:
            return False
        if c[0] == "last" and pos[c[1]] != len(order) - 1:
            return False
    return True

R_PLAN = Task("r_constraint_plan", 3, "reasoning",
              "Read problem.txt. Output a valid ordering of the tasks (the letters "
              "separated by spaces) to plan.txt.",
              BASE_TOOLS, _setup_r_plan, _check_r_plan,
              max_steps=8, max_tokens=1024, judge_path=True, parametrize=_gen_r_plan)
```

- [ ] **Step 4: Run tests + commit**

Run: `python3 -m pytest tests/test_checkers.py -k r_plan -q`
Expected: PASS.

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): procedural r_constraint_plan (validator-style grading)"
```

---

## Task 7: `t3a_error_counts` (random counts + decoys)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_checkers.py`, REPLACE the existing `t3a` checker test with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k t3a -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Replace the `_setup_t3a`/`_check_t3a`/`T3A` block:

```python
def _gen_t3a(rng: random.Random) -> dict:
    nfiles = rng.randint(2, 4)
    names = [f"{chr(ord('a') + i)}.log" for i in range(nfiles)]
    answer = {nm: rng.randint(0, 3) for nm in names}
    files = {}
    for nm, ct in answer.items():
        lines = ["ERROR something failed"] * ct + ["INFO ok", "WARN heads up"]
        rng.shuffle(lines)
        files["logs/" + nm] = "\n".join(lines) + "\n"
    files["logs/readme.txt"] = "this is not a log file\n"
    return {"files": files, "answer": answer}

def _setup_t3a(env):
    for path, content in env.scratch["params"]["files"].items():
        env.write(path, content)

def _check_t3a(env, traj):
    try:
        return json.loads(env.read("errors.json")) == env.scratch["params"]["answer"]
    except Exception:
        return False

T3A = Task("t3a_error_counts", 3, "agentic",
           "In the logs/ directory, for every file ending in '.log', count lines "
           "containing 'ERROR'. Write a JSON object mapping each .log filename "
           "(without the logs/ prefix) to its error count, to errors.json. "
           "Ignore non-.log files.",
           BASE_TOOLS, _setup_t3a, _check_t3a, max_steps=12, max_tokens=1024,
           judge_path=True, parametrize=_gen_t3a)
```

- [ ] **Step 4: Run tests + commit**

Run: `python3 -m pytest tests/test_checkers.py -k t3a -q`
Expected: PASS.

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): procedural t3a_error_counts (random counts + decoys)"
```

---

## Task 8: `t3b_sum_files` (random integers)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_checkers.py`, REPLACE the existing `t3b` checker test with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k t3b -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Replace the `_setup_t3b`/`_check_t3b`/`T3B` block:

```python
def _gen_t3b(rng: random.Random) -> dict:
    n = rng.randint(4, 6)
    vals = [rng.randint(1, 50) for _ in range(n)]
    files = {f"nums/{i + 1}.txt": str(v) for i, v in enumerate(vals)}
    return {"files": files, "answer": str(sum(vals))}

def _setup_t3b(env):
    for path, content in env.scratch["params"]["files"].items():
        env.write(path, content)

def _check_t3b(env, traj):
    try:
        return env.read("total.txt").strip() == env.scratch["params"]["answer"]
    except Exception:
        return False

T3B = Task("t3b_sum_files", 3, "agentic",
           "Each file in nums/ contains one integer. Read them all, sum the "
           "values, and write only the total to total.txt.",
           BASE_TOOLS, _setup_t3b, _check_t3b, max_steps=12, max_tokens=1024,
           judge_path=True, parametrize=_gen_t3b)
```

- [ ] **Step 4: Run tests + commit**

Run: `python3 -m pytest tests/test_checkers.py -k t3b -q`
Expected: PASS.

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): procedural t3b_sum_files (random integers)"
```

---

## Task 9: `c_impl_function` (per-seed hidden test cases)

**Files:**
- Modify: `tasks/suite.py`
- Test: `tests/test_checkers.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_checkers.py`, REPLACE `test_c_impl_passes_correct_solution` and
`test_c_impl_fails_wrong_solution` with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_checkers.py -k c_impl -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Replace the `_setup_c_impl`/`_check_c_impl`/`C_IMPL` block. Keep `_run_file` as-is (defined
earlier in the file):

```python
def _gen_c_impl(rng: random.Random) -> dict:
    cases = []
    for _ in range(5):
        lst = [rng.randint(-9, 9) for _ in range(rng.randint(0, 6))]
        cases.append((lst, sum(x for x in lst if x % 2 == 0)))
    return {"cases": cases}

def _setup_c_impl(env):
    pass  # the spec is in the goal; cases live in params for the grader

def _check_c_impl(env, traj):
    cases = env.scratch["params"]["cases"]
    grader = ("import solution\n"
              f"cases = {cases!r}\n"
              "print('PASS' if all(solution.solve(i) == e for i, e in cases) "
              "else 'FAIL')\n")
    try:
        env.write("_grader.py", grader)
    except Exception:
        return False
    proc = _run_file(env, "_grader.py")
    return bool(proc) and proc.stdout.strip().endswith("PASS")

C_IMPL = Task("c_impl_function", 3, "coding",
              "Write a function solve(nums) in solution.py that returns the sum of "
              "the EVEN integers in the list nums. Example: solve([1,2,3,4]) returns "
              "6. Note: negative even numbers count too. Do not call the function "
              "yourself; just define it.",
              CODE_TOOLS, _setup_c_impl, _check_c_impl,
              max_steps=14, max_tokens=1536, judge_path=True, parametrize=_gen_c_impl)
```

> NOTE: the goal now states negative evens count, because random cases include negatives and
> the reference sums them. Keep this wording.

- [ ] **Step 4: Run tests + commit**

Run: `python3 -m pytest tests/test_checkers.py -k c_impl -q`
Expected: PASS.

```bash
git add tasks/suite.py tests/test_checkers.py
git commit -m "feat(suite): procedural c_impl_function (per-seed hidden cases)"
```

---

## Task 10: Docs + full green gate

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md`**

1. In the "Add a task" extension-point bullet, append:
   "For **procedural** tasks set `parametrize(rng)->dict`; the dict is stashed in
   `env.scratch['params']` before `setup` and read by both `setup` (to render `problem.txt`
   / input files) and `check` (for the ground-truth answer). The generator must guarantee a
   unique, well-formed instance and is guarded by a per-family property test over 200 seeds."

2. Add a bullet under "Methodology guardrails":
   "- **Procedural generation (contamination resistance).** Seeded tasks instantiate from
   `--seed` + task_id + run_index, so the k repeats are k *distinct* instances and pass@k =
   'solved ≥1 of k distinct instances'. The seed is recorded per row in `results.csv` to
   regenerate any instance."

3. Update the `tasks/suite.py` architecture line to mention "(several tasks are
   procedurally generated via `Task.parametrize`)".

- [ ] **Step 2: Full suite green gate**

Run: `python3 -m pytest -q`
Expected: ALL PASS. The 200-seed property tests must keep total runtime low (still well
under a couple seconds). If any property test fails, a generator is producing a degenerate
instance — fix the generator (not the test).

- [ ] **Step 3: Smoke the seed flow offline**

Run: `python3 -c "from harness.core import make_environment; from tasks.suite import R_LOGIC; e=make_environment(R_LOGIC, 1); print(e.read('problem.txt')); e.destroy()"`
Expected: prints a rendered logic puzzle with a 'Who owns the ...?' line.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document procedural generation (parametrize, seeds, pass@k)"
```

---

## Out of scope (do NOT do)

- Do not run the live `claude` CLI / model matrix.
- Do not convert tier-1 tasks or `c_fix`/`c_transform`/`t4*` (later batch).
- Do not change report aggregation logic beyond the one note line.
- Do not implement the helloai `eval-summary.json` / token-split / pass@k-aggregation changes
  (tracked separately).

## Self-review notes (author)

- Spec coverage: parametrize field + seeded env (T1), runner seed threading (T2), --seed +
  CSV seed + report note (T3), 6 procedural families each with a property test (T4–T9), docs
  + green gate (T10). All spec sections mapped.
- Type consistency: every family uses `env.scratch["params"]`; generators return a dict with
  `answer` (or validator data) + render text/files; `_gen_*` names match between suite and
  tests; `make_environment(task, seed)` signature consistent across core/runner/tests;
  `parametrize=_gen_*` wired on each converted Task.
- Property tests over 200 seeds are the uniqueness/well-formedness guard the spec requires.
- Existing static-value checker tests for the 6 families are REPLACED (not added alongside),
  since those tasks now require `env.scratch["params"]`.
```
