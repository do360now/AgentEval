"""Agentic + coding + reasoning task suite — 18 tasks across 5 difficulty tiers.

All tasks live in a filesystem sandbox and are safe to run anywhere. They are
designed so a ~4B local model can plausibly clear tier 1 and some of tier 2,
while tiers 3-4 separate frontier from local. Each task ships a DETERMINISTIC
checker so the headline metric never depends on an LLM's opinion.

Tiers:
  1  single tool call
  2  2-4 step linear pipeline
  3  5-10 step conditional / looping
  4  multi-step requiring error recovery (a tool deliberately fails)
  5  hard discriminators (property-tested coding + long-horizon agentic) meant
     to separate frontier models that saturate tiers 1-4

Categories (capability axis, orthogonal to tier): retrieval, data, agentic,
coding (write+run code via run_python), reasoning (pure deduction).

To extend: append Task objects to TASKS. Keep tier budgets pinned.
"""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys

from harness.core import Environment, Task, Tool, Trajectory


# --------------------------------------------------------------------------- #
# Shared tools
# --------------------------------------------------------------------------- #
def _read(env: Environment, path: str):
    return env.read(path)

def _write(env: Environment, path: str, content: str):
    env.write(path, content)
    return f"wrote {len(content)} chars to {path}"

def _list(env: Environment, path: str = "."):
    return env.listdir(path)

def _stat(env: Environment, path: str):
    p = env.path(path)
    return {"exists": p.exists(),
            "size": p.stat().st_size if p.exists() else None}

READ = Tool("read_file", "Read a text file's contents.",
            {"type": "object", "properties": {"path": {"type": "string"}},
             "required": ["path"]}, _read)
WRITE = Tool("write_file", "Write text to a file (creates/overwrites).",
             {"type": "object", "properties": {"path": {"type": "string"},
              "content": {"type": "string"}}, "required": ["path", "content"]},
             _write)
LIST = Tool("list_dir", "List file names in a directory.",
            {"type": "object", "properties": {"path": {"type": "string"}}},
            _list)
STAT = Tool("stat_file", "Check if a file exists and its size.",
            {"type": "object", "properties": {"path": {"type": "string"}},
             "required": ["path"]}, _stat)

BASE_TOOLS = [READ, WRITE, LIST, STAT]


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


# --------------------------------------------------------------------------- #
# Tier 1 — single tool call
# --------------------------------------------------------------------------- #
def _setup_t1a(env):
    env.write("config.json", json.dumps({"timeout": 30, "retries": 3}))

def _check_t1a(env, traj):
    try:
        return json.loads(env.read("answer.txt").strip())["timeout"] == 30 \
            if env.path("answer.txt").exists() and \
            env.read("answer.txt").strip().startswith("{") \
            else env.read("answer.txt").strip() == "30"
    except Exception:
        return False

T1A = Task("t1a_read_value", 1, "retrieval",
           "Read config.json, find the value of 'timeout', and write just that "
           "value to answer.txt.",
           BASE_TOOLS, _setup_t1a, _check_t1a, max_steps=4, max_tokens=512)


def _setup_t1b(env):
    for n in ["alpha.txt", "beta.txt", "gamma.log"]:
        env.write(n, "x")

def _check_t1b(env, traj):
    try:
        return env.read("answer.txt").strip() == "1"
    except Exception:
        return False

T1B = Task("t1b_count_logs", 1, "retrieval",
           "Count how many files in the current directory end with '.log'. "
           "Write only that number to answer.txt.",
           BASE_TOOLS, _setup_t1b, _check_t1b, max_steps=4, max_tokens=512)


# --------------------------------------------------------------------------- #
# Tier 2 — 2-4 step linear pipeline
# --------------------------------------------------------------------------- #
def _setup_t2a(env):
    env.write("data.csv", "name,revenue\na,100\nb,200\nc,300\n")

def _check_t2a(env, traj):
    try:
        return env.read("result.txt").strip() == "200"
    except Exception:
        return False

T2A = Task("t2a_csv_mean", 2, "data",
           "Read data.csv. Compute the mean of the 'revenue' column. Write only "
           "the integer mean to result.txt.",
           BASE_TOOLS, _setup_t2a, _check_t2a, max_steps=6, max_tokens=768)


def _setup_t2b(env):
    env.write("in.txt", "banana\napple\ncherry\n")

def _check_t2b(env, traj):
    try:
        return env.read("sorted.txt").strip().splitlines() == \
            ["apple", "banana", "cherry"]
    except Exception:
        return False

T2B = Task("t2b_sort_lines", 2, "data",
           "Read in.txt, sort its lines alphabetically, and write the result to "
           "sorted.txt (one item per line).",
           BASE_TOOLS, _setup_t2b, _check_t2b, max_steps=6, max_tokens=768)


def _setup_t2c(env):
    env.write("note.txt", "the quick brown fox")

def _check_t2c(env, traj):
    try:
        return env.read("upper.txt").strip() == "THE QUICK BROWN FOX"
    except Exception:
        return False

T2C = Task("t2c_transform", 2, "data",
           "Read note.txt, convert its contents to uppercase, write to upper.txt.",
           BASE_TOOLS, _setup_t2c, _check_t2c, max_steps=6, max_tokens=768)


# --------------------------------------------------------------------------- #
# Tier 3 — 5-10 steps, conditional / looping
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Tier 4 — multi-step requiring error recovery
# --------------------------------------------------------------------------- #
def _setup_t4a(env):
    # The instructions point at primary.json, which is intentionally MISSING.
    # A fallback backup.json exists. The agent must detect the failure and adapt.
    env.write("backup.json", json.dumps({"port": 8080}))

def _check_t4a(env, traj):
    try:
        # Genuine recovery, not a lucky guess: the agent must have actually hit
        # the missing-file error (a valid call returning an ERROR observation)
        # AND produced the correct fallback value.
        recovered = any(s.valid and "ERROR" in str(s.observation)
                        for s in traj.steps)
        return recovered and env.read("port.txt").strip() == "8080"
    except Exception:
        return False

T4A = Task("t4a_recover_fallback", 4, "agentic",
           "Read primary.json and write its 'port' value to port.txt. If "
           "primary.json cannot be read, fall back to backup.json instead.",
           BASE_TOOLS, _setup_t4a, _check_t4a, max_steps=14, max_tokens=1280,
           judge_path=True,
           notes="primary.json is deliberately absent; tests error recovery.")


def _setup_t4b(env):
    env.write("orders.csv", "id,qty,price\n1,2,abc\n2,3,5\n3,1,10\n")

def _check_t4b(env, traj):
    # Valid rows: (3*5)+(1*10)=25. Row 1 has non-numeric price -> must be skipped.
    try:
        return env.read("revenue.txt").strip() == "25"
    except Exception:
        return False

T4B = Task("t4b_skip_bad_rows", 4, "agentic",
           "Read orders.csv. Compute total revenue as sum of qty*price across "
           "rows. Some rows contain malformed (non-numeric) values -- skip those "
           "rows. Write only the integer total to revenue.txt.",
           BASE_TOOLS, _setup_t4b, _check_t4b, max_steps=14, max_tokens=1280,
           judge_path=True, notes="row 1 price is non-numeric; tests robustness.")


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


# --------------------------------------------------------------------------- #
# Coding — write/execute code, graded on real run output
# --------------------------------------------------------------------------- #
def _run_file(env, name):
    """Run a sandbox file with the test interpreter; return CompletedProcess|None.

    Uses cwd=env.root (no -I) so that ``import`` statements can find sibling
    modules in the sandbox directory (e.g. the grader importing solution.py).
    """
    try:
        return subprocess.run([sys.executable, str(env.path(name))],
                              cwd=str(env.root), capture_output=True,
                              text=True, timeout=10)
    except Exception:
        return None


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


# --------------------------------------------------------------------------- #
# Reasoning — pure deduction, no tool-use confound (read/write only)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Tier 5 — hard discriminators (designed to separate frontier models)
# --------------------------------------------------------------------------- #
def _setup_noop(env):
    pass  # params-only task; the grader/spec needs no seeded files


# -- A: algorithmic coding, graded by property testing -------------------- #
def _merge_ref(intervals):
    """Reference: merge overlapping/touching intervals, sorted by start."""
    if not intervals:
        return []
    s = sorted([list(iv) for iv in intervals])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:               # touching (a == end) counts as overlap
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def _gen_h_merge(rng: random.Random) -> dict:
    cases = [([], []), ([[5, 5]], [[5, 5]]), ([[1, 2], [2, 3]], [[1, 3]])]  # edges
    for _ in range(7):
        ivs = []
        for _ in range(rng.randint(3, 8)):
            a = rng.randint(0, 20)
            ivs.append([a, a + rng.randint(0, 8)])
        rng.shuffle(ivs)
        cases.append((ivs, _merge_ref(ivs)))
    return {"cases": cases}

def _check_h_merge(env, traj):
    cases = [(c[0], c[1]) for c in env.scratch["params"]["cases"]]
    grader = (
        "import solution\n"
        f"cases = {cases!r}\n"
        "norm = lambda x: [list(p) for p in x]\n"
        "print('PASS' if all(norm(solution.merge(i)) == norm(e) "
        "for i, e in cases) else 'FAIL')\n"
    )
    try:
        env.write("_grader.py", grader)
    except Exception:
        return False
    proc = _run_file(env, "_grader.py")
    return bool(proc) and proc.stdout.strip().endswith("PASS")

H_MERGE = Task("h_merge_intervals", 5, "coding",
               "Write a function merge(intervals) in solution.py. It takes a list of "
               "[start, end] integer intervals (possibly unsorted and overlapping) and "
               "returns the list of merged, non-overlapping intervals sorted by start. "
               "Intervals that merely TOUCH at an endpoint (e.g. [1,2] and [2,3]) must be "
               "merged into one ([1,3]). An empty input returns an empty list. Example: "
               "merge([[1,3],[2,6],[8,10]]) returns [[1,6],[8,10]]. Define the function "
               "only; do not call it.",
               CODE_TOOLS, _setup_noop, _check_h_merge,
               max_steps=16, max_tokens=2048, judge_path=True, parametrize=_gen_h_merge)


# -- B: long-horizon agentic data-join (BASE_TOOLS only -> mental bookkeeping) #
_REGIONS = ["North", "South", "East", "West"]

def _gen_h_join(rng: random.Random) -> dict:
    customers = {f"C{i+1}": rng.choice(_REGIONS) for i in range(rng.randint(3, 5))}
    totals: dict = {}
    orders = []
    for j in range(rng.randint(10, 15)):
        cid = rng.choice(list(customers))
        amt = rng.randint(1, 99)
        orders.append((f"O{j+1}", cid, amt))
        totals[customers[cid]] = totals.get(customers[cid], 0) + amt
    cust_csv = "customer_id,region\n" + "".join(
        f"{c},{r}\n" for c, r in customers.items())
    order_csv = "order_id,customer_id,amount\n" + "".join(
        f"{o},{c},{a}\n" for o, c, a in orders)
    return {"customers_csv": cust_csv, "orders_csv": order_csv, "answer": totals}

def _setup_h_join(env):
    p = env.scratch["params"]
    env.write("customers.csv", p["customers_csv"])
    env.write("orders.csv", p["orders_csv"])

def _check_h_join(env, traj):
    try:
        got = {k: int(v) for k, v in json.loads(env.read("region_totals.json")).items()}
        return got == env.scratch["params"]["answer"]
    except Exception:
        return False

H_JOIN = Task("h_revenue_by_region", 5, "agentic",
              "orders.csv has columns order_id,customer_id,amount. customers.csv has "
              "columns customer_id,region. For each region, sum the amount of all orders "
              "placed by customers in that region. Write a JSON object mapping region "
              "name to its integer total to region_totals.json. Only include regions that "
              "have at least one order.",
              BASE_TOOLS, _setup_h_join, _check_h_join,
              max_steps=10, max_tokens=1536, judge_path=True, parametrize=_gen_h_join)


TASKS = [T1A, T1B, T2A, T2B, T2C, T3A, T3B, T4A, T4B, T4C,
         C_IMPL, C_FIX, C_TRANSFORM, R_MATH, R_LOGIC, R_PLAN,
         H_MERGE, H_JOIN]
