"""Starter agentic task suite — 12 tasks across 4 difficulty tiers.

All tasks live in a filesystem sandbox and are safe to run anywhere. They are
designed so a ~4B local model can plausibly clear tier 1 and some of tier 2,
while tiers 3-4 separate frontier from local. Each task ships a DETERMINISTIC
checker so the headline metric never depends on an LLM's opinion.

Tiers:
  1  single tool call
  2  2-4 step linear pipeline
  3  5-10 step conditional / looping
  4  multi-step requiring error recovery (a tool deliberately fails)

To extend: append Task objects to TASKS. Keep tier budgets pinned.
"""
from __future__ import annotations

import json
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
def _setup_t3a(env):
    env.write("logs/a.log", "INFO ok\nERROR bad\nERROR worse\n")
    env.write("logs/b.log", "INFO fine\n")
    env.write("logs/c.log", "ERROR oops\nWARN hm\n")
    env.write("logs/readme.txt", "not a log")

def _check_t3a(env, traj):
    try:
        got = json.loads(env.read("errors.json"))
        return got == {"a.log": 2, "b.log": 0, "c.log": 1}
    except Exception:
        return False

T3A = Task("t3a_error_counts", 3, "agentic",
           "In the logs/ directory, for every file ending in '.log', count lines "
           "containing 'ERROR'. Write a JSON object mapping each .log filename "
           "(without the logs/ prefix) to its error count, to errors.json. "
           "Ignore non-.log files.",
           BASE_TOOLS, _setup_t3a, _check_t3a, max_steps=12, max_tokens=1024,
           judge_path=True)


def _setup_t3b(env):
    for i in range(1, 6):
        env.write(f"nums/{i}.txt", str(i * i))

def _check_t3b(env, traj):
    try:
        # squares 1,4,9,16,25 -> sum 55
        return env.read("total.txt").strip() == "55"
    except Exception:
        return False

T3B = Task("t3b_sum_files", 3, "agentic",
           "Each file in nums/ contains one integer. Read them all, sum the "
           "values, and write only the total to total.txt.",
           BASE_TOOLS, _setup_t3b, _check_t3b, max_steps=12, max_tokens=1024,
           judge_path=True)


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


TASKS = [T1A, T1B, T2A, T2B, T2C, T3A, T3B, T4A, T4B]
