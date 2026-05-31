# Procedural Task Generation — Design

**Date:** 2026-05-30
**Status:** Approved (Approach A), ready for implementation plan
**Goal:** Make agenteval tasks **contamination-resistant** by instantiating them from a random
seed, so a model cannot have memorized the exact instance from training data. Each of the k
repeats becomes a *distinct* instance, strengthening pass@k into a real capability ceiling.

## Context & constraints

- Builds on the current harness: deterministic `check()` against the FINAL sandbox, pinned
  per-task budgets, strict module boundaries (loop ⊥ tool ⊥ grader). All preserved.
- **Approved decision — k semantics:** the k repeats use **k different generated instances**
  (fresh seed per `run_index`). pass@1 averages over instance difficulty; pass@k =
  "solved ≥1 of k *distinct* instances." Documented as a methodology change.
- This is the foundational item of a 4-part roadmap (procedural-gen → agent-safety →
  model-as-a-tool → real-code-in-container). It is built first so later task families are
  born contamination-proof.

## Approach A (approved): params via `env.scratch`

`Task` gains one optional field; the runner threads a seed → params → the environment; the
grader reads params from the env it already sees. **No signature changes** to existing
`setup`/`check` functions or their tests.

### Core mechanism

1. `Task` gains:
   ```python
   parametrize: Optional[Callable[[random.Random], dict]] = None
   ```
   `None` → a static task (today's behavior, `params == {}`).

2. Seed derivation (deterministic + reproducible), in the runner:
   ```python
   seed = int(hashlib.sha256(
       f"{base_seed}:{task.task_id}:{run_index}".encode()).hexdigest()[:8], 16)
   ```
   `base_seed` comes from a new `--seed` CLI flag (default 0). Each of the k repeats differs
   because `run_index` varies; reruns with the same `--seed` reproduce exactly.

3. `make_environment(task, seed)` builds `rng = random.Random(seed)`, computes
   `params = task.parametrize(rng) if task.parametrize else {}`, stores
   `env.scratch["params"] = params` and `env.seed = seed`, THEN calls `task.setup(env)`.

4. Procedural tasks' `setup(env)` reads `env.scratch["params"]` to render a per-instance
   `problem.txt` (the file-based pattern the reasoning tasks already use). The **static goal
   text never changes** — it stays generic, e.g. "Read problem.txt, solve it, write the
   answer to answer.txt." `check(env, traj)` reads `env.scratch["params"]` for the
   ground-truth answer.

5. `RunResult` gains `seed: int`; it is written to `results.csv` so any instance can be
   regenerated. `run_task` / `run_study` gain a `base_seed: int = 0` parameter.

### The generator contract (load-bearing)

`parametrize(rng) -> dict` MUST return a params dict that contains:
- everything `setup` needs to render the instance (`problem_text`, input file contents, …),
  **and**
- the ground-truth answer `check` will compare against (e.g. `params["answer"]`, or for
  validator-style tasks the data needed to validate, like the constraint set).

For puzzle families the generator **must guarantee a unique, well-formed solution**. Strategy:
generate-and-verify — draw parameters from `rng`, run a brute-force solver, and if the
instance is degenerate (0 or >1 solutions) re-draw, up to a bounded number of attempts
(e.g. 50) before raising (a generator that can't produce a valid instance is a bug, surfaced
loudly, not silently shipped).

Every procedural family ships a **property test**: over N≥200 seeds, assert the generated
instance is well-formed and its embedded answer is the unique correct solution. This is the
guard that procedural generation never silently produces unfair/unsolvable tasks.

## First batch — 6 families (establishes the pattern + reusable helpers)

Convert the discriminating mid-tier tasks; leave tier-1 static (too trivial to memorize).

| family | from | parametrization | uniqueness |
|---|---|---|---|
| `r_logic_grid` | static puzzle | random people/pets/clue set | brute-force solver asserts exactly one assignment |
| `r_multi_step_math` | fixed numbers | random quantities/prices within ranges | answer computed directly from params |
| `r_constraint_plan` | fixed constraints | random ordering constraints over N items | solver asserts ≥1 valid order; checker validates against the constraint set (already validator-style) |
| `t3a_error_counts` | fixed logs | random per-file ERROR counts + decoy non-.log files | answer = the generated count map |
| `t3b_sum_files` | fixed squares | random integers in N files | answer = their sum |
| `c_impl_function` | fixed held-out cases | random test cases generated per seed (still hidden from the goal) | reference impl in the grader computes expected |

Reusable generator helpers (random name pools, integer draws, a tiny constraint-solver for
uniqueness checks) live alongside the suite.

## What changes, exactly

- `harness/core.py`: `Task.parametrize` field; `make_environment(task, seed=0)`;
  `RunResult.seed` field.
- `harness/runner.py`: `run_task(..., base_seed=0)` derives the seed, passes it to
  `make_environment`, records it on the result; `run_study(..., base_seed=0)` threads it.
- `run_eval.py`: `--seed` flag → `base_seed`.
- `harness/report.py`: unchanged aggregation, but add a one-line note in the report that
  tasks are procedurally generated and pass@k spans distinct instances.
- `tasks/suite.py`: 6 families gain `parametrize`; their `setup`/`check` read
  `env.scratch["params"]`; generator helpers added.
- `CLAUDE.md`: document `parametrize`, the seed flow, the generator contract, and the
  pass@k-over-distinct-instances semantics.

## Testing (TDD, fully offline)

- Core: `make_environment(task, seed)` populates `env.scratch["params"]`; same seed →
  identical params; different `run_index` → different params; static tasks get `{}`.
- Runner: `RunResult.seed` recorded; CSV has the `seed` column; two repeats of a procedural
  task get different seeds.
- Per family: a **uniqueness/well-formedness property test** over ≥200 seeds, plus a
  round-trip test (regenerate from a recorded seed → identical instance → grader agrees).
- Full suite stays green and fast (property tests over 200 seeds must remain <1s each — keep
  generators cheap).

## pass@k semantics (call this out in docs + the helloai artifact)

With k distinct instances, pass@k is "solved at least one of k different instances" and
pass@1 is the mean over instances. This is a *stronger*, harder-to-fluke measure than k
repeats of one fixed instance. The future `eval-summary.json` will mark the suite as
procedurally generated so helloai can label results "contamination-resistant."

## Out of scope (YAGNI)

- Converting tier-1 tasks or the coding `c_fix`/`c_transform` (do after the pattern proves out).
- The other three roadmap items (agent-safety, model-as-a-tool, real-code-in-container) —
  each gets its own spec.
- The `eval-summary.json` emitter, token split, and pass@k report change from the helloai
  thread — tracked separately; not part of this spec.

## Files touched

`harness/core.py`, `harness/runner.py`, `run_eval.py`, `harness/report.py`,
`tasks/suite.py`, `tests/*`, `CLAUDE.md`.
