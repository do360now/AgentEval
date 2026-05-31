---
description: Run the agentic eval for one Claude model (or `all`) on the hard discriminator set
argument-hint: haiku | sonnet | opus | all  [k]  [--full]
allowed-tools: Bash(./scripts/test-model.sh:*), Bash(cat:*), Read, Glob
---

Run the agentic eval for the requested model(s): `$ARGUMENTS`

Steps:
1. Execute `./scripts/test-model.sh $ARGUMENTS`. This runs the harness through the
   `claude -p` proxy on the **hard discriminator set** (`h_merge_intervals`,
   `h_revenue_by_region`) at k=3 by default, writing to `runs/<nick>-<timestamp>/`.
   (`--full` runs the whole suite; a bare number sets k.)
2. When it finishes, read the printed `report.md` and report **pass@1 and pass@k per task**
   for the model(s) just run.
3. If `runs/` contains recent reports for *other* models, build a short side-by-side
   comparison so the spread between models is visible (this is the whole point of the hard
   set — show where a smaller model falls off).
4. Note the run directory so the user can inspect saved trajectories.

Keep the summary tight. The outcome score is deterministic; do not editorialize beyond what
the numbers show.
