---
description: Run the agentic eval for one model (Claude nick or local Ollama tag, or `all`) on the hard discriminator set
argument-hint: haiku | sonnet | opus | all | <ollama-tag e.g. qwen3:8b>  [k]  [--full]
allowed-tools: Bash(./scripts/test-model.sh:*), Read, Glob
---

Run the agentic eval for: `$ARGUMENTS`

The eval runs ENTIRELY inside `./scripts/test-model.sh`. Your only job is to invoke it and
relay the numbers. Do NOT do the eval yourself.

Steps:
1. Run exactly: `./scripts/test-model.sh $ARGUMENTS`
   (Claude nicknames haiku/sonnet/opus use the `claude -p` proxy; any other name is a local
   Ollama model tag. `--full` runs the whole suite; a bare number sets k.)
2. When it finishes, read the printed `report.md` and report **pass@1 / pass@k per task** for
   the model(s) just run.
3. If `runs/` already has recent reports for other models, give a short side-by-side
   comparison so the spread is visible.

HARD GUARDRAILS — do not violate:
- If the script exits non-zero, errors, or the model/tag is unsupported, **STOP and report the
  exact error verbatim.** Do NOT improvise, do NOT retry with a different command, and do NOT
  create, edit, or write ANY files to work around it.
- Never run code, write files, or take any action outside invoking the script and reading its
  output. The harness is sandboxed on purpose; you are not.
- Keep the summary tight and limited to what the numbers show.
