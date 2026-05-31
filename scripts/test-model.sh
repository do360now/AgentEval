#!/usr/bin/env bash
# Run the agentic eval for one model (or `all`) on the hard discriminator set.
#
# Claude nicknames (haiku/sonnet/opus) run via the `claude -p` proxy. ANY OTHER
# name is treated as a local Ollama model tag and run via --ollama, e.g.
# `qwen3:8b`, `gemma2:9b`. (run_python is bwrap-sandboxed, so scoring a local
# model cannot touch the live filesystem.)
#
# Usage:
#   ./scripts/test-model.sh haiku            # hard discriminator set, k=3
#   ./scripts/test-model.sh qwen3:8b         # local Ollama model, hard set
#   ./scripts/test-model.sh opus 5           # hard set, k=5
#   ./scripts/test-model.sh all              # every Claude model, hard set
#   ./scripts/test-model.sh haiku --full     # the WHOLE suite for one model
#
# Output: runs/<nick>-<timestamp>/{report.md,results.csv,trajectories/}
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A MAP=( [haiku]=claude-haiku-4-5 [sonnet]=claude-sonnet-4-6 [opus]=claude-opus-4-8 )
HARD_TASKS="h_merge_intervals h_revenue_by_region"

# --- parse args (nickname, optional numeric k, optional --full) ----------- #
NICK=""; K=3; FULL=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    ''|*[!0-9]*) [ -z "$NICK" ] && NICK="$a" ;;   # first non-numeric = nickname
    *) K="$a" ;;                                  # numeric = k
  esac
done

if [ -z "$NICK" ]; then
  echo "usage: $0 <haiku|sonnet|opus|all> [k] [--full]" >&2; exit 2
fi

run_one() {
  local nick="$1"
  local id="${MAP[$nick]:-}"
  # Claude nickname -> claude-cli proxy; anything else -> local Ollama model tag.
  local provider
  if [ -n "$id" ]; then
    provider="--claude-cli $id"
  else
    provider="--ollama $nick"
    echo ">>> (no Claude nickname '$nick' -- treating as a local Ollama model)"
  fi
  local out="runs/${nick//[:\/]/-}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$out"
  echo ">>> $nick  k=$K  $([ "$FULL" = 1 ] && echo '[full suite]' || echo '[hard set]')"
  # shellcheck disable=SC2086
  if [ "$FULL" = 1 ]; then
    python3 run_eval.py $provider --k "$K" --out "$out" --dump-trajectories
  else
    python3 run_eval.py $provider --tasks $HARD_TASKS --k "$K" \
        --out "$out" --dump-trajectories
  fi
  echo "=== report: $out/report.md ==="
  cat "$out/report.md"
}

if [ "$NICK" = "all" ]; then
  for n in opus sonnet haiku; do run_one "$n"; done
else
  run_one "$NICK"
fi
