#!/usr/bin/env bash
# Run the agentic eval for one Claude model (or `all`) via the claude-cli proxy.
#
# Usage:
#   ./scripts/test-model.sh haiku            # hard discriminator set, k=3
#   ./scripts/test-model.sh opus 5           # hard set, k=5
#   ./scripts/test-model.sh all              # every model, hard set
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
  if [ -z "$id" ]; then
    echo "unknown model '$nick' (valid: ${!MAP[*]} all)" >&2; exit 2
  fi
  local out="runs/${nick}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$out"
  echo ">>> $nick ($id)  k=$K  $([ "$FULL" = 1 ] && echo '[full suite]' || echo '[hard set]')"
  if [ "$FULL" = 1 ]; then
    python3 run_eval.py --claude-cli "$id" --k "$K" --out "$out" --dump-trajectories
  else
    # shellcheck disable=SC2086
    python3 run_eval.py --claude-cli "$id" --tasks $HARD_TASKS --k "$K" \
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
