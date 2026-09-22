#!/usr/bin/env bash
# One command for a full run: preview, benchmark, report.
#
#   ./run.sh            200 posts, all three arms, the two cheap models
#   ./run.sh 50         smaller and cheaper
#   BUDGET=0.25 ./run.sh 50
#   N=20 SKIP_DEMO=1 ./run.sh
#
# A hard spend cap is on by default (BUDGET, $1.00). The run stops the instant
# it is reached and whatever was measured is still written and reportable.
#
# Everything is kept: the raw per-request log in runs/, the rendered tables in
# docs/results.md, the chart in docs/reliability.svg. Commit all three and the
# README's numbers are reproducible by anyone.
set -euo pipefail

N="${1:-${N:-200}}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="runs/${STAMP}.jsonl"

PY="${PYTHON:-python3}"
BUDGET="${BUDGET:-1.00}"

if [[ -z "${OPENROUTER_API_KEY:-}" && ! -f .env ]]; then
  echo "No API key. Export OPENROUTER_API_KEY, or put it in .env (gitignored) as:" >&2
  echo "  OPENROUTER_API_KEY=sk-or-v1-..." >&2
  echo "To read posts without a key:  python -m jevbench.demo --read -n 5" >&2
  exit 1
fi

echo "==> Projected cost"
"$PY" -m jevbench.bench --estimate -n "$N" --budget "$BUDGET"

if [[ -z "${SKIP_DEMO:-}" ]]; then
  echo "==> Sanity check: judging 2 posts before spending anything"
  "$PY" -m jevbench.demo -n 2
  echo
  read -r -p "Look right? Continue with ${N} posts? [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "Stopped."; exit 0; }
fi

echo "==> Benchmarking ${N} posts (hard cap \$${BUDGET})"
"$PY" -m jevbench.bench -n "$N" --out "$RUN" --budget "$BUDGET"

echo "==> Rendering report"
"$PY" -m jevbench.report "$RUN" --svg docs/reliability.svg --md docs/results.md

echo
echo "Kept:"
echo "  $RUN                 raw per-request log"
echo "  docs/results.md      rendered tables"
echo "  docs/reliability.svg calibration chart"
echo
echo "Paste docs/results.md into the README's Results section, or link it."
