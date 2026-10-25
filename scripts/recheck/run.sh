#!/bin/bash
# usage: run.sh <path/to/stem.wip.py> [tail-lines] — executes the wip in a kernel; executed nb + log go to the sdd workspace
ROOT=/home/aure/github/GenSBI-examples
L=$ROOT/.superpowers/sdd/2026-10-25-notebook-recheck/logs
f=$(realpath "$1"); stem=$(basename "$f" .wip.py)
MPLBACKEND=Agg /home/aure/github/GenSBI/.venv/bin/python "$ROOT/scripts/recheck/run_wip.py" "$f" "$L/$stem.executed.ipynb" > "$L/$stem.log" 2>&1
echo "exit=$?"
grep -a -v -i 'warning\|fontconfig\|matplotlib created' "$L/$stem.log" | tail -${2:-25}
/home/aure/github/GenSBI/.venv/bin/python "$ROOT/scripts/recheck/errors.py" "$L/$stem.executed.ipynb" 2>/dev/null
