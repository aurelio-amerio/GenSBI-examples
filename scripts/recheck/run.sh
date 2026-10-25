#!/bin/bash
# usage: run.sh <path/to/stem.wip.py> [tail-lines] — executes the wip in a kernel; executed nb + log go to the sdd workspace
# CPU-only node: JAX forced to cpu, kernel and its workers pinned to 6 cores (user cap: 8 cores total).
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VENV=/lhome/ific/a/aamerio/data/github/GenSBI/.venv
L=$ROOT/.superpowers/sdd/2026-10-25-notebook-recheck/logs
mkdir -p "$L"
f=$(realpath "$1"); stem=$(basename "$f" .wip.py)
export PATH=$VENV/bin:$PATH  # the user-level python3 kernelspec launches plain `python`
export MPLBACKEND=Agg JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
taskset -c 0-5 "$VENV/bin/python" "$ROOT/scripts/recheck/run_wip.py" "$f" "$L/$stem.executed.ipynb" > "$L/$stem.log" 2>&1
echo "exit=$?"
grep -a -v -i 'warning\|fontconfig\|matplotlib created' "$L/$stem.log" | tail -${2:-25}
"$VENV/bin/python" "$ROOT/scripts/recheck/errors.py" "$L/$stem.executed.ipynb" 2>/dev/null
