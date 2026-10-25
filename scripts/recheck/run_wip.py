"""Execute a .wip.py working copy in a Jupyter kernel, from the notebook's own directory.

A kernel (not a plain script) is used because grain's `mp_prefetch` spawns workers
that re-import `__main__`, which only behaves as in the notebook inside a kernel.
The executed notebook is written to the given output path for inspection.

CPU-only node: before execution, the in-memory copy (never the wip on disk) is patched:
`JAX_PLATFORMS = "cuda"` becomes "cpu", and the exact-substring replacements in
`shrink/<stem>.json` (a list of [old, new] pairs, e.g. smaller `nsamples`) are applied
to every code cell. Each `old` must occur at least once, so a stale shrink file fails loudly.
"""

import json
import re
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

CUDA = re.compile(r"""(JAX_PLATFORMS["']\]\s*=\s*)(["'])cuda\2""")

wip = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
nb = jupytext.read(wip)

stem = wip.name.removesuffix(".wip.py")
shrink_file = Path(__file__).parent / "shrink" / f"{stem}.json"
pairs = json.loads(shrink_file.read_text()) if shrink_file.exists() else []
hits = [0] * len(pairs)
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    cell.source = CUDA.sub(r"\1\2cpu\2", cell.source)
    for i, (old, new) in enumerate(pairs):
        hits[i] += cell.source.count(old)
        cell.source = cell.source.replace(old, new)
missing = [old for (old, _), n in zip(pairs, hits) if n == 0]
if missing:
    sys.exit(f"shrink: not found in {wip.name}: {missing}")
for (old, new), n in zip(pairs, hits):
    print(f"shrink: {old!r} -> {new!r} ({n}x)")

client = NotebookClient(
    nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(wip.parent)}}
)
try:
    client.execute()
    status = 0
except CellExecutionError as e:
    print(e)
    status = 1
finally:
    nbformat.write(nb, out)
sys.exit(status)
