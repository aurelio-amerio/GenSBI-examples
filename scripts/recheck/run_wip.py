"""Execute a .wip.py working copy in a Jupyter kernel, from the notebook's own directory.

A kernel (not a plain script) is used because grain's `mp_prefetch` spawns workers
that re-import `__main__`, which only behaves as in the notebook inside a kernel.
The executed notebook is written to the given output path for inspection.
"""

import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

wip = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
nb = jupytext.read(wip)
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
