"""Print the failing cell and traceback tail of an executed notebook."""

import re
import sys

import nbformat

nb = nbformat.read(sys.argv[1], 4)
for cell in nb.cells:
    for out in cell.get("outputs", []):
        if out.output_type == "error":
            print(cell.source[:600])
            print(out.ename, out.evalue[:800])
            print(re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(out.traceback))[-2000:])
