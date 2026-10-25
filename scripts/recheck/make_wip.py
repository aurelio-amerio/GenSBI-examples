"""Create jupytext py:percent working copies (<stem>.wip.py) of every tracked notebook."""

import subprocess
import sys
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS = subprocess.run(
    ["git", "ls-files", "*.ipynb"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.split()


def wip_path(path):
    return ROOT / path.replace(".ipynb", ".wip.py")


def main(force=False):
    for path in NOTEBOOKS:
        out = wip_path(path)
        if out.exists() and not force:
            print(f"skip {out.relative_to(ROOT)}")
            continue
        nb = jupytext.read(ROOT / path)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        jupytext.write(nb, out, fmt="py:percent")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
