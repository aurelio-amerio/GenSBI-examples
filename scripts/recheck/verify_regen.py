"""Check that each notebook converts to the same py:percent text as its .wip.py working copy."""

import difflib
import sys
from pathlib import Path

import jupytext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_wip import NOTEBOOKS, ROOT, wip_path  # noqa: E402


def normalize(text):
    lines = text.splitlines()
    if lines and lines[0] == "# ---":
        end = lines.index("# ---", 1)
        lines = lines[end + 1 :]
    lines = [line.rstrip() for line in lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def main(paths):
    failed = False
    for path in paths:
        rel = str(Path(path).resolve().relative_to(ROOT))
        nb = jupytext.read(ROOT / rel)
        got = normalize(jupytext.writes(nb, fmt="py:percent"))
        want = normalize(wip_path(rel).read_text())
        if got == want:
            print(f"OK {rel}")
        else:
            failed = True
            print(f"DIFF {rel}")
            sys.stdout.writelines(
                line + "\n"
                for line in difflib.unified_diff(want, got, "wip", "ipynb", lineterm="")
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or NOTEBOOKS))
