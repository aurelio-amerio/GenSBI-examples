"""Compare notebooks' content (py:percent text, ignoring cell ids) between the working tree and a git ref."""

import difflib
import subprocess
import sys

import jupytext

ref, paths = sys.argv[1], sys.argv[2:]
changed = 0
for path in paths:
    old = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
    a = jupytext.writes(jupytext.reads(old, fmt="ipynb"), fmt="py:percent").splitlines()
    b = jupytext.writes(jupytext.read(path), fmt="py:percent").splitlines()
    if a == b:
        print(f"SAME {path}")
    else:
        changed += 1
        print(f"CHANGED {path}")
        print("\n".join(difflib.unified_diff(a, b, ref, "worktree", lineterm="", n=1)))
sys.exit(1 if changed else 0)
