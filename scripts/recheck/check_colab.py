"""Check Colab %cd path and badge URL of every tracked notebook (deleted in Task 12)."""
import json, os, re, subprocess

BASE = "https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/"
files = subprocess.check_output(["git", "ls-files", "*.ipynb"], text=True).split()
n_bad = 0
for p in files:
    cells = json.load(open(p))["cells"]
    d = os.path.dirname(p)
    md = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    badges = re.findall(r"https://colab\.research\.google\.com/github/[^\s)\"']+", md)
    cds = re.findall(r"%cd\s+GenSBI-examples/(\S+)", code)
    why = []
    if not cds:
        why.append("no %cd GenSBI-examples/<dir>")
    for c in cds:
        if c.rstrip("/") != d:
            why.append(f"%cd {c} != {d}")
    if badges:
        if BASE + p not in badges:
            why.append(f"badge {badges} != {BASE + p}")
    if why:
        print(f"BAD {p}: " + "; ".join(why)); n_bad += 1
    elif not badges:
        print(f"NONE {p}: no Colab badge (%cd OK)")
    else:
        print(f"OK {p}")
print(f"{len(files)} notebooks, {n_bad} BAD")
