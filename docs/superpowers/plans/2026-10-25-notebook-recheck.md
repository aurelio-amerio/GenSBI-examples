# Notebook Re-check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every one of the 25 tracked example notebooks runs end to end against current GenSBI, with fixes landed in each notebook's source of truth and all notebooks regenerated from those sources without outputs.

**Architecture:** Each notebook gets a jupytext py-percent working copy (`<name>.wip.py`) that is executed on the GPU, one at a time, from the notebook's own directory. Fixes are made in the working copy, then ported to the source of truth (template + replacement dicts, `.md`, or the `.ipynb` itself). At the end all notebooks are regenerated, diffed against the working copies, a few are executed as spot checks, and the working copies are deleted.

**Tech Stack:** Python 3.12 venv at `/home/aure/github/GenSBI/.venv` (GenSBI v0.5.1+, jupytext 1.19.1, jupyter, imageio 2.38.1), JAX on CUDA, git.

**Spec:** `docs/superpowers/specs/2026-10-25-notebook-recheck-design.md`

## Resume notes (paused 2026-10-25)

Paused to move to another machine. **Read `docs/superpowers/plans/2026-10-25-notebook-recheck-handout.md` first.** In short:

- Tasks 1–2 done (`55af9422`, `335edf7d`). Task 3 in progress: re-run from the start. Task 4 Step 1 done.
- **P1 changed:** run working copies with `scripts/recheck/run.sh <stem.wip.py>` (executes in a Jupyter kernel), never `python <stem>.wip.py`. Running it as a plain script breaks the `if colab:` blocks and spawns runaway grain `mp_prefetch` workers. Fix the hard-coded paths in `run.sh` on a new machine. `SCRATCH` below is obsolete; logs go to `.superpowers/sdd/2026-10-25-notebook-recheck/logs/`.
- There are **26** notebooks, not 25.
- Known cross-task fixes: `pipeline.sample` with a batch of conditions now raises, so use `sample_batched(key, xs, nsamples=1)[0]` (affects `two_moons_template.md`, Task 6). Large log-prob grids OOM on 12 GB, so evaluate them in chunks.
- Review Focus 3: grep for the exact placeholder keys, not `{[a-z_]*}` (false positive on `{notebook_path}`).

## Global Constraints

- Work on branch `notebook-recheck` (already created; design doc committed as `fff7927d`).
- Python: `/home/aure/github/GenSBI/.venv/bin/python`; jupytext: `/home/aure/github/GenSBI/.venv/bin/jupytext`.
- GPU runs: **one at a time, never concurrent.** The command sandbox blocks GPU access, so each run uses `dangerouslyDisableSandbox: true` (the user has agreed to the permission prompts).
- Run every working copy with `MPLBACKEND=Agg`, from the notebook's own directory.
- Working copies are named `<notebook-stem>.wip.py` next to the notebook. Nothing else may be named `*.wip.py`.
- **No retraining, no library code changes, no model config changes, no new content, no regeneration of linked static images.** Small fixes only. Anything bigger → skip and record.
- Committed checkpoints must stay byte-identical (`git status` must show no change under any `checkpoints/`).
- Committed notebooks have **no outputs**.
- Tracked `.py` files are untouched except `examples/methods_and_samplers/methods_and_samplers.py` (stale copy; may be overwritten, deleted at cleanup).
- Existing tracked images that a run overwrites are reverted after the run.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

Failure modes the spec implies that no task's run naturally exercises:

1. **Colab path cell.** `%cd GenSBI-examples/examples/...` in the Colab install cell must point at the notebook's actual directory; a wrong path means Colab users can't restore the checkpoint. (Pinned in Task 10, Step 3.)
2. **Colab badge link.** The badge URL in the first markdown cell must point at the notebook's actual path on `main`. (Pinned in Task 10, Step 3.)
3. **Template placeholder leakage.** A `{...}` placeholder left unreplaced in a generated notebook, or a per-task placeholder that only one task replaces. (Pinned in Task 10, Step 2.)
4. **Checkpoint directory writes.** A run with `train_model=False` must not create or modify anything under `checkpoints/` (orbax may create dirs on restore). (Pinned in Procedure P2, run after every notebook.)
5. **Restore-by-default flip for `NDE/flow_matching_2d_unconditional_flux1joint`.** The committed checkpoint must actually load with the tag/step the notebook asks for after the flag flip. (Pinned in Task 7, Step 2.)

---

## Procedures (referenced by tasks)

**P1 — Run a working copy.**
```bash
cd <notebook-dir>
MPLBACKEND=Agg /home/aure/github/GenSBI/.venv/bin/python <stem>.wip.py 2>&1 | tee "$SCRATCH/logs/<stem>.log"; echo "exit=${PIPESTATUS[0]}"
```
`SCRATCH` = `/tmp/claude-1000/-home-aure-github-GenSBI-examples/40207548-2f4e-4130-ae52-10a73e2d0d50/scratchpad` (create `logs/` once). Run outside the sandbox. Expected on success: `exit=0`, no Python traceback in the log, no `nan` losses, no "Checkpoint not found".

**P2 — Clean up after a run.**
```bash
cd /home/aure/github/GenSBI-examples
git status --short <notebook-dir>
```
Expected: only `?? <notebook-dir>/<stem>.wip.py` (or ` M` on it if already tracked). For anything else:
- modified tracked images/data (`.png`, `.gif`, `.pdf`, `.txt`): `git checkout -- <file>`
- new untracked files that are run artifacts: delete them
- **anything under `checkpoints/`** (modified or new): this is Review Focus 4 — revert/delete it, and record in the report why it happened and what in the notebook causes it (fix in the wip if small, e.g. a stray `save_model=True`).

**P3 — Record.** Append a row to `docs/superpowers/specs/2026-10-25-notebook-recheck-report.md`:
`| <path> | ran clean / fixed / skipped / flagged | <what was fixed, or error + suspected cause, or what looks off> |`

**P4 — Review the text.** Read every markdown cell and comment in the wip (they appear as `# %% [markdown]` blocks and `#` comments). Fix spelling/grammar/stale names in the wip. Known so far: `ocmpute` in both templates' `get_samples` comment; top-level README "CUDA 12" (Task 9).

---

### Task 1: Working copies, runner scaffolding, report skeleton

**Files:**
- Create: `scripts/recheck/make_wip.py`, `scripts/recheck/verify_regen.py` (both deleted in Task 12)
- Create: `examples/**/<stem>.wip.py` (25 files)
- Create: `docs/superpowers/specs/2026-10-25-notebook-recheck-report.md`

**Interfaces:**
- Produces: `NOTEBOOKS` list in `scripts/recheck/make_wip.py` — `list[str]` of the 25 tracked `.ipynb` paths relative to repo root (from `git ls-files '*.ipynb'`), importable by `verify_regen.py`.
- Produces: `verify_regen.py` CLI: `python scripts/recheck/verify_regen.py [paths...]` → for each notebook, converts the `.ipynb` to py:percent text with `jupytext.writes(nb, fmt="py:percent")`, compares with the `.wip.py` text after dropping the jupytext YAML header (lines between the first two `# ---` lines) and trailing whitespace per line, prints `OK <path>` or `DIFF <path>` plus a unified diff, exits 1 if any DIFF.

- [ ] **Step 1: Write `scripts/recheck/make_wip.py`**

For each path in `NOTEBOOKS`: `nb = jupytext.read(path)`; strip outputs and execution counts from code cells; `jupytext.write(nb, path.replace(".ipynb", ".wip.py"), fmt="py:percent")`. Skip if the `.wip.py` exists unless `--force`.

- [ ] **Step 2: Run it and check the count**

Run: `/home/aure/github/GenSBI/.venv/bin/python scripts/recheck/make_wip.py && git ls-files --others --exclude-standard '*.wip.py' | wc -l`
Expected: `25`

- [ ] **Step 3: Write `scripts/recheck/verify_regen.py`** per the interface above.

- [ ] **Step 4: Run verify on the fresh copies**

Run: `/home/aure/github/GenSBI/.venv/bin/python scripts/recheck/verify_regen.py`
Expected: 25 × `OK`, exit 0 (nothing has changed yet, so this proves the comparison is sound).

- [ ] **Step 5: Create the report file** with a header, the table header `| Notebook | Status | Notes |`, and a "Flagged results" section (empty).

- [ ] **Step 6: Commit**

```bash
git add scripts/recheck docs/superpowers/specs/2026-10-25-notebook-recheck-report.md 'examples/**/*.wip.py'
git commit -m "chore(recheck): working copies, regen verifier, report skeleton"
```

---

### Task 2: `getting_started/my_first_model`

**Files:**
- Modify: `examples/getting_started/my_first_model.wip.py`, then `examples/getting_started/my_first_model.md`
- Regenerate: `examples/getting_started/my_first_model.ipynb`

The `.md` is newer than the `.ipynb`. The wip was made from the `.ipynb`, so first bring it up to the `.md`.

- [ ] **Step 1: Rebuild the wip from the `.md`**

Run: `cd examples/getting_started && /home/aure/github/GenSBI/.venv/bin/jupytext --to py:percent -o my_first_model.wip.py my_first_model.md`

- [ ] **Step 2: Run (P1)** — the notebook calls `pipeline.restore_model(2)`; expected: restores, samples, log_prob plot, SBC/TARP/L-C2ST run; `exit=0`.

- [ ] **Step 3: Fix in the wip** until P1 passes; apply P4 (this is where the user's known typo lives — find it).

- [ ] **Step 4: Port fixes to `my_first_model.md`** (same cells, MyST format).

- [ ] **Step 5: Regenerate and verify**

Run: `cd examples/getting_started && /home/aure/github/GenSBI/.venv/bin/jupytext --to ipynb my_first_model.md && cd ../.. && /home/aure/github/GenSBI/.venv/bin/python scripts/recheck/verify_regen.py examples/getting_started/my_first_model.ipynb`
Expected: `OK`

- [ ] **Step 6: P2, P3, commit**

```bash
git add examples/getting_started docs/superpowers/specs/2026-10-25-notebook-recheck-report.md
git commit -m "fix(getting_started): my_first_model runs clean; regen from md"
```

---

### Task 3: `methods_and_samplers`

**Files:**
- Modify: `examples/methods_and_samplers/methods_and_samplers.wip.py`, then `methods_and_samplers.md`
- Regenerate: `methods_and_samplers.ipynb`

- [ ] **Step 1: Rebuild the wip from the `.md`** (as Task 2 Step 1, with these names). Confirm with `verify_regen.py` whether the committed `.ipynb` already matches the `.md`; if `DIFF`, the `.md` wins.
- [ ] **Step 2: Run (P1).** Three checkpoints (`flow`, `edm`, `sm`) are restored; expected `exit=0`.
- [ ] **Step 3: Fix in the wip; P4.**
- [ ] **Step 4: Port to `methods_and_samplers.md`.**
- [ ] **Step 5: Regenerate with jupytext and verify `OK`** (as Task 2 Step 5).
- [ ] **Step 6: P2, P3, commit** `fix(methods_and_samplers): runs clean; regen from md`.

---

### Task 4: Generic-template notebooks (9) — run and fix the working copies

**Files:**
- Modify: the 9 `.wip.py` under `examples/sbi-benchmarks/{bernoulli_glm,gaussian_linear,gaussian_mixture}/{flow_flux,flow_flux1joint}/` and `slcp/{flow_flux,flow_flux1joint,flow_simformer}/`

**Interfaces:**
- Produces: a list in the report, "Template fixes (generic)", of each fix as `(cell, before, after, applies-to: all | <task>)`, consumed by Task 5.

- [ ] **Step 1: Baseline the generator.** Run `scripts/make_notebook.py` in a scratch copy and confirm the reviewer's finding: `/home/aure/github/GenSBI/.venv/bin/python scripts/make_notebook.py && git status --short examples/sbi-benchmarks`
Expected: only `gaussian_mixture/flow_flux1joint/gaussian_mixture_flow_flux1joint.ipynb` modified. Then `git checkout -- examples/sbi-benchmarks` to restore it. (If more files differ, record the diff in the report before restoring — those are more hand-patches to carry into placeholders in Task 5.)
- [ ] **Step 2: For each of the 9, in this order** (cheapest first): gaussian_linear ×2, bernoulli_glm ×2, gaussian_mixture ×2, slcp ×3 — do P1, fix in the wip, P4, P2, P3. Compare the printed C2ST/diagnostics against the number in that example's `README.md`; flag if far off.
- [ ] **Step 3: Write the "Template fixes (generic)" list** in the report, de-duplicated across the 9. Every fix must be marked `all` or task-specific.
- [ ] **Step 4: Commit** `fix(recheck): generic-template working copies run clean`.

---

### Task 5: Generic template — port fixes and add gaussian_mixture placeholders

**Files:**
- Modify: `scripts/notebook_template.md` (lines ~189 `get_samples`, ~215 `plot_marginals`, plus Task 4 fixes)
- Modify: `scripts/make_notebook.py` (the 9 replacement dicts)
- Regenerate: the 9 `.ipynb`

**Interfaces:**
- Produces: two new placeholders in `notebook_template.md`:
  - `{use_ema_default}` in `def get_samples(idx, nsamples=10_000, use_ema={use_ema_default}, key=None):`
  - `{plot_marginals_extra}` in `plot_marginals(samples[-1,...,0], backend="seaborn", gridsize=50{plot_marginals_extra})`
- Every replacement dict gains `"{use_ema_default}": "False"` and `"{plot_marginals_extra}": ""`, except `replacements_gaussian_mixture_flow_flux1joint`: `"True"` and `", true_param=true_param.reshape(-1), range=((-9,-7), (4,6))"`.

- [ ] **Step 1: Add the placeholders** to the template and all 9 dicts as above.
- [ ] **Step 2: Regenerate and check that nothing else changed**
Run: `/home/aure/github/GenSBI/.venv/bin/python scripts/make_notebook.py && git status --short examples/sbi-benchmarks`
Expected: no modified `.ipynb` (the gaussian_mixture joint one now regenerates to its committed content).
- [ ] **Step 3: Port the Task 4 "all" fixes** into the template; task-specific ones become further placeholders only if they can't be expressed as `all` (prefer rewording so one template line works for all tasks).
- [ ] **Step 4: Regenerate and verify against the working copies**
Run: `/home/aure/github/GenSBI/.venv/bin/python scripts/make_notebook.py && /home/aure/github/GenSBI/.venv/bin/python scripts/recheck/verify_regen.py examples/sbi-benchmarks/{bernoulli_glm,gaussian_linear,gaussian_mixture,slcp}/*/*.ipynb`
Expected: 9 × `OK`. If a `DIFF` shows, the template fix differs from what ran — fix the template, not the wip.
- [ ] **Step 5: Placeholder leak check** (Review Focus 3)
Run: `grep -l '{[a-z_]*}' examples/sbi-benchmarks/*/*/*.ipynb | grep -v two_moons`
Expected: no output.
- [ ] **Step 6: Commit** `fix(template): generic template fixes; gaussian_mixture tuning via placeholders`.

---

### Task 6: two_moons template notebooks (6) — run, fix, port, regenerate

**Files:**
- Modify: the 6 `.wip.py` under `examples/sbi-benchmarks/two_moons/{flow,diffusion}_{flux,flux1joint,simformer}/`
- Modify: `scripts/two_moons_template.md`, `scripts/make_notebook_two_moons.py` (only if a task-specific placeholder is needed)
- Regenerate: the 6 `.ipynb`

Note: these write `animated_plot_samples.gif` (tracked) — P2 reverts it.

- [ ] **Step 1: Baseline the generator** as Task 4 Step 1 with `scripts/make_notebook_two_moons.py`; expected: no modified `.ipynb`.
- [ ] **Step 2: For each of the 6** (flow_flux first): P1, fix, P4, P2, P3; compare C2ST/L-C2ST output to the README.
- [ ] **Step 3: Port de-duplicated fixes** to `two_moons_template.md` (`all` fixes; task-specific → placeholder in `make_notebook_two_moons.py`).
- [ ] **Step 4: Regenerate and verify**
Run: `/home/aure/github/GenSBI/.venv/bin/python scripts/make_notebook_two_moons.py && /home/aure/github/GenSBI/.venv/bin/python scripts/recheck/verify_regen.py examples/sbi-benchmarks/two_moons/*/*.ipynb`
Expected: 6 × `OK`; `grep -l '{[a-z_]*}' examples/sbi-benchmarks/two_moons/*/*.ipynb` → no output.
- [ ] **Step 5: Commit** `fix(two_moons): template fixes; regen`.

---

### Task 7: NDE notebooks (5) — run, fix, strip outputs, link PNGs

**Files:**
- Modify: `examples/NDE/{diffusion_EDM_2d_unconditional,diffusion_SM_2d_unconditional,flow_matching_2d_unconditional,flow_matching_2d_unconditional_flux1joint,gensbi_text}.wip.py`, then the matching `.ipynb` (regenerated from the wip)

Existing PNGs on disk (link targets): `examples/NDE/{diffusion_EDM,diffusion_SM,flow_matching}_unconditional{,_forward,_reverse,_trajectories}.png`; `gensbi_text` has only GIFs (already linked).

- [ ] **Step 1: Flip the flags** in `flow_matching_2d_unconditional_flux1joint.wip.py`: `restore_model = True`, `train_model = False`.
- [ ] **Step 2: Run that notebook first (P1)** — Review Focus 5: expected the committed checkpoint under `examples/NDE/checkpoints/flow_matching_2d_example_flux1joint` restores at the tag the notebook requests and sampling succeeds. If restore fails for a non-trivial reason: skip, record, and leave the committed flags as they are (don't ship a notebook that can't restore).
- [ ] **Step 3: Run the other four (P1)**, fix, P4, P2 (these overwrite the tracked PNGs/PDF — revert), P3.
- [ ] **Step 4: Link figures.** In each wip, after each code cell whose `savefig` writes one of the existing PNGs above, add a markdown cell `![<stem>](<png filename>)` — or an `<img>` tag matching the style already used in `gensbi_text` (raw.githubusercontent URL on `main`); use the raw URL style so it renders on GitHub and Colab. Skip where no PNG exists.
- [ ] **Step 5: Regenerate each `.ipynb` from its wip**
Run: `cd examples/NDE && for s in diffusion_EDM_2d_unconditional diffusion_SM_2d_unconditional flow_matching_2d_unconditional flow_matching_2d_unconditional_flux1joint gensbi_text; do /home/aure/github/GenSBI/.venv/bin/jupytext --to ipynb -o $s.ipynb $s.wip.py; done`
Then: `python -c "import json,sys; [sys.exit(f'outputs in {p}') for p in sys.argv[1:] for c in json.load(open(p))['cells'] if c.get('outputs')]" examples/NDE/*.ipynb` → expected: no output. And `verify_regen.py examples/NDE/*.ipynb` → 5 × `OK`.
- [ ] **Step 6: Commit** `fix(NDE): notebooks run clean, restore by default, outputs stripped, figures linked`.

---

### Task 8: `.ipynb`-only sbi-benchmarks (4): gw, lensing, tarflow_NLE, maf_NLE

**Files:**
- Modify: `examples/sbi-benchmarks/gravitational_waves/gw_example.wip.py`, `lensing/lensing_example.wip.py`, `slcp/tarflow_NLE/slcp_tarflow_nle.wip.py`, `two_moons/maf_NLE/two_moons_maf_nle.wip.py`, then the matching `.ipynb`

`gw_example` and `lensing_example` use `load_dataset` (Hugging Face) — the run needs network; add `huggingface.co` and `*.hf.co` / `cdn-lfs*.huggingface.co` to `allowed_domains` (or rely on the sandbox being disabled for the GPU run).

- [ ] **Step 1: Known fix** in `lensing_example.wip.py`: `savefig("img/lensing.png", ...)` → `"imgs/lensing.png"`.
- [ ] **Step 2: Run each (P1)** in order maf_NLE, tarflow_NLE, lensing, gw; fix, P4, P2 (gw/lensing write under `imgs/` — revert tracked, delete new), P3.
- [ ] **Step 3: Regenerate each `.ipynb` from its wip** with `jupytext --to ipynb -o <stem>.ipynb <stem>.wip.py`; check no outputs (same one-liner as Task 7 Step 5) and `verify_regen.py` → 4 × `OK`.
- [ ] **Step 4: Commit** `fix(sbi-benchmarks): gw, lensing, tarflow_NLE, maf_NLE run clean; outputs stripped`.

---

### Task 9: READMEs

**Files:**
- Modify: `README.md` (lines 27, 39: "CUDA 12" → "CUDA 13"), per-example `README.md` files only where they describe the notebook and are wrong.

- [ ] **Step 1: Fix the top-level README** CUDA labels; read the rest for typos and stale notebook links (every Colab link must point at an existing `.ipynb`: `grep -o 'blob/main/[^)]*\.ipynb' README.md | sed 's|blob/main/||' | xargs ls` → no "No such file").
- [ ] **Step 2: Skim the 15 per-example READMEs** for typos; fix. Don't touch C2ST numbers.
- [ ] **Step 3: Commit** `docs: README typo fixes`.

---

### Task 10: Full regeneration and cross-checks

- [ ] **Step 1: Regenerate everything from sources**
Run: `/home/aure/github/GenSBI/.venv/bin/python scripts/make_notebook.py && /home/aure/github/GenSBI/.venv/bin/python scripts/make_notebook_two_moons.py && (cd examples/getting_started && jupytext --to ipynb my_first_model.md) && (cd examples/methods_and_samplers && jupytext --to ipynb methods_and_samplers.md)` (use the venv's jupytext).
Expected: `git status --short` shows no `.ipynb` modified (they were already regenerated in Tasks 2–8).
- [ ] **Step 2: Placeholder leak + outputs check across all 25** (Review Focus 3)
Run: `grep -l '{[a-z_]*}' $(git ls-files '*.ipynb')` → no output; outputs one-liner over `$(git ls-files '*.ipynb')` → no output.
- [ ] **Step 3: Colab path and badge check across all 25** (Review Focus 1, 2)
For each notebook: the `%cd GenSBI-examples/<dir>` in the install cell must equal the notebook's directory, and the badge URL must contain the notebook's path. Script it (`scripts/recheck/check_colab.py`, deleted in Task 12): parse both from the first cells, compare with the path, print `OK`/`BAD`. Expected: 25 × `OK`.
- [ ] **Step 4: Verify** `verify_regen.py` over all 25 → 25 × `OK`.
- [ ] **Step 5: Commit** `chore(recheck): regen all notebooks; colab checks`.

---

### Task 11: Spot-check runs of regenerated notebooks

Run the real `.ipynb` (not the wip) for: `gaussian_linear/flow_flux`, `two_moons/flow_flux`, `getting_started/my_first_model`, `NDE/flow_matching_2d_unconditional`. Outputs go to the scratchpad so nothing lands in the repo.

- [ ] **Step 1: For each, one at a time (GPU, outside sandbox)**
Run: `cd <dir> && MPLBACKEND=Agg /home/aure/github/GenSBI/.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output-dir "$SCRATCH/spot" <stem>.ipynb`
Expected: exit 0, executed copy written under `$SCRATCH/spot/`. Then P2.
- [ ] **Step 2: Record** the four results in the report under "Spot checks".
- [ ] **Step 3: Commit** `docs(recheck): spot-check results`.

---

### Task 12: Cleanup and final report

- [ ] **Step 1: Delete working copies and helpers**
Run: `git rm -q $(git ls-files '*.wip.py') examples/methods_and_samplers/methods_and_samplers.py && git rm -rq scripts/recheck`
Expected: `git ls-files '*.wip.py' | wc -l` → `0`; `ls scripts/recheck` → no such directory.
- [ ] **Step 2: Final tree checks**
Run: `git status --short` → clean after staging; `git diff main --stat -- '**/checkpoints/**'` → empty (checkpoints untouched); outputs one-liner over all `.ipynb` → no output.
- [ ] **Step 3: Finish the report:** per-notebook table complete (25 rows), "Flagged results" and "Skipped" sections filled (or "none"), list of source-of-truth files changed.
- [ ] **Step 4: Commit** `chore(recheck): remove working copies; final report`.
- [ ] **Step 5: Hand off** — summarise the report to the user; the branch is ready for them to merge.
