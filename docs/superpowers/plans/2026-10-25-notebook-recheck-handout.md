# Notebook re-check: handout (paused 2026-10-25)

**Superseded (2026-10-26):** the re-check is complete; see `docs/superpowers/specs/2026-10-25-notebook-recheck-report.md`.

Plan: `docs/superpowers/plans/2026-10-25-notebook-recheck.md`
Spec: `docs/superpowers/specs/2026-10-25-notebook-recheck-design.md`
Report (per-notebook results so far): `docs/superpowers/specs/2026-10-25-notebook-recheck-report.md`
Branch: `notebook-recheck`

Work paused on purpose so it can resume on a different machine. Nothing is pushed yet.

## Status

| Task | State | Commits |
|---|---|---|
| 1. Working copies, verifier, report skeleton | **done** | `55af9422` |
| 2. `getting_started/my_first_model` | **done**: runs clean, `.md` fixed, `.ipynb` regenerated, verify OK | `335edf7d` |
| 3. `methods_and_samplers` | **in progress**: chunked FM log-prob in wip and `.md` (committed as WIP). A run got through all flow matching and EDM cells (35 code cells, no errors) before it was stopped by hand at the score-matching sampling cell. **Must be re-run from the start.** | WIP commit (see `git log`) |
| 4. Generic template, 9 notebooks | Step 1 (generator baseline) done; text issues noted (below); no runs yet | — |
| 5–12 | not started | — |

## How runs work now (differs from the plan's P1)

Use `scripts/recheck/run.sh <path/to/stem.wip.py> [tail-lines]`, **not** `python stem.wip.py`.

- It runs `scripts/recheck/run_wip.py`, which reads the wip with jupytext and executes it **in a Jupyter kernel** (nbclient), from the notebook's own directory, with `MPLBACKEND=Agg`.
- Why: (a) jupytext comments out magics, so `if colab:` blocks become empty and the `.py` won't parse; (b) grain's `.mp_prefetch()` starts spawn workers that re-import `__main__`. In a plain script that re-runs the whole notebook in every worker. It happened once: 9 runaway processes that were hard to kill.
- Output: log and executed notebook go to `.superpowers/sdd/2026-10-25-notebook-recheck/logs/` (git-ignored). `scripts/recheck/errors.py <executed.ipynb>` prints the failing cell and traceback.
- `run.sh` and `run_wip.py` hard-code `/home/aure/github/GenSBI-examples` and `/home/aure/github/GenSBI/.venv`. **Edit `ROOT` and the python path in `run.sh` on the new machine.**
- The process exit code of `run.sh` is not meaningful; read the `exit=` line it prints (0 = all cells ran).
- GPU runs still need to be outside the command sandbox, one at a time.
- `savefig` calls in some notebooks overwrite tracked PNGs (e.g. all of `methods_and_samplers/*.png`). Run `git checkout -- <dir>/*.png` after each run (P2).

Other helpers in `scripts/recheck/` (all deleted in Task 12): `make_wip.py`, `verify_regen.py` (plan's verifier), `content_diff.py <ref> <ipynb...>` (compares notebook content against a git ref, ignoring cell ids).

## Findings that affect later tasks

1. **`pipeline.sample` on a batch of conditions now raises** (`x_o has a leading batch axis of size N > 1 ... Use sample_batched()`). The L-C2ST cells drew one sample per condition this way. Fix used in my_first_model: `pipeline.sample_batched(key, xs, nsamples=1)[0]` (shape `(N, dim, ch)`). **The same bug is in `scripts/two_moons_template.md` (~line 402)**, so all 6 two_moons notebooks need this (Task 6).
2. **Exact log-prob on a ~132k-point grid OOMs on a 12 GB GPU** (RTX 4070: one `diffeqsolve` asks for 19.45 GiB). Fixed by evaluating in 10k-point chunks with `jnp.concatenate([...])`. If the new machine has a much bigger GPU this won't reproduce, but keep the chunking so the notebooks run on ordinary GPUs. Watch for the same pattern in the NDE notebooks (Task 7).
3. **Regenerating notebooks gives every cell a new random id**, so git diffs are noisy even when content is unchanged. `jupytext --update --to ipynb x.md` keeps the ids for `.md`-sourced notebooks. Use `content_diff.py` to check that content really is unchanged.
4. **Review Focus 3 (placeholder leak check):** the plan's `grep '{[a-z_]*}'` also matches the legitimate f-string `{notebook_path}` in the generic template. Grep instead for the exact placeholder keys defined in `scripts/make_notebook*.py`.
5. **Task 4, Step 1 baseline confirmed:** `make_notebook.py` regenerates all 9 notebooks with identical content except `gaussian_mixture/flow_flux1joint` (`use_ema=True`, `plot_marginals` range), as the spec says. `make_notebook_two_moons.py` regenerates all 6 identically.
6. **Generic-template text issues noted for Task 5** (`scripts/notebook_template.md`): `ocmpute` (l.188); `excercise` (l.231); "evaluate the trained Simformer model" (l.179, wrong for Flux1 notebooks); "Edge masks and marginalization functions are used…" (l.133, Simformer-only wording); TOC row 5 promises "Visualize loss … compute log prob", which the notebook doesn't have (l.31); `# plot the 2D posterior` for tasks with more than 2 dimensions (l.109).
7. The repo has **26** tracked notebooks, not 25 (the spec's own inventory adds up to 26). Expect 26 × OK wherever the plan says 25.
8. `my_first_model` diagnostics for the record: SBC KS p-values 0.18 / 0.70 / 0.37; TARP ATC −0.21 with KS p = 1.0. Not flagged.

## Rulings made so far (decisions taken without asking)

- 26 notebooks, not 25. Cost if wrong: none.
- Commit trailer is `Co-Authored-By: Claude Opus 5.5`, not the plan's Fable 5.1 line (it reflects the model actually used). Cost if wrong: trivial amend.
- The spec's `imageio` step was already done in GenSBI (`c6807dd8`, imageio 2.38.1 installed), so nothing to do.
- In my_first_model, 6 `savefig` lines labelled "uncomment to save the figure" were active; they are now commented out to match their own comment. Cost if wrong: re-enable 6 lines.
- Runs use the kernel-based runner instead of `python wip.py` (see above). Cost if wrong: none.
- Log-prob grids are evaluated in chunks (my_first_model, methods_and_samplers FM). Cost if wrong: a slightly longer cell.
- The leak check uses exact placeholder keys. Cost if wrong: none.

## To resume

1. On the new machine: `git fetch && git checkout notebook-recheck` (after this branch is pushed). Check that the venv has GenSBI ≥ 0.5.1, jupytext 1.19.1, nbclient, ipykernel and imageio. Edit the paths in `scripts/recheck/run.sh`.
2. Recreate the ledger if you use superpowers executing-plans: `.superpowers/` is git-ignored, so the ledger didn't travel. Its content is reproduced in this handout; Tasks 1 and 2 are complete.
3. Resume at **Task 3, Step 2**: `scripts/recheck/run.sh examples/methods_and_samplers/methods_and_samplers.wip.py`, then fix, P4 (text read: nothing found), port to `.md` (chunk fix already ported), `jupytext --update --to ipynb methods_and_samplers.md`, verify, P2 (revert PNGs), P3, commit.
4. Continue with Task 4 onward as in the plan.
