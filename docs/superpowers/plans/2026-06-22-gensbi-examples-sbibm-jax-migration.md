# GenSBI-examples → sbibm-jax TaskDataset Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GenSBI-examples load all benchmark task data through `sbibm_jax.data.TaskDataset` and delete the local `gensbi_examples` package entirely.

**Architecture:** The 15 vector-task notebooks are *generated* from two MyST-Markdown templates, so they are migrated by editing the templates + regenerating; the two image/time-series notebooks (gravitational waves, lensing) and four `.py` scripts are edited by hand. The old `gensbi_examples.tasks` API maps almost 1:1 onto `TaskDataset` (see Global Constraints). After all consumers are migrated, the package, its tests, the obsolete stats script, and the related CI/README references are removed.

**Tech Stack:** Python ≥3.11, JAX 0.9.x, `sbibm-jax[loader]` (grain + datasets + huggingface_hub), jupytext (notebook generation), uv (env/deps).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-22-gensbi-examples-sbibm-jax-migration-design.md`.
- **Repo edited:** `/lhome/ific/a/aamerio/data/github/GenSBI-examples` (this repo). Source dependency lives at `/lustre/ific.uv.es/ml/ific088/github/sbibm-jax`.
- **Data repo:** read from `TaskDataset`'s built-in default (test repo `aurelio-amerio/SBI-benchmarks-test`). **Never pass a `repo=` argument.**
- **Train size:** preserve the old 100k subsample. In templates use `num_samples=nsamples` (the existing `nsamples = int(1e5)` variable); in `train_sbi_model.py` use `num_samples=100_000`; keep the explicit `dataset_size` in `train_sbi_model_sbibm.py`.
- **Dependency (committed):** `sbibm-jax[loader]` from **PyPI**. Do **not** commit a `[tool.uv.sources]` path entry. (For local verification before the package is on PyPI, install it editable out-of-band — see Task 1.)
- **API mapping (old → new):** `get_task(name, …)`/`TwoMoons(…)`/`GravitationalWaves()`/`GravitationalLensing()` → `TaskDataset(name, …)`; `dim_obs`→`dim_theta`; `dim_cond`→`dim_x`; `dim_joint` unchanged; `get_train_dataset`→`get_train_loader`; `get_val_dataset`→`get_val_loader`; `nsamples=`→`num_samples=`; `normalize_data=True`→`normalize=True`; `normalize_cond`→`normalize_x`; `unnormalize_obs`→`unnormalize_theta`; `get_reference`/`get_true_parameters`/`dataset["test"]`/`df_train`/`df_val`/`df_test` unchanged.
- **Names that DO NOT change:** the local variables `dim_obs`/`dim_cond` and the GenSBI model arguments `dim_obs=`/`dim_cond=` stay — only their right-hand side changes (`= task.dim_theta` / `= task.dim_x`).
- **Lensing:** the example directory stays `examples/sbi-benchmarks/lensing/`; only the in-code task string becomes `"toy_lensing"`.
- **Sandbox:** writes to this repo are blocked by the default command sandbox (only the sbibm-jax cwd + `$TMPDIR` are writable). Run write/commit commands with the sandbox disabled, or add this repo to the writable allowlist via `/sandbox`. Reads work in-sandbox.
- **Branch:** the repo is on `main`; create a feature branch before editing (e.g. `git switch -c migrate-to-sbibm-jax`).
- **Lint baseline:** bare `flake8` is never clean here (pre-existing E501 etc.). Judge each changed `.py` by **new** violations vs `HEAD`, using the CI selection `--select=E9,F63,F7,F82` for hard errors.
- **Verification env:** `export JAX_PLATFORMS=cpu` and `export HF_HOME=$TMPDIR/hfhome` for any command that constructs a `TaskDataset` (forces CPU; caches HF downloads in tmp).

---

## File Structure

**Modified**
- `pyproject.toml` — add `sbibm-jax[loader]` dep; make project non-packaged; drop test infra.
- `scripts/notebook_template.md` — generic vector-task notebook template.
- `scripts/two_moons_template.md` — two_moons notebook template.
- `scripts/notebook_stub.txt` — legacy unused stub (kept consistent).
- `examples/sbi-benchmarks/<task>/<model>/*.ipynb` — 15 regenerated notebooks.
- `examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb` — hand-edited.
- `examples/sbi-benchmarks/lensing/lensing_example.ipynb` — hand-edited.
- `scripts/train_sbi_model.py`, `scripts/train_sbi_model_sbibm.py` — migrated.
- `examples/sbi-benchmarks/gravitational_waves/train-gw.py`, `examples/sbi-benchmarks/lensing/train-lensing.py` — migrated.
- `.github/workflows/python-app.yml` — drop pytest/coverage/badge steps.
- `README.md` — remove `src/gensbi_examples` references.

**Deleted**
- `tests/test_tasks.py`
- `scripts/write_stats.py`
- `src/gensbi_examples/` (`tasks.py`, `graph.py`, `mask.py`, `__init__.py`, `stats/*.npz`)

---

## Task 1: Dependency wiring & packaging

Lay the foundation: declare the new dependency, stop building the deleted-soon package, and prove the new loader API works end-to-end against the test repo.

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: an environment where `from sbibm_jax.data import TaskDataset` imports and serves `(theta, x)` for vector tasks; the verified API surface every later task relies on.

- [ ] **Step 1: Create the feature branch**

Run (sandbox disabled):
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git switch -c migrate-to-sbibm-jax
```

- [ ] **Step 2: Edit `pyproject.toml` — add the dependency**

In `[project].dependencies`, add the line (keep the rest):
```toml
    "sbibm-jax[loader]",
```

- [ ] **Step 3: Edit `pyproject.toml` — make the project non-packaged**

Remove the build-system table entirely:
```toml
[build-system]
requires = ["uv_build>=0.10.7,<0.11.0"]
build-backend = "uv_build"
```
and add a `[tool.uv]` table (anywhere after `[project]`):
```toml
[tool.uv]
package = false
```

- [ ] **Step 4: Edit `pyproject.toml` — drop the test infrastructure**

Remove the `test` dependency-group and reduce `dev` to lint only:
```toml
[dependency-groups]
lint = [
    "flake8",
]
dev = [
    {include-group = "lint"},
]
```
and delete the `[tool.pytest.ini_options]` table (the `addopts`/`env`/`testpaths` block).

- [ ] **Step 5: Install sbibm-jax editable for local verification**

`sbibm-jax` is not on PyPI yet, so `uv sync` cannot resolve the committed dep. Install it editable from the sibling repo into the active venv instead (out-of-band; not committed):
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
uv pip install -e "/lustre/ific.uv.es/ml/ific088/github/sbibm-jax[loader]"
```
Expected: resolves and installs `sbibm-jax` plus `grain`, `datasets`, `huggingface_hub`, `numpyro`, `diffrax`, `fyeldgenerator`, `pandas`. (Do **not** run `uv sync` until `sbibm-jax` is published — it would fail to resolve the PyPI dep.)

- [ ] **Step 6: Verify the loader works (the smoke check)**

Run (throwaway; not committed):
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
JAX_PLATFORMS=cpu HF_HOME=$TMPDIR/hfhome python - <<'PY'
from sbibm_jax.data import TaskDataset

# conditional: returns (theta, x), each tokenized to (bs, dim, 1)
t = TaskDataset("two_moons", kind="conditional", use_prefetching=False)
theta, x = next(iter(t.get_train_loader(64, num_samples=1000)))
assert theta.shape == (64, t.dim_theta, 1), theta.shape
assert x.shape == (64, t.dim_x, 1), x.shape
vtheta, vx = next(iter(t.get_val_loader(64)))
assert vtheta.shape[1:] == (t.dim_theta, 1)
obs, ref = t.get_reference(1)
tp = t.get_true_parameters(1)
print("conditional OK", t.dim_theta, t.dim_x, ref.shape, tp.shape)

# joint: returns a single (bs, dim_theta + dim_x, 1) array
tj = TaskDataset("two_moons", kind="joint", use_prefetching=False)
b = next(iter(tj.get_train_loader(64, num_samples=1000)))
assert b.shape == (64, tj.dim_theta + tj.dim_x, 1), b.shape
print("joint OK", b.shape)
print("ALL OK")
PY
```
Expected: prints `conditional OK …`, `joint OK …`, `ALL OK` with no assertion error. (Confirms the PyPI-bound API, the default test repo, and the tokenization/shape contract later tasks rely on.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml
git commit -m "build: depend on sbibm-jax[loader], drop gensbi_examples packaging"
```

---

## Task 2: Migrate the notebook templates and regenerate

Edit the two templates + the legacy stub, then regenerate all 15 vector-task notebooks. The placeholders (`{task_name_gensbi}`, `{kind}`, …) must survive the edits so the generators still substitute them.

**Files:**
- Modify: `scripts/notebook_template.md`, `scripts/two_moons_template.md`, `scripts/notebook_stub.txt`
- Regenerate: `examples/sbi-benchmarks/{bernoulli_glm,gaussian_linear,gaussian_mixture,slcp}/<model>/*.ipynb` and `examples/sbi-benchmarks/two_moons/<model>/*.ipynb`

**Interfaces:**
- Consumes: the verified `TaskDataset` API from Task 1.
- Produces: regenerated `.ipynb` notebooks containing the new API.

- [ ] **Step 1: Edit `scripts/notebook_template.md`**

Colab install cell — replace:
```python
    !uv pip install --quiet "gensbi[cuda13,examples]"
```
with:
```python
    !uv pip install --quiet "gensbi[cuda13,examples]" "sbibm-jax[loader]"
```

Import/construct cell — replace:
```python
from gensbi_examples.tasks import get_task
task = get_task("{task_name_gensbi}", kind="{kind}", use_prefetching=False)
```
with:
```python
from sbibm_jax.data import TaskDataset
task = TaskDataset("{task_name_gensbi}", kind="{kind}", use_prefetching=False)
```

Loader cell — replace:
```python
train_dataset = task.get_train_dataset(batch_size)
val_dataset = task.get_val_dataset(batch_size)
```
with:
```python
train_dataset = task.get_train_loader(batch_size, num_samples=nsamples)
val_dataset = task.get_val_loader(batch_size)
```

Dims cell — replace:
```python
dim_obs = task.dim_obs  # Number of parameters to infer
dim_cond = task.dim_cond    # Number of observed data dimensions
```
with:
```python
dim_obs = task.dim_theta  # Number of parameters to infer
dim_cond = task.dim_x    # Number of observed data dimensions
```
(Leave the following `dim_joint = task.dim_joint` line unchanged.)

- [ ] **Step 2: Edit `scripts/two_moons_template.md`**

Apply the identical Colab, loader, and dims edits as Step 1. The import/construct cell here is different — replace:
```python
from gensbi_examples.tasks import TwoMoons
task = TwoMoons(kind="{kind}")
```
with:
```python
from sbibm_jax.data import TaskDataset
task = TaskDataset("two_moons", kind="{kind}")
```
(This template's dims cell ends with `dim_joint = dim_obs + dim_cond` — leave that line unchanged. The calibration section's `task.dataset["test"].with_format("jax")[:n]` is unchanged.)

- [ ] **Step 3: Edit `scripts/notebook_stub.txt`**

Apply the same edits as Step 1 (Colab cell, `get_task`→`TaskDataset`, loaders→`get_*_loader` with `num_samples=nsamples`, `dim_obs`/`dim_cond`→`dim_theta`/`dim_x`). The stub uses the `# %%`-style cell markers but the code lines are identical strings.

- [ ] **Step 4: Verify no template still references the old package**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
grep -n "gensbi_examples\|get_train_dataset\|get_val_dataset\|\.dim_obs\|\.dim_cond" \
  scripts/notebook_template.md scripts/two_moons_template.md scripts/notebook_stub.txt
```
Expected: no output (exit code 1).

- [ ] **Step 5: Regenerate the notebooks**

Run (requires `jupytext`; install with `uv pip install jupytext` if missing):
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
python scripts/make_notebook.py
python scripts/make_notebook_two_moons.py
```
Expected: prints `Created notebook: …` for all 15 notebooks, no errors.

- [ ] **Step 6: Verify the regenerated notebooks carry the new API**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
echo "stale refs (expect none):"
grep -rl "gensbi_examples" examples/sbi-benchmarks/{two_moons,bernoulli_glm,gaussian_linear,gaussian_mixture,slcp} --include="*.ipynb" || echo "  none"
echo "new API present (expect 15):"
grep -rl "from sbibm_jax.data import TaskDataset" examples/sbi-benchmarks/{two_moons,bernoulli_glm,gaussian_linear,gaussian_mixture,slcp} --include="*.ipynb" | wc -l
```
Expected: `none` for stale refs; `15` for new-API count.

- [ ] **Step 7: Commit**

```bash
git add scripts/notebook_template.md scripts/two_moons_template.md scripts/notebook_stub.txt \
  examples/sbi-benchmarks/two_moons examples/sbi-benchmarks/bernoulli_glm \
  examples/sbi-benchmarks/gaussian_linear examples/sbi-benchmarks/gaussian_mixture \
  examples/sbi-benchmarks/slcp
git commit -m "refactor(examples): generate SBI notebooks from sbibm-jax TaskDataset"
```

---

## Task 3: Migrate the gravitational-waves and lensing notebooks

These two are hand-edited: import + constructor only. All inline normalization, `split_data`, and `df_*` usage is unchanged. The `.ipynb` source strings are unique, so edit the raw JSON directly (or use a notebook cell editor).

**Files:**
- Modify: `examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb`
- Modify: `examples/sbi-benchmarks/lensing/lensing_example.ipynb`

**Interfaces:**
- Consumes: `TaskDataset(name)` exposes `df_train`/`df_val`/`df_test` (verified to exist — same as the old task object).
- Produces: nothing downstream.

- [ ] **Step 1: Edit `gw_example.ipynb`**

Replace `from gensbi_examples.tasks import GravitationalWaves` with `from sbibm_jax.data import TaskDataset`.
Replace `task = GravitationalWaves()` with `task = TaskDataset("gravitational_waves")`.
In the Colab install cell, replace `"gensbi[cuda13,examples]"` with `"gensbi[cuda13,examples]" "sbibm-jax[loader]"`.

- [ ] **Step 2: Edit `lensing_example.ipynb`**

Replace `from gensbi_examples.tasks import GravitationalLensing` with `from sbibm_jax.data import TaskDataset`.
Replace `task = GravitationalLensing()` with `task = TaskDataset("toy_lensing")` (note: `toy_lensing`, not `lensing`).
In the Colab install cell, replace `"gensbi[cuda13,examples]"` with `"gensbi[cuda13,examples]" "sbibm-jax[loader]"`.

- [ ] **Step 3: Verify the edits (static)**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
grep -l "gensbi_examples" examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb \
  examples/sbi-benchmarks/lensing/lensing_example.ipynb || echo "no stale refs OK"
grep -o 'TaskDataset("[a-z_]*")' examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb \
  examples/sbi-benchmarks/lensing/lensing_example.ipynb
```
Expected: `no stale refs OK`; and the second grep prints `TaskDataset("gravitational_waves")` and `TaskDataset("toy_lensing")`.

- [ ] **Step 4 (optional runtime check): construct both tasks**

Only if you want a runtime check and can afford the dataset download (large for these tasks):
```bash
JAX_PLATFORMS=cpu HF_HOME=$TMPDIR/hfhome python - <<'PY'
from sbibm_jax.data import TaskDataset
for name in ("gravitational_waves", "toy_lensing"):
    t = TaskDataset(name)
    row = t.df_test[0]
    print(name, "OK", {k: getattr(row[k], "shape", None) for k in ("thetas", "xs")})
PY
```
Expected: prints `gravitational_waves OK …` and `toy_lensing OK …`. (Skip if bandwidth-constrained; Step 3 already proves the code change.)

- [ ] **Step 5: Commit**

```bash
git add examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb \
  examples/sbi-benchmarks/lensing/lensing_example.ipynb
git commit -m "refactor(examples): load gw/lensing data via sbibm-jax TaskDataset"
```

---

## Task 4: Migrate the training/utility `.py` scripts

Four scripts, all mechanical swaps. `train_sbi_model_sbibm.py` additionally renames the normalization helpers. Full execution needs a GPU, `gensbi`, and checkpoints, so verification is `py_compile` + flake8 + grep (static).

**Files:**
- Modify: `scripts/train_sbi_model.py`
- Modify: `scripts/train_sbi_model_sbibm.py`
- Modify: `examples/sbi-benchmarks/gravitational_waves/train-gw.py`
- Modify: `examples/sbi-benchmarks/lensing/train-lensing.py`

**Interfaces:**
- Consumes: `TaskDataset` (loaders, dims, `normalize_x`/`unnormalize_theta`, `df_*`).
- Produces: nothing downstream.

- [ ] **Step 1: Edit `scripts/train_sbi_model.py`**

- `from gensbi_examples.tasks import get_task` → `from sbibm_jax.data import TaskDataset`
- `    task = get_task(task_name, kind=kind)` → `    task = TaskDataset(task_name, kind=kind)`
- `    train_dataset = task.get_train_dataset(batch_size)` → `    train_dataset = task.get_train_loader(batch_size, num_samples=100_000)`
- `    val_dataset = task.get_val_dataset(` → `    val_dataset = task.get_val_loader(` (the `512` and closing paren on the next lines stay)
- `    dim_obs = task.dim_obs` → `    dim_obs = task.dim_theta`
- `    dim_cond = task.dim_cond` → `    dim_cond = task.dim_x`
- Leave `dim_joint = task.dim_joint`, `task.get_reference(...)`, `task.get_true_parameters(...)`, and `task.dataset["test"]...` unchanged.

- [ ] **Step 2: Edit `scripts/train_sbi_model_sbibm.py`**

- `from gensbi_examples.tasks import get_task` → `from sbibm_jax.data import TaskDataset`
- `    task = get_task(` → `    task = TaskDataset(` and on its argument line `task_name, kind=kind, normalize_data=True, use_prefetching=True, max_workers=2` → `task_name, kind=kind, normalize=True, use_prefetching=True, max_workers=2`
- `    train_dataset = task.get_train_dataset(batch_size, nsamples=dataset_size)` → `    train_dataset = task.get_train_loader(batch_size, num_samples=dataset_size)`
- `    val_dataset = task.get_val_dataset(` → `    val_dataset = task.get_val_loader(`
- `    dim_obs = task.dim_obs` → `    dim_obs = task.dim_theta`
- `    dim_cond = task.dim_cond` → `    dim_cond = task.dim_x`
- `        obs_for_model = task.normalize_cond(obs_for_model)` → `        obs_for_model = task.normalize_x(obs_for_model)`
- `        samples = task.unnormalize_obs(samples)` → `        samples = task.unnormalize_theta(samples)`

- [ ] **Step 3: Edit `examples/sbi-benchmarks/gravitational_waves/train-gw.py`**

- `from gensbi_examples.tasks import GravitationalWaves` → `from sbibm_jax.data import TaskDataset`
- `    task = GravitationalWaves()` → `    task = TaskDataset("gravitational_waves")`

- [ ] **Step 4: Edit `examples/sbi-benchmarks/lensing/train-lensing.py`**

- `from gensbi_examples.tasks import GravitationalLensing` → `from sbibm_jax.data import TaskDataset`
- `    task = GravitationalLensing()` → `    task = TaskDataset("toy_lensing")`

- [ ] **Step 5: Verify (compile + lint + grep)**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
python -m py_compile scripts/train_sbi_model.py scripts/train_sbi_model_sbibm.py \
  examples/sbi-benchmarks/gravitational_waves/train-gw.py \
  examples/sbi-benchmarks/lensing/train-lensing.py && echo "compile OK"
grep -n "gensbi_examples\|get_train_dataset\|get_val_dataset\|\.dim_obs\|\.dim_cond\|normalize_cond\|unnormalize_obs" \
  scripts/train_sbi_model.py scripts/train_sbi_model_sbibm.py \
  examples/sbi-benchmarks/gravitational_waves/train-gw.py \
  examples/sbi-benchmarks/lensing/train-lensing.py || echo "no stale refs OK"
uv run flake8 scripts/train_sbi_model.py scripts/train_sbi_model_sbibm.py \
  examples/sbi-benchmarks/gravitational_waves/train-gw.py \
  examples/sbi-benchmarks/lensing/train-lensing.py \
  --select=E9,F63,F7,F82 --show-source --statistics && echo "flake8 hard-errors OK"
```
Expected: `compile OK`; `no stale refs OK`; `flake8 hard-errors OK` (zero E9/F63/F7/F82). If `uv run flake8` cannot resolve the env, run `flake8` from the active venv directly.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_sbi_model.py scripts/train_sbi_model_sbibm.py \
  examples/sbi-benchmarks/gravitational_waves/train-gw.py \
  examples/sbi-benchmarks/lensing/train-lensing.py
git commit -m "refactor(scripts): migrate training scripts to sbibm-jax TaskDataset"
```

---

## Task 5: Remove `gensbi_examples` and clean up CI/docs

With every consumer migrated, delete the package, its tests, the obsolete stats script, and prune the CI/README references to them.

**Files:**
- Delete: `src/gensbi_examples/`, `tests/test_tasks.py`, `scripts/write_stats.py`
- Modify: `.github/workflows/python-app.yml`, `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Delete the package and obsolete files**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git rm -r src/gensbi_examples
git rm tests/test_tasks.py scripts/write_stats.py
```
(If `tests/` is now empty and contains no other files, it may be left empty or removed — `git rm` already dropped the tracked file.)

- [ ] **Step 2: Edit `.github/workflows/python-app.yml`**

Delete the `Test with pytest`, `Copy badges to img/badges`, and `Commit and push badges to main` steps (everything from the `- name: Test with pytest` line through the end of the file). Keep the `Lint with flake8` step as the last step. The build then installs deps and lints only.

- [ ] **Step 3: Edit `README.md`**

Remove the structure bullet:
```markdown
- `src/gensbi_examples`: Helper utilities for the examples.
```
And reword the download paragraph that claims a helper package — replace:
```markdown
The `gensbi-examples` package provides the helper utilities, but the notebooks and training scripts live in this repository. To get them, clone the repo:
```
with:
```markdown
The notebooks and training scripts live in this repository. Task data is loaded via [`sbibm-jax`](https://github.com/aurelio-amerio/sbibm-jax) (`pip install "sbibm-jax[loader]"`). To get the examples, clone the repo:
```

- [ ] **Step 4: Verify the package is fully gone**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
echo "remaining gensbi_examples refs (expect only docs/specs+plans, .git excluded):"
grep -rI "gensbi_examples" . --exclude-dir=.git --exclude-dir=.venv --exclude-dir=docs || echo "  none outside docs"
test ! -e src/gensbi_examples && echo "package dir gone OK"
grep -c "cov=gensbi_examples" .github/workflows/python-app.yml || echo "CI clean OK"
```
Expected: `none outside docs`; `package dir gone OK`; `CI clean OK` (grep -c prints 0 / the `|| echo` fires).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove gensbi_examples package, tests, and stats script"
```

---

## Self-Review

**1. Spec coverage**
- Templates + regenerate (15 notebooks) → Task 2. ✓
- gw/lensing notebooks → Task 3. ✓
- `.py` scripts incl. normalize rename → Task 4. ✓
- Deletions (package, tests, write_stats) → Task 5. ✓
- Packaging (dep, non-package, test-config trim) → Task 1. ✓
- CI + README → Task 5. ✓
- Colab install cells → Tasks 2 & 3. ✓
- Locked decisions (test-repo default, 100k via `nsamples`, PyPI dep) → Global Constraints + Steps. ✓
- `lensing` dir vs `toy_lensing` string → Global Constraints + Tasks 3/4. ✓

**2. Placeholder scan:** every code step shows the exact before/after string; verification steps show exact commands + expected output. No TBD/TODO. The one open call (delete vs keep an empty `tests/`) has a stated default. ✓

**3. Type/name consistency:** `dim_theta`/`dim_x`/`dim_joint`, `get_train_loader(num_samples=…)`, `get_val_loader`, `normalize_x`/`unnormalize_theta`, and `TaskDataset("gravitational_waves")`/`TaskDataset("toy_lensing")` are used identically across Tasks 1–4 and match the verified API in Task 1, Step 6. ✓
