# Migrate GenSBI-examples to `sbibm_jax.data.TaskDataset`

- **Date:** 2026-06-22
- **Status:** Approved (design); ready for implementation plan
- **Repos:**
  - Target (edited): `GenSBI-examples` (`/lhome/ific/a/aamerio/data/github/GenSBI-examples`)
  - Source dependency (read-only here): `sbibm-jax` (`/lustre/ific.uv.es/ml/ific088/github/sbibm-jax`)

## Context & motivation

The example notebooks and training scripts in `GenSBI-examples` load benchmark
task data through a local helper package, `gensbi_examples.tasks`, which pulls
pre-generated `(theta, x)` rows from the HuggingFace dataset and exposes task
dims, reference posteriors, and grain data loaders.

`sbibm-jax` now ships a consumer loader, `sbibm_jax.data.TaskDataset`, that does
the same job, driven entirely by the published `metadata.json` (new schema:
`x_kind`/`x_shape`, `theta_kind`/`theta_shape`, `stats`, `has_reference`,
`num_observations`). The goal is to make `GenSBI-examples` depend on
`sbibm-jax` for all task data and **remove `gensbi_examples` entirely**.

The old API surface used across the repo is small and maps almost 1:1 onto
`TaskDataset`, so this is a surgical swap, not a rewrite. The 15 vector-task
notebooks are *generated* from two MyST-Markdown templates, so they are migrated
by editing the templates and regenerating — only the two image/time-series
notebooks (gravitational waves, lensing) are hand-edited.

## Goals

- All task data in `GenSBI-examples` flows through `sbibm_jax.data.TaskDataset`.
- `gensbi_examples` (the `src/gensbi_examples/` package) is deleted, along with
  its now-obsolete consumers (tests, the stats-writing script).
- Notebooks and scripts remain runnable end-to-end (same behaviour as before,
  minus the package indirection).

## Non-goals

- No use of `OnlineTaskDataset` (simulate-on-the-fly). Examples keep using the
  pre-generated HF splits.
- No changes to GenSBI model / pipeline code (`gensbi.*`). The model-side
  parameter names `dim_obs` / `dim_cond` are GenSBI's own API and are untouched;
  only the *source* of those values changes.
- No new datasets, no dataset regeneration, no graph/edge masks (the notebooks
  never used them).
- The `gaussian_random_field` / `gaussian_random_field_256` example folders have
  no notebooks yet and are out of scope.

## Locked decisions

1. **Data repo:** read from `TaskDataset`'s built-in default, the **test repo**
   (`aurelio-amerio/SBI-benchmarks-test`). It is currently the only repo
   carrying the new-schema `metadata.json` + stats for all tasks. No `repo=`
   argument in the examples. (Future: flip to production with a one-line change
   once it is refreshed.)
2. **Train size:** preserve the old 100k-row subsample. The templates already
   define an (otherwise unused) `nsamples = int(1e5)` variable — wire it in as
   `get_train_loader(batch_size, num_samples=nsamples)`. Scripts that pass an
   explicit size keep it.
3. **Dependency:** declare `sbibm-jax[loader]` from **PyPI** in `pyproject.toml`
   and in the Colab install cells. (The package will be published in parallel.
   For local verification before it lands on PyPI, install it editable from the
   sibling repo — see Verification.)
4. **Stats:** already ported into `metadata.json` by the maintainer; no action
   needed here.

## API mapping (old → new)

| Old (`gensbi_examples.tasks`) | New (`sbibm_jax.data.TaskDataset`) |
|---|---|
| `from gensbi_examples.tasks import get_task` | `from sbibm_jax.data import TaskDataset` |
| `from gensbi_examples.tasks import TwoMoons` | `from sbibm_jax.data import TaskDataset` |
| `from gensbi_examples.tasks import GravitationalWaves` | `from sbibm_jax.data import TaskDataset` |
| `from gensbi_examples.tasks import GravitationalLensing` | `from sbibm_jax.data import TaskDataset` |
| `get_task("slcp", kind=k, use_prefetching=False)` | `TaskDataset("slcp", kind=k, use_prefetching=False)` |
| `TwoMoons(kind=k)` | `TaskDataset("two_moons", kind=k)` |
| `GravitationalWaves()` | `TaskDataset("gravitational_waves")` |
| `GravitationalLensing()` | `TaskDataset("toy_lensing")` |
| `task.dim_obs` | `task.dim_theta` |
| `task.dim_cond` | `task.dim_x` |
| `task.dim_joint` | `task.dim_joint` (unchanged) |
| `task.get_train_dataset(bs)` | `task.get_train_loader(bs, num_samples=nsamples)` |
| `task.get_train_dataset(bs, nsamples=N)` | `task.get_train_loader(bs, num_samples=N)` |
| `task.get_val_dataset(bs)` | `task.get_val_loader(bs)` |
| `task.get_test_dataset(bs)` | `task.get_test_loader(bs)` (not used in examples) |
| `task.normalize_cond(x)` | `task.normalize_x(x)` |
| `task.unnormalize_obs(theta)` | `task.unnormalize_theta(theta)` |
| `get_task(..., normalize_data=True)` | `TaskDataset(..., normalize=True)` |
| `task.get_reference(num_observation=i)` | unchanged |
| `task.get_true_parameters(i)` | unchanged |
| `task.dataset["test"].with_format("jax")[:n]` | unchanged |
| `task.df_train` / `task.df_val` / `task.df_test` | unchanged |

Notes:
- **Directory vs task name:** the lensing example directory stays
  `examples/sbi-benchmarks/lensing/`. Only the in-code task string becomes
  `"toy_lensing"` (the published config name).
- **Local variable names stay `dim_obs` / `dim_cond`.** Only the right-hand side
  changes (`= task.dim_theta` / `= task.dim_x`). The values still flow into
  `Flux1Params(dim_obs=…, dim_cond=…)` and `init_pipeline_from_config(…, dim_obs,
  dim_cond, …)` unchanged.
- **Conditional vs joint semantics are preserved.** `TaskDataset`'s collate
  tokenizes each scalar to a length-1 token (`[..., None]`) and, for
  `kind="conditional"`, returns `(theta, x)`; for `kind="joint"`, concatenates
  `[theta; x]` along axis 1 — identical to the old `process_conditional` /
  `process_joint`.

## Detailed change map

### 1. Notebook templates (real edit sites for the 15 vector notebooks)

**`scripts/notebook_template.md`** (drives bernoulli_glm, gaussian_linear,
gaussian_mixture, slcp):
- Colab install cell: `!uv pip install --quiet "gensbi[cuda13,examples]"` →
  `!uv pip install --quiet "gensbi[cuda13,examples]" "sbibm-jax[loader]"`
- Import/construct:
  ```python
  from sbibm_jax.data import TaskDataset
  task = TaskDataset("{task_name_gensbi}", kind="{kind}", use_prefetching=False)
  ```
- Loaders:
  ```python
  train_dataset = task.get_train_loader(batch_size, num_samples=nsamples)
  val_dataset = task.get_val_loader(batch_size)
  ```
- Dims:
  ```python
  dim_obs = task.dim_theta   # Number of parameters to infer
  dim_cond = task.dim_x      # Number of observed data dimensions
  dim_joint = task.dim_joint # Joint dimension (for model input)
  ```

**`scripts/two_moons_template.md`** (drives the 6 two_moons variants): same
changes, plus:
- Import/construct: `from sbibm_jax.data import TaskDataset` /
  `task = TaskDataset("two_moons", kind="{kind}")`
- `dim_joint = dim_obs + dim_cond` line stays unchanged.
- Calibration section's `task.dataset["test"].with_format("jax")[:n]` stays
  unchanged.

**`scripts/notebook_stub.txt`** (legacy stub, not read by the generators):
apply the same import / loader / dims edits for consistency so no stale
`gensbi_examples` reference remains. (Alternative: delete it — decide during
implementation; updating is lower-risk.)

### 2. Regenerate the notebooks

Run both generators (require `jupytext`, which they import):
```bash
python scripts/make_notebook.py
python scripts/make_notebook_two_moons.py
```
This rewrites all 15 `.ipynb` under `examples/sbi-benchmarks/{task}/{model}/`.
Confirm via `git diff` that only the intended cells changed.

### 3. Hand-edit the two special notebooks

**`examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb`**
- Colab cell: add `"sbibm-jax[loader]"`.
- `from gensbi_examples.tasks import GravitationalWaves` →
  `from sbibm_jax.data import TaskDataset`
- `task = GravitationalWaves()` → `task = TaskDataset("gravitational_waves")`
- Everything else (inline `normalize`, `split_data`, `df_train/df_val/df_test`,
  hardcoded `dim_obs`/`ch_obs`, normalization constants) is untouched.

**`examples/sbi-benchmarks/lensing/lensing_example.ipynb`**
- Colab cell: add `"sbibm-jax[loader]"`.
- `from gensbi_examples.tasks import GravitationalLensing` →
  `from sbibm_jax.data import TaskDataset`
- `task = GravitationalLensing()` → `task = TaskDataset("toy_lensing")`
- Directory stays `lensing/`. Everything else untouched.

### 4. Training / utility `.py` scripts

**`scripts/train_sbi_model.py`**
- import + `get_task(task_name, kind=kind)` → `TaskDataset(task_name, kind=kind)`
- `task.get_train_dataset(batch_size)` →
  `task.get_train_loader(batch_size, num_samples=100_000)` (preserve old 1e5
  default)
- `task.get_val_dataset(...)` → `task.get_val_loader(...)`
- `task.dim_obs`/`task.dim_cond` → `task.dim_theta`/`task.dim_x`
- `task.dataset["test"]`, `get_reference`, `get_true_parameters`: unchanged.

**`scripts/train_sbi_model_sbibm.py`** (the only normalization user)
- `get_task(task_name, kind=kind, normalize_data=True, use_prefetching=True,
  max_workers=2)` → `TaskDataset(task_name, kind=kind, normalize=True,
  use_prefetching=True, max_workers=2)`
- `task.get_train_dataset(batch_size, nsamples=dataset_size)` →
  `task.get_train_loader(batch_size, num_samples=dataset_size)`
- `task.get_val_dataset(512)` → `task.get_val_loader(512)`
- `task.dim_obs`/`task.dim_cond` → `task.dim_theta`/`task.dim_x`
- `task.normalize_cond(obs_for_model)` → `task.normalize_x(obs_for_model)`
- `task.unnormalize_obs(samples)` → `task.unnormalize_theta(samples)`
- Behavioural note: `TaskDataset.normalize_x` / `normalize_theta` apply the
  metadata stats unconditionally (they do not depend on the `normalize` flag);
  since this script always normalized, semantics match.

**`examples/sbi-benchmarks/gravitational_waves/train-gw.py`**
- `from gensbi_examples.tasks import GravitationalWaves` →
  `from sbibm_jax.data import TaskDataset`
- `task = GravitationalWaves()` → `task = TaskDataset("gravitational_waves")`
  (only `df_train/df_val/df_test` are used afterwards — unchanged).

**`examples/sbi-benchmarks/lensing/train-lensing.py`**
- `from gensbi_examples.tasks import GravitationalLensing` →
  `from sbibm_jax.data import TaskDataset`
- `task = GravitationalLensing()` → `task = TaskDataset("toy_lensing")`.

### 5. Deletions

- `tests/test_tasks.py` — tested the old package; no longer relevant.
- `scripts/write_stats.py` — obsolete; normalization stats now live in
  `metadata.json` (computed during dataset generation).
- `src/gensbi_examples/` — the entire package (`tasks.py`, `graph.py`,
  `mask.py`, `__init__.py`, `stats/`).

### 6. Packaging, CI, docs

**`pyproject.toml`**
- Add `"sbibm-jax[loader]"` to `[project].dependencies`.
- The repo no longer ships a Python package, so stop building one:
  - Remove the `[build-system]` table (the `uv_build` backend builds
    `src/gensbi_examples`, which is being deleted).
  - Add `[tool.uv]` with `package = false` so uv treats the project as a
    non-packaged environment manager.
- Test config cleanup: with `tests/` emptied, remove the `[tool.pytest.ini_options]`
  block and the `test` dependency-group entries that exist only for the deleted
  tests (keep the `lint`/flake8 group). Final call made during implementation.

**`.github/workflows/python-app.yml`**
- The test step runs `pytest … --cov=gensbi_examples …`. With the package and
  tests removed, remove that step (or repoint it at a lightweight import smoke
  check). Read the workflow during implementation and choose the minimal edit.

**`README.md`**
- Line listing `src/gensbi_examples: Helper utilities for the examples` — remove
  or replace with a note that examples load data via `sbibm-jax[loader]`.

## Edge cases & risks

- **`lensing` ↔ `toy_lensing`.** Directory name stays `lensing`; the published
  HF config is `toy_lensing`. Easy to get wrong — the task string is the thing
  that must read `toy_lensing`.
- **`jupytext` prerequisite.** The generators import it. If absent in the active
  env, install it (or add it to a dev/authoring group) before regenerating.
- **uv resolution.** `sbibm-jax`'s base deps (`numpyro`, `diffrax`,
  `fyeldgenerator`, `pandas`, `jax>=0.9.1`) get added to the env. Must resolve
  against GenSBI-examples' `jax>=0.9.0,<0.10.0` pin and `gensbi`. Expected to be
  compatible (jax 0.9.x), but verify the lock resolves.
- **`sbibm-jax` not yet on PyPI.** The declared dependency targets PyPI per the
  decision; until it is published, local installs/verification use an editable
  path install from the sibling repo (see Verification). Do not commit a path
  source — the declared dep stays PyPI.
- **Implicit test-repo dependency.** Examples now read from
  `SBI-benchmarks-test` via the default. Acceptable for now; note for a future
  production flip.
- **Worker cap.** `TaskDataset` clamps `max_workers ≤ 8` and only prefetches
  when both `use_prefetching` and `max_workers` are set. Notebooks pass
  `use_prefetching=False` (no prefetch, as before); `train_sbi_model_sbibm.py`
  uses `max_workers=2`. Performance-only, not correctness.

## Verification

Run in an environment that has both `gensbi` and `sbibm-jax[loader]`. Until
`sbibm-jax` is on PyPI, install it editable for testing:
`uv pip install -e "/lustre/ific.uv.es/ml/ific088/github/sbibm-jax[loader]"`
(or via `PYTHONPATH` to the sibling `src`), forcing CPU (`JAX_PLATFORMS=cpu`).

1. **Loader smoke test** (no GPU, no model): for each vector task and both
   kinds, construct `TaskDataset(name, kind=…, use_prefetching=False)`, pull one
   batch from `get_train_loader(bs, num_samples=1000)` and `get_val_loader(bs)`,
   and assert shapes:
   - conditional → `(theta, x)` with `theta.shape == (bs, dim_theta, 1)`,
     `x.shape == (bs, dim_x, 1)`;
   - joint → `(bs, dim_theta + dim_x, 1)`.
   Check `get_reference(1)` and `get_true_parameters(1)` return arrays. (This is
   the new-API analogue of the deleted `tests/test_tasks.py`.)
2. **gw / lensing:** construct `TaskDataset("gravitational_waves")` and
   `TaskDataset("toy_lensing")`; read one row from `df_test["xs"]`/`["thetas"]`.
3. **Regeneration diff:** regenerate and `git diff` the 15 `.ipynb` to confirm
   only the data/dims cells changed.
4. **No stragglers:** `grep -rI gensbi_examples` over the repo (excluding
   `.git`/`.venv`) returns nothing.
5. **Lint:** `flake8` the changed `.py`; judge by *new* violations vs HEAD
   (the repo has pre-existing baseline violations).
6. **Scope of testing:** full notebook execution needs a GPU, `gensbi`, and
   trained checkpoints — out of scope for automated verification. Verification
   is loader/import smoke-level plus regeneration diffs.

## Implementation sequence

1. `pyproject.toml`: add `sbibm-jax[loader]`, make the project non-packaged;
   install the env (editable `sbibm-jax` for local testing).
2. Edit the two templates + the stub; regenerate the 15 notebooks; hand-edit the
   gw and lensing notebooks.
3. Migrate the four `.py` scripts.
4. Deletions: `tests/test_tasks.py`, `scripts/write_stats.py`,
   `src/gensbi_examples/`.
5. CI workflow + README cleanup.
6. Run the verification steps above.
