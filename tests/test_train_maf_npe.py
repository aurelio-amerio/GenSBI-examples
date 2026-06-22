import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe before any pyplot import

import importlib.util
import pathlib

import numpy as np
import pytest

# --- load the example script as a module (it lives outside the package) ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py"
_CONFIG = _REPO_ROOT / "examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("train_maf_npe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_yaml_has_required_sections():
    import yaml
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    assert cfg["task_name"] == "two_moons"
    for section in ("model", "optimizer", "training", "evaluation"):
        assert section in cfg, f"missing section {section!r}"
    assert cfg["model"]["transformer"] in ("affine", "rqspline")
    assert cfg["training"]["nsamples"] < 100_000  # below get_train_dataset's default nsamples; real cap is task.max_samples (runtime)
    assert cfg["evaluation"]["grid_size"] > 0


def test_build_transformer_selects_type():
    mod = _load_script_module()
    from gensbi.normalizing_flows import Affine, RQSpline
    assert isinstance(mod.build_transformer({"transformer": "affine"}), Affine)
    rq = mod.build_transformer({"transformer": "rqspline", "num_bins": 6})
    assert isinstance(rq, RQSpline)
    assert rq.num_bins == 6
    with pytest.raises(ValueError):
        mod.build_transformer({"transformer": "nope"})


def test_build_flow_runs_log_prob():
    mod = _load_script_module()
    from flax import nnx
    import jax.numpy as jnp
    flow = mod.build_flow(nnx.Rngs(0), dim_obs=2, dim_cond=2,
                          model_cfg={"n_layers": 2, "transformer": "affine"})
    x = jnp.zeros((4, 2))
    cond = jnp.zeros((4, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(lp)))


def test_build_training_config_merges_defaults_and_overrides(tmp_path):
    mod = _load_script_module()
    from gensbi.recipes import ConditionalFlowPipeline
    cfg = {
        "optimizer": {"max_lr": 4.0e-4, "min_lr": 4.0e-6, "warmup_steps": 500,
                      "decay_transition": 0.80},
        "training": {"nsteps": 123, "ema_decay": 0.99, "val_every": 50,
                     "early_stopping": True, "experiment_id": 7},
    }
    tc = mod.build_training_config(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
    # every key the pipeline reads must be present (defaults filled in)
    for key in ConditionalFlowPipeline.get_default_training_config():
        assert key in tc, f"missing required training_config key {key!r}"
    # overrides applied
    assert tc["nsteps"] == 123
    assert tc["max_lr"] == 4.0e-4
    assert tc["experiment_id"] == 7
    assert tc["checkpoint_dir"] == str(tmp_path / "ckpt")


def test_load_obs_stats_shape_two_moons():
    mod = _load_script_module()
    import jax.numpy as jnp
    mean, std = mod.load_obs_stats("two_moons", dim_obs=2)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert bool(jnp.all(std > 0))


@pytest.fixture
def tiny_pipeline(tmp_path):
    """A minimal ConditionalFlowPipeline on a tiny CPU flow (no HF, no training)."""
    mod = _load_script_module()
    from flax import nnx
    from gensbi.recipes import ConditionalFlowPipeline
    import jax.numpy as jnp

    flow = mod.build_flow(nnx.Rngs(0), dim_obs=2, dim_cond=2,
                          model_cfg={"n_layers": 2, "transformer": "affine"})
    obs = jnp.zeros((8, 2, 1))   # (B, dim_obs, 1) — theta
    cond = jnp.zeros((8, 2, 1))  # (B, dim_cond, 1) — x
    dummy = [(obs, cond)]
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc["checkpoint_dir"] = str(tmp_path / "ckpt")
    pipe = ConditionalFlowPipeline(flow, dummy, dummy, dim_obs=2, dim_cond=2,
                                   training_config=tc)
    return mod, pipe


def test_apply_standardization_sets_buffers(tiny_pipeline):
    mod, pipe = tiny_pipeline
    import jax.numpy as jnp
    mean = jnp.array([1.0, -2.0])
    std = jnp.array([3.0, 4.0])
    mod.apply_standardization(pipe, mean, std)
    assert pipe._standardized is True
    # both model and EMA must carry the buffer (EMA averages only Params)
    for m in (pipe.model, pipe.ema_model):
        std_bijection = [b for b in m.chain.bijections
                         if b.__class__.__name__ == "Standardize"][0]
        assert bool(jnp.allclose(std_bijection.mean.value, mean))
        assert bool(jnp.allclose(std_bijection.std.value, std))


def test_make_density_grid_shapes_and_bounds():
    mod = _load_script_module()
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2)) * 0.3
    xx, yy, grid_pts = mod.make_density_grid(ref, grid_size=20, padding=0.5)
    assert xx.shape == (20, 20)
    assert yy.shape == (20, 20)
    assert grid_pts.shape == (400, 2)
    assert grid_pts[:, 0].min() <= ref[:, 0].min()
    assert grid_pts[:, 0].max() >= ref[:, 0].max()


def test_posterior_density_shape_and_finite(tiny_pipeline):
    mod, pipe = tiny_pipeline
    import jax.numpy as jnp
    mod.apply_standardization(pipe, jnp.zeros(2), jnp.ones(2))
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2)) * 0.3
    xx, yy, grid_pts = mod.make_density_grid(ref, grid_size=15, padding=0.5)
    Z = mod.posterior_density(pipe, grid_pts, obs=np.array([0.1, 0.2]), grid_size=15)
    assert Z.shape == (15, 15)
    assert np.all(np.isfinite(Z))
    assert np.all(Z >= 0.0)


def test_plot_posterior_contour_returns_figure(tiny_pipeline):
    mod, pipe = tiny_pipeline
    import jax.numpy as jnp
    import matplotlib
    mod.apply_standardization(pipe, jnp.zeros(2), jnp.ones(2))
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2)) * 0.3
    xx, yy, grid_pts = mod.make_density_grid(ref, grid_size=15, padding=0.5)
    Z = mod.posterior_density(pipe, grid_pts, obs=np.array([0.0, 0.0]), grid_size=15)
    fig, ax = mod.plot_posterior_contour(xx, yy, Z, true_param=np.array([0.0, 0.0]),
                                         ref_samples=ref, n_ref_overlay=100)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
