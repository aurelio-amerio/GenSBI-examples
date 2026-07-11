import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless-safe before any pyplot import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import pathlib

import numpy as np
import pytest

# --- load the example script as a module (it lives outside the package) ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = (_REPO_ROOT /
           "examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py")
_CONFIG = (_REPO_ROOT /
           "examples/sbi-benchmarks/gaussian_random_field/tarflow/config/config_1.yaml")

# Tiny rope model: 8x8 image, patch 4 -> T=4 tokens of F=16; head_dim 8 (%4==0).
_TINY_MODEL_CFG = {
    "img_size": 8, "patch_size": 4, "img_channels": 1, "cond_dim": 2,
    "use_rope": True, "rope_theta": 10000, "head_dim": 8, "num_heads": 2,
    "num_blocks": 2, "layers_per_block": 1, "permutation": "flip",
}


def _load_script_module():
    spec = importlib.util.spec_from_file_location("train_tarflow_grf", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_yaml_has_required_sections():
    import yaml
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "optimizer", "training", "sampling"):
        assert section in cfg, f"missing section {section!r}"
    m = cfg["model"]
    assert m["use_rope"] is True                 # rope-only example by design
    assert m["head_dim"] % 4 == 0                # TarFlowParams rope requirement
    assert m["img_size"] == 32
    assert m["img_size"] % m["patch_size"] == 0  # ImageTokenizer requirement
    assert m["cond_dim"] == 2                    # theta = (log_std, alpha)


def test_build_flow_rope_log_prob_and_sample_shapes():
    mod = _load_script_module()
    from flax import nnx
    import jax
    import jax.numpy as jnp
    flow = mod.build_flow(nnx.Rngs(0), _TINY_MODEL_CFG)
    x = jax.random.normal(jax.random.key(0), (3, 8, 8, 1))
    cond = jax.random.normal(jax.random.key(1), (3, 2, 1))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (3,) and bool(jnp.all(jnp.isfinite(lp)))
    s = flow.sample(jax.random.key(2), cond=cond[:2])
    assert s.shape == (2, 8, 8, 1) and bool(jnp.all(jnp.isfinite(s)))


def test_build_flow_uses_rope_and_vector_prefix():
    mod = _load_script_module()
    from flax import nnx
    flow = mod.build_flow(nnx.Rngs(0), _TINY_MODEL_CFG)
    blk = flow.blocks[0]
    assert blk.pos_embed is None          # rope replaces the learned table
    assert blk.freqs_cis is not None
    assert type(blk.conditioner).__name__ == "VectorConditioner"


def test_build_training_config_merges_defaults_and_overrides(tmp_path):
    mod = _load_script_module()
    from gensbi.recipes import ConditionalFlowPipeline
    cfg = {
        "optimizer": {"max_lr": 4.0e-4},
        "training": {"nsteps": 123, "experiment_id": 7},
    }
    tc = mod.build_training_config(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
    # every key the pipeline reads must be present (defaults filled in)
    for key in ConditionalFlowPipeline.get_default_training_config():
        assert key in tc, f"missing required training_config key {key!r}"
    assert tc["nsteps"] == 123
    assert tc["max_lr"] == 4.0e-4
    assert tc["experiment_id"] == 7
    assert tc["checkpoint_dir"] == str(tmp_path / "ckpt")


def test_to_obs_cond_swaps_and_adds_channel():
    mod = _load_script_module()
    # sbibm_jax's collate already tokenizes theta with a trailing channel
    # axis (verified against the real gaussian_random_field loader), so the
    # fake batch here mirrors that real shape rather than a bare (4, 2).
    theta = np.zeros((4, 2, 1), dtype=np.float32)
    x = np.zeros((4, 8, 8, 1), dtype=np.float32)
    obs, cond = mod.to_obs_cond((theta, x))
    assert obs.shape == (4, 8, 8, 1)     # obs = the field, native shape
    assert cond.shape == (4, 2, 1)       # cond = theta, channel-carrying


def test_heldout_bits_per_dim_matches_gaussian_at_identity_init():
    # zero_init=True (TarFlowParams default) makes every MetaBlock the
    # identity, so the untrained flow is exactly N(0, I): bits/dim of
    # N(0,1) data must equal 0.5*log2(2*pi*e) ~= 2.047 up to MC error.
    mod = _load_script_module()
    from flax import nnx
    import jax
    flow = mod.build_flow(nnx.Rngs(0), _TINY_MODEL_CFG)
    k1, k2 = jax.random.split(jax.random.key(0))
    fields = np.asarray(jax.random.normal(k1, (256, 8, 8, 1)))
    theta = np.asarray(jax.random.normal(k2, (256, 2, 1)))
    got = mod.heldout_bits_per_dim(flow, fields, theta, batch_size=128)
    expected = 0.5 * np.log2(2 * np.pi * np.e)
    assert abs(got - expected) < 0.05


def test_radial_power_spectrum_white_noise_level_and_range():
    mod = _load_script_module()
    rng = np.random.default_rng(0)
    field = rng.normal(size=(128, 128))
    k, pk = mod.radial_power_spectrum(field)
    assert k.shape == pk.shape and len(k) > 10
    assert k.min() > 0 and k.max() <= 0.5      # cycles/pixel, Nyquist bound
    assert np.all(pk > 0)
    # white noise: E[P(k)] = sigma^2 = 1; high-k bins average many modes
    high = k > 0.1
    assert abs(pk[high].mean() - 1.0) < 0.1


def test_plot_helpers_write_files(tmp_path):
    mod = _load_script_module()
    rng = np.random.default_rng(0)
    truths = rng.normal(size=(2, 16, 16))
    samples = [rng.normal(size=(3, 16, 16)) for _ in range(2)]
    sim_fields = [rng.normal(size=(4, 16, 16)) for _ in range(2)]
    thetas = np.array([[0.0, 2.0], [0.5, 3.0]])
    p_grid = tmp_path / "grid.png"
    p_pk = tmp_path / "pk.png"
    p_loss = tmp_path / "loss.png"
    mod.plot_field_grid(truths, samples, thetas, n_show=3, path=str(p_grid))
    mod.plot_power_spectra(sim_fields, samples, thetas, field_size=16,
                           path=str(p_pk))
    mod.plot_losses(np.ones(10), np.ones(10), val_every=100, path=str(p_loss))
    assert p_grid.exists() and p_pk.exists() and p_loss.exists()


def test_main_exists_and_smoke_config_is_tiny():
    import yaml
    mod = _load_script_module()
    assert callable(mod.main)
    smoke = (_REPO_ROOT / "examples/sbi-benchmarks/gaussian_random_field"
             / "tarflow/config/config_smoke.yaml")
    with open(smoke) as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["nsteps"] <= 50          # cheap enough for CPU
    assert cfg["model"]["head_dim"] % 4 == 0        # rope requirement holds
    assert cfg["sampling"]["nsamples"] <= 4
