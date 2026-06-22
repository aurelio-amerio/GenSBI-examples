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
