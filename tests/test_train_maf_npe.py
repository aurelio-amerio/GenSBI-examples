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
    assert cfg["training"]["nsamples"] < 100_000  # default get_train_dataset cap
    assert cfg["evaluation"]["grid_size"] > 0
