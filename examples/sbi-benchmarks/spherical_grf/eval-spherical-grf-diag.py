# -*- coding: utf-8 -*-
"""Eval-only sampler diagnostic for the spherical GRF flow-matching model (exp-7).

Design: docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md

Restores the frozen exp-7 EMA checkpoint read-only (never trains) and runs
either the sampler-variant matrix (ODE step-size / adaptive-solver
comparisons, ``diagnostic.mode: ode`` or ``sde``) or a TARP coverage check
(``diagnostic.mode: tarp_ode`` / ``tarp_sde``) against it, per the run
config's ``diagnostic`` section.

Everything else matches the training script this was copied from: the
HealSwin encoder (with its global bottleneck-attention stage) compresses
each nside-64 map to the bottleneck grid, which conditions a gensbi Flux1
flow-matching model, via spherical HEALPix RoPE ids (see COND_ID_KIND),
over the 3-dim posterior (logA, n, alpha) of the sbibm-jax `spherical_grf`
task. Data comes from the published HF dataset (offline TaskDataset, NEST
ordering, Hub normalization stats; first use downloads ~24 GB into the HF
cache). TARP pairs are simulated fresh via the task's healpy simulator.

Run headless. The script defaults to the GPU (``JAX_PLATFORMS=cuda``) and
will fail fast on a machine with no CUDA device.

    python eval-spherical-grf-diag.py --config config/config_healpix_7diag_ode.yaml

Debug modes (both CPU-safe, both accept --config):

    SMOKE=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py
        forward-shape check, no data, no training
    QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py
        tiny end-to-end run (few steps, few samples)
"""

from __future__ import annotations

import os
import sys

# Any spawned worker re-importing this module must never grab the GPU; the
# main process defaults to CUDA (an explicit JAX_PLATFORMS from the caller
# still wins via setdefault). Same pattern as mnist_healpix_classify.py.
if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import time
from math import isqrt

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention

from absl import flags

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corner import corner

import argparse

import yaml

from gensbi.core import FlowMatchingMethod
from gensbi.flow_matching.solver import NonSingularSolver, ZeroEndsSolver
from gensbi.models import Flux1, Flux1Params, HealSwinEncoder, HealSwinParams
from gensbi.models.flux1.layers import EmbedND
from gensbi.models.flux1.math import apply_rope
from gensbi.recipes import ConditionalPipeline, HealpixRope
from gensbi.recipes.flux1 import parse_flux1_params
from gensbi.recipes.utils import init_ids_1d, parse_training_config
from gensbi.utils.plotting import plot_marginals
from gensbi.diagnostics import run_tarp, plot_tarp
from gensbi.diagnostics.metrics import c2st

from sbibm_jax.tasks import get_task
from sbibm_jax.data import TaskDataset

# grain's mp_prefetch reads absl flags; parse argv once so a plain
# `python ...` run doesn't hit UnparsedFlagAccessError on first prefetch.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)

QUICK = os.environ.get("QUICK") == "1"

# --- configuration (edit the YAML files, not this block) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_parser = argparse.ArgumentParser(description="Spherical GRF flow-matching NPE")
_parser.add_argument(
    "--config", default=os.path.join("config", "config_healpix.yaml"),
    help="YAML run config: config/config_healpix.yaml or config/config_pos1d.yaml",
)
_args, _ = _parser.parse_known_args()
CONFIG_PATH = (_args.config if os.path.isabs(_args.config)
               else os.path.join(BASE_DIR, _args.config))

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


def _deep_merge(base, overlay):
    """Overlay wins; dicts merge recursively, everything else replaces."""
    out = dict(base)
    for k, v in overlay.items():
        out[k] = (_deep_merge(out[k], v)
                  if isinstance(v, dict) and isinstance(out.get(k), dict)
                  else v)
    return out


if "base_config" in CONFIG:
    _base_rel = CONFIG.pop("base_config")
    _base_path = (_base_rel if os.path.isabs(_base_rel)
                  else os.path.join(BASE_DIR, _base_rel))
    with open(_base_path) as f:
        CONFIG = _deep_merge(yaml.safe_load(f), CONFIG)
    # parse_flux1_params / parse_training_config re-read the YAML from a
    # path; point them at the merged view via a temp file.
    import tempfile
    _merged = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="7diag_merged_", delete=False)
    yaml.safe_dump(CONFIG, _merged)
    _merged.close()
    CONFIG_PATH = _merged.name

# task / data
NSIDE = CONFIG["data"]["nside"]
NPIX = 12 * NSIDE ** 2
SEED = CONFIG["data"]["seed"]
DIM_THETA = 3
THETA_LABELS = (r"$\log A$", r"$n$", r"$\alpha$")

# HealSwin encoder geometry (bottleneck derived, never written twice)
_ENC = CONFIG["encoder"]
EMBED_DIM = _ENC["embed_dim"]
DEPTHS = tuple(_ENC["depths"])
ENC_NUM_HEADS = tuple(_ENC["num_heads"])
WINDOW_SIZE = _ENC["window_size"]
PATCH_SIZE = _ENC.get("patch_size", 4)  # power of four; 1 = no regrouping
# patch embed divides nside by sqrt(patch_size), each merging halves it
NSIDE_BOTTLENECK = NSIDE // (isqrt(PATCH_SIZE) * 2 ** (len(DEPTHS) - 1))
COND_TOKENS = 12 * NSIDE_BOTTLENECK ** 2
COND_FEATURES = EMBED_DIM * 2 ** (len(DEPTHS) - 1)

_DTYPES = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
ENC_DTYPE = _DTYPES[_ENC.get("dtype", "bfloat16")]
ENC_PARAM_DTYPE = _DTYPES[_ENC.get("param_dtype", "float32")]

# Raw-map power pass-through (exp-8b): 3-scale log local power of the input
# map, projected into the bottleneck tokens. Off unless the YAML switch is
# set, or the SMOKE-only env override is set (used by Step 1/3 of the v8
# power-passthrough task to exercise the code path without a config edit).
POWER_PASSTHROUGH = bool(CONFIG["encoder"].get("power_passthrough", False)
                         or os.environ.get("POWER_PASSTHROUGH_SMOKE"))

# Global bottleneck attention (v2): full-attention blocks over the bottleneck
# tokens, appended after the last (face-windowed) HealSwin stage.
GLOBAL_DEPTH = int(_ENC.get("global_depth", 0))
GLOBAL_NUM_HEADS = int(_ENC.get("global_num_heads", 8))
GLOBAL_AXES_DIM = tuple(_ENC.get("global_axes_dim", (22, 22, 20)))
if GLOBAL_DEPTH:
    assert COND_FEATURES % GLOBAL_NUM_HEADS == 0, (
        f"encoder features {COND_FEATURES} not divisible by "
        f"global_num_heads {GLOBAL_NUM_HEADS}")
    _GLOBAL_HEAD_DIM = COND_FEATURES // GLOBAL_NUM_HEADS
    assert sum(GLOBAL_AXES_DIM) == _GLOBAL_HEAD_DIM and not any(
        a % 2 for a in GLOBAL_AXES_DIM), (
        f"global_axes_dim {GLOBAL_AXES_DIM} must be even and sum to "
        f"head_dim {_GLOBAL_HEAD_DIM}")

# Flux1 model hyperparameters (model-side id_embedding_strategy lives here)
FLUX_PARAMS_DICT = parse_flux1_params(CONFIG_PATH)
assert FLUX_PARAMS_DICT["context_in_dim"] == COND_FEATURES, (
    f"model.context_in_dim={FLUX_PARAMS_DICT['context_in_dim']} must equal "
    f"encoder features {COND_FEATURES}")

# Pipeline-side cond-id builder: HealpixRope object or "pos1d" string.
COND_ID_KIND = CONFIG["ids"]["cond_kind"]
if COND_ID_KIND == "healpix-rope":
    COND_STRATEGY = HealpixRope(nside=NSIDE_BOTTLENECK)
    if FLUX_PARAMS_DICT["theta"] is None:
        FLUX_PARAMS_DICT["theta"] = COND_STRATEGY.theta
elif COND_ID_KIND == "pos1d":
    COND_STRATEGY = "pos1d"
else:
    raise ValueError(f"unknown ids.cond_kind: {COND_ID_KIND!r}")

# training / data loading
_TRAIN = CONFIG["training"]
BATCH_SIZE = 8 if QUICK else _TRAIN["batch_size"]
VAL_BATCH_SIZE = 8 if QUICK else _TRAIN["val_batch_size"]
NSTEPS = 5 if QUICK else _TRAIN["nsteps"]
NUM_WORKERS = 0 if QUICK else min(8, max(1, (os.cpu_count() or 2) - 2))
TRAIN_MODEL = _TRAIN["train_model"]
RESTORE_MODEL = _TRAIN["restore_model"]
VAL_EVERY = 2 if QUICK else _TRAIN["val_every"]

# evaluation
_EVAL = CONFIG["evaluation"]
EVAL_OBSERVATIONS = (1,) if QUICK else tuple(_EVAL["observations"])
NUM_POSTERIOR_SAMPLES = 64 if QUICK else _EVAL["num_posterior_samples"]
SAMPLE_STEP_SIZE = 0.25 if QUICK else _EVAL["sample_step_size"]
# Jitted-sampler memory scales with nsamples; chunk to fit the GPU. Default
# (no config key) = None = one device call, the pre-change behavior.
SAMPLE_CHUNK = 16 if QUICK else _EVAL.get("sample_chunk_size", None)
# tarp_pairs 0 disables the TARP diagnostic entirely (also in QUICK).
TARP_PAIRS = 0 if _EVAL["tarp_pairs"] == 0 else (2 if QUICK else _EVAL["tarp_pairs"])
TARP_POSTERIOR_SAMPLES = 8 if QUICK else _EVAL["tarp_posterior_samples"]

# diagnostic mode
_DIAG = CONFIG["diagnostic"]
MODE = _DIAG["mode"]                       # ode | sde | tarp_ode | tarp_sde
IS_TARP = MODE.startswith("tarp")
VARIANTS = tuple(_DIAG["variants"])        # dicts: name/kind[/step_size/alpha/baseline]
SCATTER_REPEATS = 0 if QUICK else int(_DIAG.get("scatter_repeats", 0))
SCATTER_OBS = tuple(_DIAG.get("scatter_observations", ()))
# raw-vs-EMA diagnostic: false samples the primary (non-EMA) weights.
# restore_model always restores both; this only selects which one samples.
USE_EMA = bool(_DIAG.get("use_ema", True))

# Outputs get the 7diag stem; the checkpoint dir is exp-7's own (read-only
# restore) and must NOT get the _quick suffix, or QUICK finds no checkpoint.
EXPERIMENT_ID = _TRAIN["run_name"] + "diag_" + MODE + ("_quick" if QUICK else "")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", _TRAIN["run_name"])
IMGS_DIR = os.path.join(BASE_DIR, "imgs")
RESULTS_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_ID}_results.txt")
# ------------------------------------------------------------------------


def make_encoder_params() -> HealSwinParams:
    return HealSwinParams(
        nside=NSIDE,
        in_channels=1,
        out_channels=1,  # required by the dataclass; unused by the encoder
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=ENC_NUM_HEADS,
        window_size=WINDOW_SIZE,
        patch_size=PATCH_SIZE,
        dtype=_ENC.get("dtype", "bfloat16"),
        param_dtype=_ENC.get("param_dtype", "float32"),
    )


def make_flux_params(rngs: nnx.Rngs) -> Flux1Params:
    return Flux1Params(
        rngs=rngs, dim_obs=DIM_THETA, dim_cond=COND_TOKENS, **FLUX_PARAMS_DICT)


def _make_rope_attention_fn(pe):
    """flax ``attention_fn`` that rotates q/k with a fixed RoPE table first.

    ``pe`` is the (1, 1, L, head_dim/2, 2, 2) table for the static bottleneck
    grid, closed over as a jit constant. flax hands q/k/v as (B, L, H, D);
    gensbi's ``apply_rope`` wants (B, H, L, D).
    """
    def rope_attention(query, key, value, **kwargs):
        q = jnp.swapaxes(query, 1, 2)
        k = jnp.swapaxes(key, 1, 2)
        q, k = apply_rope(q, k, pe)
        query = jnp.swapaxes(q, 1, 2).astype(value.dtype)
        key = jnp.swapaxes(k, 1, 2).astype(value.dtype)
        return dot_product_attention(query, key, value, **kwargs)
    return rope_attention


class GlobalTokenBlock(nnx.Module):
    """Pre-LN transformer block, full self-attention over all bottleneck tokens.

    Un-windowed dot-product attention with spherical HEALPix RoPE and
    QK-RMS normalization (``normalize_qk``), then a 4x GELU MLP. LayerNorms
    run in fp32 (the repo's fp32-island convention); everything else follows
    the encoder's compute/param dtypes.
    """

    def __init__(self, dim, num_heads, pe, *, dtype, param_dtype, rngs):
        self.norm1 = nnx.LayerNorm(
            dim, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim,
            dtype=dtype, param_dtype=param_dtype,
            use_bias=True, normalize_qk=True, decode=False,
            attention_fn=_make_rope_attention_fn(pe), rngs=rngs)
        self.norm2 = nnx.LayerNorm(
            dim, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.fc1 = nnx.Linear(
            dim, 4 * dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.fc2 = nnx.Linear(
            4 * dim, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(nnx.gelu(self.fc1(self.norm2(x))))


class SphericalGRFModel(nnx.Module):
    """HealSwin encoder + global bottleneck attention feeding Flux1."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = HealSwinEncoder(make_encoder_params(), rngs=rngs)
        assert self.encoder.num_features == COND_FEATURES
        blocks = []
        if GLOBAL_DEPTH:
            grope = HealpixRope(nside=NSIDE_BOTTLENECK)
            ids, _ = grope.build(COND_TOKENS)  # (1, COND_TOKENS, 3) float32
            pe = EmbedND(dim=COND_FEATURES // GLOBAL_NUM_HEADS,
                         theta=grope.theta,
                         axes_dim=list(GLOBAL_AXES_DIM))(ids)
            blocks = [
                GlobalTokenBlock(
                    COND_FEATURES, GLOBAL_NUM_HEADS, pe,
                    dtype=ENC_DTYPE, param_dtype=ENC_PARAM_DTYPE, rngs=rngs)
                for _ in range(GLOBAL_DEPTH)]
        self.global_blocks = nnx.List(blocks)

        # Raw-map power pass-through (exp-8b): the GRF sufficient statistic
        # is quadratic in the map; hand Flux1 local log-band-powers at three
        # scales so amplitude (logA) has a linear channel past the LayerNorms.
        # Zero-init => exactly the exp-7 function at init.
        self.power_proj = nnx.data(None)
        if POWER_PASSTHROUGH:
            self.power_proj = nnx.Linear(
                3, COND_FEATURES,
                kernel_init=nnx.initializers.zeros_init(),
                dtype=ENC_DTYPE, param_dtype=ENC_PARAM_DTYPE, rngs=rngs)
        self.flux = Flux1(make_flux_params(rngs))

    def _power_tokens(self, cond):
        """(B, NPIX, 1) NEST map -> (B, 192, 3) log local power at
        nside 16/8/4, on the nside-4 (bottleneck-token) grid."""
        x2 = jnp.square(cond[..., 0].astype(jnp.float32))    # (B, NPIX)
        b = x2.shape[0]
        feats = []
        for s in (16, 8, 4):                 # children per nside-4 cell: 16,4,1
            p = x2.reshape(b, 12 * s * s, -1).mean(-1)       # (B, 12 s^2)
            f = jnp.log(p + 1e-12)
            feats.append(f.reshape(b, 192, -1).mean(-1))     # (B, 192)
        return jnp.stack(feats, axis=-1)                     # (B, 192, 3)

    def __call__(self, t, obs, obs_ids, cond, cond_ids,
                 conditioned=True, guidance=None, **kwargs):
        tokens, _skips = self.encoder(cond)  # (B, NPIX, 1) -> (B, COND_TOKENS, COND_FEATURES)
        for blk in self.global_blocks:
            tokens = blk(tokens)
        if self.power_proj is not None:
            tokens = tokens + self.power_proj(
                self._power_tokens(cond).astype(ENC_DTYPE))
        return self.flux(t=t, obs=obs, obs_ids=obs_ids, cond=tokens,
                         cond_ids=cond_ids, conditioned=conditioned,
                         guidance=guidance)


def make_datasets():
    """Offline HF TaskDataset + train/val loaders of normalized NEST (theta, x) batches.

    First use downloads the spherical_grf config (~24 GB) into the HF cache
    (respects HF_HOME); afterwards everything is served locally.
    Normalization stats come from the published Hub metadata.
    """
    ds = TaskDataset(
        "spherical_grf", ordering="nest", normalize=True, dtype=np.float32,
        seed=SEED, max_workers=NUM_WORKERS,
    )
    train_loader = ds.get_train_loader(BATCH_SIZE)
    # The pipeline draws one fixed val batch at train start.
    val_loader = ds.get_val_loader(VAL_BATCH_SIZE)
    return ds, train_loader, val_loader


def prep_x(x_ring, ds):
    """Raw RING maps (B, NPIX) -> normalized NEST tokens (B, NPIX, 1).

    Mirrors the training collate exactly: permute, tokenize, normalize.
    """
    x = np.asarray(x_ring)[:, ds._x_perm][..., None]
    return jnp.asarray(ds.normalize_x(x), dtype=jnp.float32)


def make_training_config():
    cfg = ConditionalPipeline.get_default_training_config()
    cfg.update(parse_training_config(CONFIG_PATH))
    cfg["checkpoint_dir"] = CHECKPOINT_DIR
    if QUICK:
        cfg["nsteps"] = NSTEPS
        cfg["warmup_steps"] = 2
        cfg["val_every"] = 2
        cfg["decay_transition"] = 0
    return cfg


def make_pipeline(model, train_loader, val_loader):
    # Cond ids are first-class now: pass "pos1d" or a HealpixRope IdStrategy
    # straight to the pipeline. The model-side vocabulary (Flux1Params
    # id_embedding_strategy, e.g. ("absolute", "rope")) is configured
    # independently in the YAML `model:` section.
    return ConditionalPipeline(
        model, train_loader, val_loader,
        dim_obs=DIM_THETA, dim_cond=COND_TOKENS,
        method=FlowMatchingMethod(),
        ch_obs=1, ch_cond=COND_FEATURES,
        id_embedding_strategy=("absolute", COND_STRATEGY),
        training_config=make_training_config(),
    )


def sampler_kwargs(v):
    """YAML variant dict -> kwargs for pipeline.sample / sample_batched.

    Forwarded through get_sampler to FlowMatchingMethod.build_sampler_fn.
    QUICK coarsens fixed steps to 0.25 (4 steps) like the training script.
    """
    kind = v["kind"]
    step = v.get("step_size")
    if QUICK and step is not None:
        step = 0.25
    if kind == "euler":
        return {"step_size": step, "method": "Euler"}
    if kind == "dopri5":
        return {"step_size": None, "method": "Dopri5"}  # adaptive; atol=rtol=1e-5
    if kind == "nonsingular":
        return {"step_size": step,
                "solver": (NonSingularSolver, {"alpha": float(v["alpha"])})}
    if kind == "zeroends":
        return {"step_size": step,
                "solver": (ZeroEndsSolver, {"alpha": float(v["alpha"])})}
    raise ValueError(f"unknown diagnostic variant kind: {kind!r}")


def evaluate_matrix(pipeline, ds, log):
    """Sampler-variant matrix: posterior vs reference for every observation.

    Per variant x obs: mean/std vs reference, C2ST. Baseline-flagged
    variants get SCATTER_REPEATS extra independent draws on SCATTER_OBS
    (C2ST mean +/- std prices the run-to-run scatter). Overlay corners for
    obs 1-3 only, every variant.
    """
    key = jax.random.PRNGKey(SEED + 7)
    labels = list(THETA_LABELS)

    for v in VARIANTS:
        kw = sampler_kwargs(v)
        vname = v["name"]
        log(f"variant {vname}: {kw}")
        for i in EVAL_OBSERVATIONS:
            x_raw, ref = ds.get_reference(i)                 # RING map, (S, 3)
            x_o = prep_x(np.asarray(x_raw).reshape(1, -1), ds)
            theta_true = np.asarray(ds.get_true_parameters(i)).reshape(-1)
            ref = np.asarray(ref)

            ndraws = 1 + (SCATTER_REPEATS
                          if v.get("baseline") and i in SCATTER_OBS else 0)
            c2sts, first_flow, n_c2st = [], None, 0
            for rep in range(ndraws):
                t0 = time.time()
                key, sk = jax.random.split(key)
                samples = np.asarray(pipeline.sample(
                    sk, x_o, NUM_POSTERIOR_SAMPLES,
                    chunk_size=SAMPLE_CHUNK, use_ema=USE_EMA, **kw))
                flow = np.asarray(ds.unnormalize_theta(samples))[:, :, 0]
                n_c2st = int(min(flow.shape[0], ref.shape[0]))
                _rng = np.random.default_rng(SEED + 1000 * rep + i)
                f_c = flow[_rng.choice(flow.shape[0], n_c2st, replace=False)]
                r_c = ref[_rng.choice(ref.shape[0], n_c2st, replace=False)]
                c2sts.append(float(c2st(jnp.asarray(f_c), jnp.asarray(r_c),
                                        seed=SEED + rep)))
                if rep == 0:
                    first_flow = flow
                    log(f"{vname} obs {i}: {flow.shape[0]} samples in "
                        f"{time.time() - t0:.0f}s | "
                        f"true {np.array2string(theta_true, precision=3)} | "
                        f"flow mean {np.array2string(flow.mean(0), precision=3)} "
                        f"std {np.array2string(flow.std(0), precision=3)} | "
                        f"ref mean {np.array2string(ref.mean(0), precision=3)} "
                        f"std {np.array2string(ref.std(0), precision=3)}")

            if ndraws > 1:
                log(f"{vname} obs {i}: C2ST = {np.mean(c2sts):.3f} +/- "
                    f"{np.std(c2sts):.3f} over {ndraws} draws "
                    f"{np.array2string(np.asarray(c2sts), precision=3)}")
            else:
                log(f"{vname} obs {i}: C2ST(flow vs ref) = {c2sts[0]:.3f} "
                    f"over {n_c2st} samples (0.5=indistinguishable, "
                    f"1.0=separable)")

            if i in (1, 2, 3):
                fig = corner(ref, labels=labels, truths=list(theta_true),
                             color="C0", hist_kwargs={"density": True},
                             plot_contours=not QUICK, plot_density=not QUICK)
                corner(first_flow, fig=fig, color="C1",
                       hist_kwargs={"density": True},
                       plot_contours=not QUICK, plot_density=not QUICK)
                fig.suptitle(f"obs {i} [{vname}]: reference (blue) vs "
                             f"flow (orange)")
                fig.savefig(os.path.join(
                    IMGS_DIR, f"{EXPERIMENT_ID}_{vname}_overlay_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
                plt.close(fig)


def tarp_diagnostic(pipeline, ds, log, key):
    """TARP coverage on freshly simulated pairs (normalized theta space)."""
    v = VARIANTS[0]
    kw = sampler_kwargs(v)
    log(f"TARP variant {v['name']}: {kw}")
    task = get_task("spherical_grf")
    kt, ks, kp = jax.random.split(key, 3)
    sim = task.get_simulator(jax.random.PRNGKey(SEED + 300))
    theta = np.asarray(task.get_prior(kt, TARP_PAIRS))       # (P, 3)
    t0 = time.time()
    x = np.asarray(sim(ks, jnp.asarray(theta)))               # (P, NPIX) RING
    x_tok = prep_x(x, ds)
    post = pipeline.sample_batched(
        kp, x_tok, TARP_POSTERIOR_SAMPLES, chunk_size=20, use_ema=USE_EMA, **kw
    )
    post = np.asarray(post)[:, :, :, 0]                       # (S, P, 3)
    # ds theta stats are tokenized (1, 3, 1): tokenize, normalize, un-tokenize.
    theta_norm = np.asarray(ds.normalize_theta(theta[..., None]))[:, :, 0]
    res = run_tarp(jnp.asarray(theta_norm), jnp.asarray(post), bootstrap=False)
    plot_tarp(res, mode="both")
    plt.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_tarp.png"),
                dpi=100, bbox_inches="tight")
    plt.close("all")
    log(f"TARP: {TARP_PAIRS} pairs x {TARP_POSTERIOR_SAMPLES} samples "
        f"in {time.time() - t0:.0f}s -> {EXPERIMENT_ID}_tarp.png")


def main():
    os.makedirs(IMGS_DIR, exist_ok=True)
    results_file = open(RESULTS_FILE, "w")

    def log(line):
        print(line, flush=True)
        results_file.write(line + "\n")
        results_file.flush()

    log(f"quick={QUICK} batch={BATCH_SIZE} nsteps={NSTEPS} workers={NUM_WORKERS} "
        f"nside={NSIDE} embed_dim={EMBED_DIM} depths={DEPTHS} window={WINDOW_SIZE} "
        f"patch={PATCH_SIZE} cond={COND_TOKENS}x{COND_FEATURES} "
        f"global={GLOBAL_DEPTH}x{GLOBAL_NUM_HEADS}h "
        f"enc_dtype={_ENC.get('dtype', 'bfloat16')}/{_ENC.get('param_dtype', 'float32')} "
        f"flux_dtype={FLUX_PARAMS_DICT['dtype'].__name__}"
        f"/{FLUX_PARAMS_DICT['param_dtype'].__name__} "
        f"flux={FLUX_PARAMS_DICT['depth']}d+{FLUX_PARAMS_DICT['depth_single_blocks']}s "
        f"heads={FLUX_PARAMS_DICT['num_heads']} "
        f"ids={FLUX_PARAMS_DICT['id_embedding_strategy']} cond_ids={COND_ID_KIND}")

    t0 = time.time()
    ds, train_loader, val_loader = make_datasets()
    log(f"data: HF {ds.repo}/spherical_grf, Hub stats "
        f"x_mean={float(np.ravel(ds.x_mean)[0]):.6g} "
        f"x_std={float(np.ravel(ds.x_std)[0]):.6g} ({time.time() - t0:.1f}s)")

    model = SphericalGRFModel(rngs=nnx.Rngs(SEED))
    pipeline = make_pipeline(model, train_loader, val_loader)

    assert RESTORE_MODEL and not TRAIN_MODEL, (
        "7diag is eval-only: set training.train_model false / restore_model true")
    t0 = time.time()
    pipeline.restore_model()
    pipeline._wrap_model()
    log(f"restored exp-7 checkpoint from {CHECKPOINT_DIR} "
        f"(step {make_training_config()['experiment_id']}) in {time.time() - t0:.0f}s; "
        f"sampling with {'EMA' if USE_EMA else 'RAW (non-EMA)'} weights")

    if IS_TARP:
        if TARP_PAIRS:
            tarp_diagnostic(pipeline, ds, log, jax.random.PRNGKey(SEED + 7))
        else:
            log("TARP: skipped (evaluation.tarp_pairs = 0)")
    else:
        evaluate_matrix(pipeline, ds, log)
    results_file.close()


if __name__ == "__main__" and os.environ.get("SMOKE") != "1":
    main()

if __name__ == "__main__" and os.environ.get("SMOKE") == "1":
    # Forward-shape smoke check: no data, no training; runs on CPU.
    model = SphericalGRFModel(rngs=nnx.Rngs(0))
    model.eval()
    B = 2
    obs_ids, _ = init_ids_1d(DIM_THETA, 0)  # (1, 3, 2) — broadcast over batch
    print("cond tokens:", COND_TOKENS, "features:", COND_FEATURES,
          "global blocks:", GLOBAL_DEPTH)
    print(f"power passthrough: {'ON' if POWER_PASSTHROUGH else 'off'}")
    if COND_ID_KIND == "healpix-rope":
        cond_ids, _ = COND_STRATEGY.build(COND_TOKENS)  # (1, COND_TOKENS, 3) float32
    else:
        cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, COND_TOKENS, 2)
    v = model(
        t=jnp.full((B,), 0.5),
        obs=jnp.zeros((B, DIM_THETA, 1)),
        obs_ids=obs_ids,
        cond=jnp.zeros((B, NPIX, 1)),
        cond_ids=cond_ids,
    )
    print("vector field shape:", v.shape)
    assert v.shape == (B, DIM_THETA, 1)
    print("forward smoke check OK")
