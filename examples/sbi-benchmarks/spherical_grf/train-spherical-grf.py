# -*- coding: utf-8 -*-
"""Flow-matching NPE on the spherical GRF task with a HealSwin encoder.

A HEALPix-native Swin encoder compresses each nside-64 spherical map to a
bottleneck token grid (config-derived; 768 tokens x 512 features at nside 8
for config_healpix.yaml), which conditions a gensbi Flux1
flow-matching model, via spherical HEALPix RoPE ids (see COND_ID_KIND),
over the 3-dim posterior (logA, n, alpha) of the
sbibm-jax `spherical_grf` task. Training data streams from the published
HF dataset (offline TaskDataset, NEST ordering, Hub normalization stats;
first use downloads ~24 GB into the HF cache). TARP pairs are still
simulated fresh via the task's healpy simulator.

Run headless. The script defaults to the GPU (``JAX_PLATFORMS=cuda``) and
will fail fast on a machine with no CUDA device.

    python train-spherical-grf.py --config config/config_healpix.yaml

Or submit to a GPU node: ``condor_submit sub/spherical_grf.sub``
(edit `config = ...` in the sub file to pick the A/B arm).

Debug modes (both CPU-safe, both accept --config):

    SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py
        forward-shape check, no data, no training
    QUICK=1 JAX_PLATFORMS=cpu python train-spherical-grf.py
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

from absl import flags

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corner import corner

import argparse

import yaml

from gensbi.core import FlowMatchingMethod
from gensbi.models import Flux1, Flux1Params, HealSwinEncoder, HealSwinParams
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
TARP_PAIRS = 2 if QUICK else _EVAL["tarp_pairs"]
TARP_POSTERIOR_SAMPLES = 8 if QUICK else _EVAL["tarp_posterior_samples"]

EXPERIMENT_ID = _TRAIN["run_name"] + ("_quick" if QUICK else "")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_ID)
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


class SphericalGRFModel(nnx.Module):
    """HealSwin spherical encoder feeding Flux1's conditioning stream."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = HealSwinEncoder(make_encoder_params(), rngs=rngs)
        assert self.encoder.num_features == COND_FEATURES
        self.flux = Flux1(make_flux_params(rngs))

    def __call__(self, t, obs, obs_ids, cond, cond_ids,
                 conditioned=True, guidance=None, **kwargs):
        tokens, _skips = self.encoder(cond)  # (B, NPIX, 1) -> (B, COND_TOKENS, COND_FEATURES)
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


def evaluate(pipeline, ds, log):
    """Posterior vs reference for canonical observations, then TARP."""
    key = jax.random.PRNGKey(SEED + 7)
    labels = list(THETA_LABELS)

    for i in EVAL_OBSERVATIONS:
        x_raw, ref = ds.get_reference(i)                  # RING map, (S, 3) ref
        x_o = prep_x(np.asarray(x_raw).reshape(1, -1), ds)  # (1, NPIX, 1)
        theta_true = np.asarray(ds.get_true_parameters(i)).reshape(-1)
        ref = np.asarray(ref)
        t0 = time.time()
        # One condition -> sample; sample_batched is for batches of conditions.
        # chunk_size bounds device memory: 10k samples in one jitted batch
        # OOM'd a 40 GB A100 for the patch-1 models (52 GiB).
        key, sk = jax.random.split(key)
        samples = np.asarray(pipeline.sample(
            sk, x_o, NUM_POSTERIOR_SAMPLES, chunk_size=SAMPLE_CHUNK,
            step_size=SAMPLE_STEP_SIZE,
        ))
        # ds theta stats are tokenized (1, 3, 1); un-tokenize after unnorm.
        flow = np.asarray(ds.unnormalize_theta(samples))[:, :, 0]  # (S, 3)
        log(f"obs {i}: {flow.shape[0]} samples in {time.time() - t0:.0f}s | "
            f"true {np.array2string(theta_true, precision=3)} | "
            f"flow mean {np.array2string(flow.mean(0), precision=3)} "
            f"std {np.array2string(flow.std(0), precision=3)} | "
            f"ref mean {np.array2string(ref.mean(0), precision=3)} "
            f"std {np.array2string(ref.std(0), precision=3)}")

        # C2ST: classifier accuracy of flow vs reference posterior samples.
        # 0.5 = indistinguishable (ideal), 1.0 = perfectly separable. Balance
        # the two sets to equal size so the classifier baseline is 0.5.
        n_c2st = int(min(flow.shape[0], ref.shape[0]))
        _rng = np.random.default_rng(SEED + i)
        f_c = flow[_rng.choice(flow.shape[0], n_c2st, replace=False)]
        r_c = ref[_rng.choice(ref.shape[0], n_c2st, replace=False)]
        c2st_acc = float(c2st(jnp.asarray(f_c), jnp.asarray(r_c), seed=SEED))
        log(f"obs {i}: C2ST(flow vs ref) = {c2st_acc:.3f} over {n_c2st} "
            f"samples (0.5=indistinguishable, 1.0=separable)")

        # Overlay: reference (blue) under flow posterior (orange).
        fig = corner(ref, labels=labels, truths=list(theta_true), color="C0",
                     hist_kwargs={"density": True},
                     plot_contours=not QUICK, plot_density=not QUICK)
        corner(flow, fig=fig, color="C1", hist_kwargs={"density": True},
               plot_contours=not QUICK, plot_density=not QUICK)
        fig.suptitle(f"obs {i}: reference (blue) vs flow (orange)")
        fig.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_overlay_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Separate corners, in case the overlay hides one under the other.
        plot_marginals(ref, true_param=theta_true, labels=labels, gridsize=30)
        plt.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_reference_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close("all")
        plot_marginals(flow, true_param=theta_true, labels=labels, gridsize=30)
        plt.savefig(os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_flow_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close("all")

    tarp_diagnostic(pipeline, ds, log, key)


def tarp_diagnostic(pipeline, ds, log, key):
    """TARP coverage on freshly simulated pairs (normalized theta space)."""
    task = get_task("spherical_grf")
    kt, ks, kp = jax.random.split(key, 3)
    sim = task.get_simulator(jax.random.PRNGKey(SEED + 300))
    theta = np.asarray(task.get_prior(kt, TARP_PAIRS))       # (P, 3)
    t0 = time.time()
    x = np.asarray(sim(ks, jnp.asarray(theta)))               # (P, NPIX) RING
    x_tok = prep_x(x, ds)
    post = pipeline.sample_batched(
        kp, x_tok, TARP_POSTERIOR_SAMPLES, step_size=SAMPLE_STEP_SIZE, chunk_size=20
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


def save_loss_history(losses, val_losses, log):
    """Persist the loss history to .npz and plot train/val vs step.

    ``pipeline.train`` records one point every ``val_every`` steps: ``losses``
    is the 0.99-EMA of the batch loss, ``val_losses`` the validation loss.
    """
    losses = np.asarray(losses)
    val_losses = np.asarray(val_losses)
    steps = (np.arange(len(losses)) + 1) * VAL_EVERY
    npz_path = os.path.join(BASE_DIR, f"{EXPERIMENT_ID}_losses.npz")
    np.savez(npz_path, steps=steps, train_loss=losses, val_loss=val_losses)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, label="train (EMA)", color="C0")
    ax.plot(steps, val_losses, label="val", color="C1")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("flow-matching loss")
    ax.set_title(f"{EXPERIMENT_ID}: loss history")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    loss_png = os.path.join(IMGS_DIR, f"{EXPERIMENT_ID}_loss.png")
    fig.savefig(loss_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log(f"loss history: {len(losses)} points -> {os.path.basename(loss_png)}, "
        f"{os.path.basename(npz_path)}")


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

    if TRAIN_MODEL:
        t0 = time.time()
        losses, val_losses = pipeline.train(nnx.Rngs(SEED + 2), save_model=True)
        log(f"training: {len(losses)} steps in {time.time() - t0:.0f}s, "
            f"final train loss {float(losses[-1]):.4f}, "
            f"final val loss {float(val_losses[-1]):.4f}")
        save_loss_history(losses, val_losses, log)
    if RESTORE_MODEL:
        pipeline.restore_model()
        pipeline._wrap_model()

    evaluate(pipeline, ds, log)
    results_file.close()


if __name__ == "__main__" and os.environ.get("SMOKE") != "1":
    main()

if __name__ == "__main__" and os.environ.get("SMOKE") == "1":
    # Forward-shape smoke check: no data, no training; runs on CPU.
    model = SphericalGRFModel(rngs=nnx.Rngs(0))
    model.eval()
    B = 2
    obs_ids, _ = init_ids_1d(DIM_THETA, 0)  # (1, 3, 2) — broadcast over batch
    print("cond tokens:", COND_TOKENS, "features:", COND_FEATURES)
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
