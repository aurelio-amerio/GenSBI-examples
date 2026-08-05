#!/bin/bash
# HTCondor executable wrapper for the spherical GRF flow-matching example.
#
#   $1 = example directory, $2 = config file name (inside $1/config/),
#   $3 = training script name (optional, default train-spherical-grf.py)
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main
# process, cpu for spawned grain workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training. Uses the python on PATH (getenv=True passes the submitter's
# environment) — activate an env with gensbi + heal-swin-nnx + sbibm-jax
# before condor_submit.
unset JAX_PLATFORMS

# All spherical_grf results (runs 1-7) train on the TEST dataset repo
# (aurelio-amerio/SBI-benchmarks-test, x_std 129.503). Export explicitly:
# earlier runs inherited this from the interactive submit shell via
# getenv=True, which silently flips to the production repo (different
# normalization stats) when submitting from a shell without it.
export SBIBM_JAX_USE_TEST=1

eval "$(conda shell.bash hook)"
conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi

exec python "${3:-train-spherical-grf.py}" --config "config/$2"
