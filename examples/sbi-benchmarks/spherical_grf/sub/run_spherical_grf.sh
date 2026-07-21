#!/bin/bash
# HTCondor executable wrapper for the spherical GRF flow-matching example.
#
#   $1 = example directory, $2 = config file name (inside $1/config/)
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main
# process, cpu for spawned grain workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training. Uses the python on PATH (getenv=True passes the submitter's
# environment) — activate an env with gensbi + heal-swin-nnx + sbibm-jax
# before condor_submit.
unset JAX_PLATFORMS

eval "$(conda shell.bash hook)"
conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi

exec python train-spherical-grf.py --config "config/$2"
