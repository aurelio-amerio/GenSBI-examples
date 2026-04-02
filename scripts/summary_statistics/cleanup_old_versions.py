"""
cleanup_old_versions.py

Deletes all non-v12 config directories, budget-specific top-level config yamls,
and non-v12 checkpoint/ema directories across all sbibm tasks and methods.

What gets DELETED:
  - config/v1/ through config/v11/ (and any other v* that isn't v12)
  - config/config_<method>_sbibm_<budget>.yaml  (budget-specific top-level yamls)
  - sbibm/<budget>/checkpoints/<N>/  for N != 12
  - sbibm/<budget>/checkpoints/ema/<N>/  for N != 12

What gets KEPT:
  - config/v12/
  - config/config_<method>.yaml  (the base config, no budget suffix)
  - sbibm/<budget>/checkpoints/12/
  - sbibm/<budget>/checkpoints/ema/12/
  - sbibm/<budget>/c2st_results/ (untouched)

Usage:
  python cleanup_old_versions.py          # dry run (shows what would be deleted)
  python cleanup_old_versions.py --delete # actually delete
"""

import os
import re
import shutil
import glob
import argparse
from tqdm import tqdm

BASE_DIR = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks"
KEEP_VERSION = "v12"
KEEP_VERSION_NUM = "12"

TASKS = ["two_moons", "bernoulli_glm", "gaussian_linear", "gaussian_mixture", "slcp"]
METHODS = [
    "flow_flux", "diffusion_flux", "score_matching_flux",
    "flow_flux1joint", "diffusion_flux1joint", "score_matching_flux1joint",
]
BUDGETS = ["10000", "30000", "100000"]


def collect_deletions():
    """Collect all paths to delete. Returns (dirs_to_delete, files_to_delete)."""
    dirs_to_delete = []
    files_to_delete = []

    for task in TASKS:
        for method in METHODS:
            method_root = os.path.join(BASE_DIR, task, method)
            if not os.path.isdir(method_root):
                continue

            config_dir = os.path.join(method_root, "config")
            if os.path.isdir(config_dir):
                # 1) Delete v* directories that aren't v12
                for entry in sorted(os.listdir(config_dir)):
                    entry_path = os.path.join(config_dir, entry)
                    if os.path.isdir(entry_path) and re.match(r"^v\d+$", entry) and entry != KEEP_VERSION:
                        dirs_to_delete.append(entry_path)

                # 2) Delete budget-specific top-level config yamls (contain _sbibm_ in name)
                for f in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
                    if "_sbibm_" in os.path.basename(f):
                        files_to_delete.append(f)

            # 3) Delete non-v12 checkpoint and ema dirs
            for budget in BUDGETS:
                ckpt_dir = os.path.join(method_root, "sbibm", budget, "checkpoints")
                if not os.path.isdir(ckpt_dir):
                    continue

                for entry in sorted(os.listdir(ckpt_dir)):
                    entry_path = os.path.join(ckpt_dir, entry)
                    if entry == "ema":
                        # Handle ema subdirectories
                        ema_dir = entry_path
                        if os.path.isdir(ema_dir):
                            for ema_entry in sorted(os.listdir(ema_dir)):
                                ema_entry_path = os.path.join(ema_dir, ema_entry)
                                if os.path.isdir(ema_entry_path) and ema_entry != KEEP_VERSION_NUM:
                                    dirs_to_delete.append(ema_entry_path)
                    elif os.path.isdir(entry_path) and entry != KEEP_VERSION_NUM:
                        dirs_to_delete.append(entry_path)

    return dirs_to_delete, files_to_delete


def main():
    parser = argparse.ArgumentParser(description="Clean up old config/checkpoint versions")
    parser.add_argument("--delete", action="store_true", help="Actually delete (default is dry run)")
    args = parser.parse_args()

    dirs_to_delete, files_to_delete = collect_deletions()

    print(f"{'=' * 60}")
    print(f"{'DRY RUN' if not args.delete else 'DELETING'}")
    print(f"  Directories to delete: {len(dirs_to_delete)}")
    print(f"  Files to delete:       {len(files_to_delete)}")
    print(f"{'=' * 60}\n")

    if not args.delete:
        print("--- Directories ---")
        for d in dirs_to_delete:
            print(f"  [DIR]  {d}")
        print(f"\n--- Files ---")
        for f in files_to_delete:
            print(f"  [FILE] {f}")
        print(f"\nRe-run with --delete to actually delete these.")
        return

    # Actually delete
    for d in tqdm(dirs_to_delete, desc="Deleting dirs"):
        shutil.rmtree(d)

    for f in tqdm(files_to_delete, desc="Deleting files"):
        os.remove(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
