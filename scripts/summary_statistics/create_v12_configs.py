"""
create_v12_configs.py

Reads best_configurations.csv, and for each row:
1. Creates a v12/ config directory next to the existing version directories.
2. Copies the best config file there, renaming it to match the actual budget.
3. Edits the copy:
   - training.train_model  -> False
   - training.restore_model -> True
   - training.experiment_id -> 12
"""

import os
import shutil
import pandas as pd
import yaml
from tqdm import tqdm

CSV_PATH = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats/best_configurations.csv"
TARGET_VERSION = "v12"
EXPERIMENT_ID = 12

df = pd.read_csv(CSV_PATH)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    src_path = row["config_path"]
    budget = int(row["budget"])
    method = row["method"]

    if not os.path.isfile(src_path):
        tqdm.write(f"SKIP (not found): {src_path}")
        continue

    # Destination directory: .../config/v12/
    config_dir = os.path.dirname(os.path.dirname(src_path))  # .../config/
    dest_dir = os.path.join(config_dir, TARGET_VERSION)
    os.makedirs(dest_dir, exist_ok=True)

    # Build the canonical destination filename: config_<method>_sbibm_<budget>.yaml
    dest_filename = f"config_{method}_sbibm_{budget}.yaml"
    dest_path = os.path.join(dest_dir, dest_filename)

    # Copy
    shutil.copy2(src_path, dest_path)

    # Patch
    with open(dest_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("training", {})
    cfg["training"]["train_model"] = False
    cfg["training"]["restore_model"] = True
    cfg["training"]["experiment_id"] = EXPERIMENT_ID
    cfg["training"]["early_stopping"] = True
    cfg["training"]["val_error_ratio"] = 1.3

    with open(dest_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # --- Copy checkpoints ---
    # Source version number (strip leading 'v')
    src_version_num = row["best_version"].lstrip("v")

    # Infer the method root: .../examples/sbi-benchmarks/<task>/<method>/
    # src_path = .../config/v<N>/config_xxx.yaml → 3 levels up
    method_root = os.path.dirname(os.path.dirname(os.path.dirname(src_path)))
    ckpt_base = os.path.join(method_root, "sbibm", str(budget), "checkpoints")

    # Regular checkpoint: checkpoints/<N>/ -> checkpoints/12/
    src_ckpt = os.path.join(ckpt_base, src_version_num)
    dst_ckpt = os.path.join(ckpt_base, str(EXPERIMENT_ID))
    if os.path.isdir(src_ckpt):
        shutil.copytree(src_ckpt, dst_ckpt, dirs_exist_ok=True)
        print(f"  ckpt  {src_ckpt} -> {dst_ckpt}")
    else:
        print(f"  WARN  checkpoint not found: {src_ckpt}")

    # EMA checkpoint: checkpoints/ema/<N>/ -> checkpoints/ema/12/
    src_ema = os.path.join(ckpt_base, "ema", src_version_num)
    dst_ema = os.path.join(ckpt_base, "ema", str(EXPERIMENT_ID))
    if os.path.isdir(src_ema):
        shutil.copytree(src_ema, dst_ema, dirs_exist_ok=True)
        print(f"  ema   {src_ema} -> {dst_ema}")
    else:
        print(f"  WARN  ema checkpoint not found: {src_ema}")

    print(f"OK  {dest_path}")

print("\nDone.")
