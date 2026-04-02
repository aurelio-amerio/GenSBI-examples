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
import re
import pandas as pd
import yaml

CSV_PATH = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats/best_configurations.csv"
TARGET_VERSION = "v12"
EXPERIMENT_ID = 12

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    src_path = row["config_path"]
    budget = int(row["budget"])
    method = row["method"]

    if not os.path.isfile(src_path):
        print(f"SKIP (not found): {src_path}")
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

    print(f"OK  {dest_path}")

print("\nDone.")
