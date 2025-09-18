#!/bin/bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
eval "$(conda shell.bash hook)"
conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi
python scripts/train_sbi_model_simformer_v4.py --config $1

exit