#!/bin/bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
eval "$(conda shell.bash hook)"
conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi
python scripts/train_sbi_model_sbibm_v2.py --config $1 --dsize $2

exit