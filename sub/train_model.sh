#!/bin/bash
cd $1
eval "$(conda shell.bash hook)"
conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi
python $2

exit