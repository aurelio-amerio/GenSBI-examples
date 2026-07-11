#!/bin/bash
cd $1
eval "$(conda shell.bash hook)"
conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi
python $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} 

exit