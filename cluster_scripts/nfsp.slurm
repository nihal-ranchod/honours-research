#!/bin/bash
#SBATCH --job-name=nfsp_train
#SBATCH --output=./cluster_scripts/logs/nfsp_results.txt
#SBATCH -p stampede
#SBATCH -N 1 ## Number of nodes
#SBATCH -n 1 ## Number of cores per node

source ~/.bashrc
conda activate research

python play_chess.py --train_nfsp=True