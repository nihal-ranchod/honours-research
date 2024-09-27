#!/bin/bash
#SBATCH --job-name=mcts-train
#SBATCH --output=./cluster_scripts/logs/mcts_results.txt
#SBATCH -p stampede
#SBATCH -N 1 ## Number of nodes
#SBATCH -n 1 ## Number of cores per node

source ~/.bashrc
conda activate research

python mcts_compare.py