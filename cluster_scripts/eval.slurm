#!/bin/bash
#SBATCH --job-name=chess_eval
#SBATCH --output=./cluster_scripts/logs/eval_results.txt
#SBATCH -p stampede
#SBATCH -N 1 ## Number of nodes
#SBATCH -n 1 ## Number of cores per node

source ~/.bashrc
conda activate research

python evaluate_chess_agents.py