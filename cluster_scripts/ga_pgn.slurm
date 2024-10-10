#!/bin/bash
#SBATCH --job-name=GA-Training
#SBATCH --output=./cluster_scripts/logs/ga_standard_results.txt
#SBATCH -p stampede
#SBATCH -N 1 ## Number of nodes
#SBATCH -n 1 ## Number of cores per node

source ~/.bashrc
conda activate research

python play_chess_ga_past_pgn.py --train_ga=True