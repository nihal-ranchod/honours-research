#!/bin/bash
#SBATCH --job-name=GA-Training
#SBATCH --output=./cluster_scripts/logs/results.txt
#SBATCH -p stampede
#SBATCH -N 1 ## Number of nodes
#SBATCH -n 1 ## Number of cores per node

source ~/.bashrc
conda activate research

python genetic_algorithm_train.py