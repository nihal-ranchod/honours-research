#!/bin/bash
#SBATCH --job-name=nfsp_train
#SBATCH --output=./cluster_scripts/logs/nfsp_results.txt
#SBATCH -p stampede
#SBATCH -N 1 ## Number of nodes
#SBATCH -n 1 ## Number of cores per node

source ~/.bashrc
conda activate research

python3 nfsp_algorithm_training.py --pgn_path "PGN_Data/lichess_db_standard_rated_2013-01.pgn" --model_type standard --batch_size 64 --save_path nfsp_standard_model.pt

