#!/bin/bash -l
#SBATCH --mail-type=ALL
#SBATCH --partition=compute_full_node
#SBATCH -p compute_full_node
#SBATCH --account=def-zhansu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH -t 00:20:00
#SBATCH -J training
cd $SLURM_SUBMIT_DIR

source ~/.virtualenvs/JC_Ocean/bin/activate

python run.py

