#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=BaselineSmoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=baseline_smoke.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/VLM-Latent-Explorer/experiment
python baseline_smoke.py