#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallData
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=install_data.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/VLM-Latent-Explorer-Davide/data"

python -m pip install --user -U datasets huggingface_hub pillow

python prepare_subset.py
