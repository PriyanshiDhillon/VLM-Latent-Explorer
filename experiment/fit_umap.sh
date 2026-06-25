#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=FitUMAP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=fit_umap.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate vlm-latent

cd "$HOME/VLM-Latent-Explorer-Davide/experiment"
python fit_umap.py
