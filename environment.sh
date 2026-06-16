#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=install_env.out

module purge
module load 2025
module load Anaconda3/2025.06-1

conda create -n vlm-latent python=3.10 -y
source activate vlm-latent

cd ~/VLM-Latent-Explorer
pip install --upgrade pip
pip install -r requirements.txt