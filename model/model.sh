#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=install_model.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate vlm-latent

cd $HOME/VLM-Latent-Explorer/model

hf download NOVAglow646/Monet-7B \
  --local-dir Monet-7B

hf download vincentleebang/LVR-7B \
  --local-dir LVR-7B