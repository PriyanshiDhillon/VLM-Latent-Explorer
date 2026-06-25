#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ExtractTokens
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=extract_token_cache.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate vlm-latent

cd "$HOME/VLM-Latent-Explorer-Davide/experiment"

# Start small. After this succeeds, increase --limit or use --model all.
python extract_token_cache.py --model all --limit 50 --max-new-tokens 128 --overwrite
