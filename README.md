First setup environment
git clone git@github.com:PriyanshiDhillon/VLM-Latent-Explorer.git
cd ~/VLM-Latent-Explorer
sbatch environment.sh
sbatch data/data.sh

To run app:

srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=9 --time=01:00:00 --pty bash

This should allocate you some node (e.g. gcn54)
Once allocated do

module load 2025
module load Anaconda3/2025.06-1
cd ~/VLM-Latent-Explorer
python app.py

Within windows powershell (locally) run. Replace scur0239 with your own scur and gcn54 with the allocated node
ssh -L 8050:127.0.0.1:8050 -J scur0239@snellius.surf.nl scur0239@gcn54

Now it should show if you open it in browser on: localhost:8051
