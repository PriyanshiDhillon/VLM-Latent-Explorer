## First setup environment

```bash
git clone git@github.com:PriyanshiDhillon/VLM-Latent-Explorer.git VLM-Latent-Explorer-Davide
cd ~/VLM-Latent-Explorer-Davide
sbatch environment.sh
sbatch data/data.sh
```

## Download the models

```bash
sbatch model/baseline.sh   # Qwen2.5-VL-7B-Instruct
sbatch model/model.sh      # Monet-7B + LVR-7B
```

This can take a while — the checkpoints are large. Wait for both jobs to finish before continuing.

## (Optional) Run a smoke test

Confirms the model + processor load correctly and can generate a response before running the full extraction:

```bash
sbatch experiment/baseline_smoke.sh
```

## Extract token activations (offline)

```bash
sbatch experiment/extract_token_cache.sh
```

This runs all three models over the data subset and saves per-token hidden states to:

```
precomputed/corpus_embeddings/{qwen,monet,lvr}/{example_id}.npz
```

Verify it worked before moving on:

```bash
ls precomputed/corpus_embeddings/qwen/
```

## Fit the UMAP manifold (offline)

This step builds the 2D UMAP projection used as the background scatter in the dashboard, and saves the fitted UMAP model so new queries can be projected onto it later.

```bash
sbatch experiment/fit_umap.sh
```

Verify it worked, you should see 6 new files:

```bash
ls precomputed/
# umap_qwen.pkl   umap_monet.pkl   umap_lvr.pkl
# corpus_2d_qwen.npz   corpus_2d_monet.npz   corpus_2d_lvr.npz
```

## To run the app

```bash
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=9 --time=01:00:00 --pty bash
```

This should allocate you some node (e.g. `gcn54`). Once allocated do:

```bash
cd ~/VLM-Latent-Explorer-Davide
module load 2025
module load Anaconda3/2025.06-1
source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate vlm-latent
python app.py
```

Monet and LVR use recurrent continuous hidden-state decoding automatically.
Monet generates 10 latent positions per span by default; override this with
`MONET_LATENT_SIZE`. LVR predicts its own end marker and uses
`LVR_MAX_LATENT_STEPS=64` as a safety limit.

After changing either latent decoder, regenerate the Monet and LVR caches with
`--overwrite`, then rerun `experiment/fit_umap.py`. The fitted projection uses
PCA followed by UMAP so live instances can be transformed without loading the
legacy ~1 GB UMAP artifacts.


## If browser doesn't work
Within Windows PowerShell (locally) run. Replace `scur0239` with your own scur and `gcn40` with the allocated node:

```bash
ssh -L 9001:127.0.0.1:9001 -J scur0265@snellius.surf.nl scur0265@gcn40
```

Now it should show if you open it in browser on: `localhost:9001`
