# SatBae

## Training

```bash
# Basic training
./run_dynamicvis_training.sh

# Debug mode (2 epochs, small batch)
./run_dynamicvis_training.sh --debug

# Multi-GPU training
./run_dynamicvis_training.sh --gpus 4

# Custom configuration
./run_dynamicvis_training.sh --epochs 50 --batch-size 64 --lr 2e-4

# Disable wandb
./run_dynamicvis_training.sh --no-wandb

# Resume training
./run_dynamicvis_training.sh --resume
```

---

## Preprocessing: Patch Embedding & Clustering

The preprocessing pipeline extracts small patches from FMoW satellite images,
embeds them with a DINOv3 ViT-L/16 model, and clusters the embeddings for
downstream use by DynamicVis. It is split into two independent stages so you
can re-run clustering without re-embedding.

### Architecture

| File | Purpose |
|------|---------|
| `scripts/pipeline_utils.py` | Shared code: data classes, image loading, patching, DINOv3 model definition (with fp32 attention fix), embedding cache I/O, batched embedder, clustering/PCA/viz helpers |
| `scripts/embed_patches.py` | **Stage 1** — Load images → extract patches → embed with DINOv3 → cache per-cell `.npz` files to disk. Supports multi-GPU sharding. |
| `scripts/cluster_viz.py` | **Stage 2** — Load cached embeddings → PCA → cluster (KMeans / HDBSCAN / GMM) → generate visualizations and summary JSON. No GPU required. |
| `run_gpu_embed_patches.sh` | SLURM wrapper for Stage 1 (parallel embedding across MIG slices) |
| `run_gpu_cluster_viz.sh` | SLURM wrapper for Stage 2 (clustering + visualization) |

### Quick Start

```bash
# Stage 1: Embed patches on 8 MIG GPU slices (writes to outputs/preprocess_cache_dinov3/)
sbatch run_gpu_embed_patches.sh --num-gpus 8

# Stage 2: Cluster + visualize (reads from cache, no GPU needed)
sbatch run_gpu_cluster_viz.sh
```

### Stage 1 — Embedding (`run_gpu_embed_patches.sh`)

Embeds all sampled images and caches per-grid-cell embeddings to disk.
Multiple workers shard the image list and write to the same cache directory.

```bash
# Default: single GPU
sbatch run_gpu_embed_patches.sh --data-root ./data/fmow

# 8-way parallel across MIG slices
sbatch run_gpu_embed_patches.sh --data-root ./data/fmow --num-gpus 8

# Custom settings
sbatch run_gpu_embed_patches.sh \
  --data-root ./data/fmow \
  --cluster-num-images 35000 \
  --embed-batch 2048 \
  --gpu-batch-size 512 \
  --pool-mode cls_avg \
  --num-gpus 8
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `data/fmow` | Path to FMoW dataset |
| `--cluster-num-images` | `35000` | Number of images to embed (stratified sample) |
| `--embed-batch` | `2048` | Patches accumulated before GPU flush |
| `--gpu-batch-size` | `512` | Patches per ViT forward pass |
| `--pool-mode` | `cls_avg` | Pooling: `cls` (1024-d), `avg` (1024-d), `cls_avg` (2048-d) |
| `--num-gpus` | `1` | Number of MIG slices for parallel embedding |
| `--cache-dir` | `outputs/preprocess_cache_dinov3` | Where to write cached `.npz` files |

### Stage 2 — Clustering & Visualization (`run_gpu_cluster_viz.sh`)

Loads cached embeddings, runs PCA + clustering, and generates per-image
cluster overlay visualizations.

```bash
# Default: KMeans k=40, PCA to 256 dims
sbatch run_gpu_cluster_viz.sh

# Custom clustering
sbatch run_gpu_cluster_viz.sh \
  --k 60 \
  --pca-dim 256 \
  --viz-num-images 100 \
  --clusterer sklearn_kmeans
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--clusterer` | `sklearn_kmeans` | Clustering method: `sklearn_kmeans`, `bisecting_kmeans`, `gmm`, `hdbscan` |
| `--k` | `40` | Number of clusters (ignored for HDBSCAN) |
| `--pca-dim` | `256` | PCA dimensions before clustering |
| `--viz-num-images` | `100` | Number of images to render visualizations for |
| `--cluster-num-images` | `35000` | Must match the value used in Stage 1 |
| `--cache-dir` | `outputs/preprocess_cache_dinov3` | Must match Stage 1 |

> **Important:** `--cluster-num-images`, `--seed`, `--data-root`, `--split`,
> and all patching parameters (`--large-size`, `--small-size`, `--small-stride`,
> `--small-stride-x/y`) must be identical between Stage 1 and Stage 2 for the
> cache keys to match. The shell scripts use matching defaults.

---

## Composition-Aware Training

Trains DynamicVis backbone with a projection head that maps the backbone's
768-d global embedding to DINOv3's 2048-d space using a three-part loss:

1. **Contrastive (InfoNCE)** — each projected embedding should match its
   compositional target (average of cluster centroid embeddings for all
   small patches in the large patch).
2. **Smoothness** — adjacent grid cells from the same image should have
   similar embeddings.
3. **Consistency** — noisy perturbations of the same backbone features
   should produce similar projections.

### Architecture

| File | Purpose |
|------|---------|
| `losses/composition_loss.py` | `CompositionAwareLoss` — three-part loss module (registered with mmpretrain) |
| `models/composition_head.py` | `CompositionHead` (projection MLP 768→2048) + `CompositionAwareDynamicVis` (top-level model) |
| `datasets/fmow_composition_dataset.py` | `FMoWCompositionDataset` — loads image cells + pre-computed compositional targets |
| `configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py` | MMEngine config |
| `train_dynamicvis_composition.py` | Training script (thin wrapper around MMEngine Runner) |
| `run_gpu_composition_training.sh` | SLURM script for composition training |

### Full Pipeline (3 stages)

```bash
# Stage 1: Embed patches with DINOv3 (same as preprocessing)
sbatch run_gpu_embed_patches.sh --data-root ./data/fmow --num-gpus 8

# Stage 2: Cluster + export compositional targets
sbatch run_gpu_cluster_viz.sh --save-cluster-data --cluster-data-dir outputs/cluster_data

# Stage 3: Train DynamicVis with composition-aware loss
sbatch run_gpu_composition_training.sh --epochs 100 --num-gpus 2
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `32` | Per-GPU batch size |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `4e-4` | Learning rate |
| `--num-gpus` | `2` | MIG slices for DDP |
| `--cluster-data-dir` | `outputs/cluster_data` | Pre-computed cluster data directory |
| `--resume` | — | Resume from latest checkpoint |
| `--no-wandb` | — | Disable WandB logging |
| `--debug` | — | 2 epochs, batch size 8 |