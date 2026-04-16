# SatBae

A foundation model for satellite imagery built on the DynamicVis backbone with composition-aware distillation from DINOv3.

---

## Quick Start: Full Pipeline (QSACL Mode)

Run these commands **in order**. Each step depends on the previous.

### Step 0: Download fMoW Data

```bash
# Download 50% of fMoW RGB images (stratified by class)
python scripts/download_fmow.py \
    --output-dir data/fmow \
    --split train \
    --use-rgb \
    --fraction 0.5 \
    --workers 32
```

**Output**: `data/fmow/train/<category>/<location>/*.jpg`
**Time**: ~2-4 hours depending on network speed

### Step 1: Embed Patches with DINOv3

Extracts 2048-d embeddings for each 128x128 patch and caches them to disk.

```bash
sbatch run_gpu_embed_patches.sh \
    --data-root data/fmow \
    --num-gpus 8 \
    --cluster-num-images 999999  # Process ALL downloaded images
```

**Output**: `outputs/preprocess_cache_dinov3/*.npz`
**Time**: ~4-8 hours depending on dataset size

### Step 2: Generate Cell Manifest

Creates `manifest.json` listing all cells. Use `--manifest-only` to skip clustering (not needed for QSACL mode).

```bash
sbatch run_gpu_cluster_viz.sh \
    --manifest-only \
    --save-cluster-data \
    --cluster-data-dir outputs/cluster_data \
    --cluster-num-images 999999
```

**Output**: `outputs/cluster_data/manifest.json`
**Time**: ~10-30 minutes

### Step 3: Assign Cell Labels

Assigns fMoW category labels to cells based on bounding box IoU (needed for aux classification loss).

```bash
python scripts/assign_cell_labels.py \
    --cluster-data-dir outputs/cluster_data \
    --data-root data/fmow
```

**Output**: `outputs/cluster_data/cell_labels.npy`
**Time**: ~5-15 minutes

### Step 4: Train Composition-Aware Model (QSACL)

Trains the DynamicVis backbone with Two-View Query-Slot Attention Contrastive Learning.

```bash
sbatch run_gpu_composition_training.sh \
    --cluster-data-dir outputs/cluster_data \
    --epochs 100 \
    --num-gpus 8 \
    --batch-size 4
```

**Output**: `outputs/fmow_dynamicvis_b_composition_<job_id>/` (final_model.pth, final_backbone.pth)

### Important Notes

- **`--cluster-num-images 999999`** processes ALL available images (use actual count if known)
- **`--manifest-only`** skips PCA/KMeans clustering (not needed when using QSACL with `loss_comp=0`)
- Each step must complete before the next starts
- Add `--resume` to Step 4 to continue from a checkpoint

---

## Alternative: Full Clustering Pipeline (with composition targets)

If you want to use composition targets (`loss_comp > 0`), replace Step 2 with:

```bash
sbatch run_gpu_cluster_viz.sh \
    --save-cluster-data \
    --cluster-data-dir outputs/cluster_data \
    --cluster-num-images 999999 \
    --use-pca-targets
```

This generates `targets.npy` in addition to `manifest.json`.

---

## Download Options

```bash
# Download 10% of data (quick test)
python scripts/download_fmow.py --output-dir data/fmow --split train --use-rgb --fraction 0.1

# Download 50% of data (recommended)
python scripts/download_fmow.py --output-dir data/fmow --split train --use-rgb --fraction 0.5

# Download 100% of data (~350GB)
python scripts/download_fmow.py --output-dir data/fmow --split train --use-rgb

# Download validation set
python scripts/download_fmow.py --output-dir data/fmow --split val --use-rgb --fraction 0.5
```

---

## Pipeline Summary

```
data/fmow/                          # Raw images (Step 0: download)
    ↓
outputs/preprocess_cache_dinov3/    # DINOv3 embeddings (Step 1: embed_patches)
    ↓
outputs/cluster_data/manifest.json  # Cell enumeration (Step 2: cluster_viz --manifest-only)
    ↓
outputs/cluster_data/cell_labels.npy # fMoW labels (Step 3: assign_cell_labels)
    ↓
Training (Step 4: composition_training)
```

See [CONTEXT.md](CONTEXT.md) for detailed architecture and loss function documentation.

---

## Vanilla Pretrain Training

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

### Stage 2 — Manifest & Clustering (`run_gpu_cluster_viz.sh`)

Loads cached embeddings and generates cell manifest. Optionally runs PCA + clustering.

```bash
# Manifest only (QSACL mode - skips clustering, fast!)
sbatch run_gpu_cluster_viz.sh \
    --manifest-only \
    --save-cluster-data \
    --cluster-data-dir outputs/cluster_data \
    --cluster-num-images 999999

# Full clustering with composition targets
sbatch run_gpu_cluster_viz.sh \
    --save-cluster-data \
    --cluster-data-dir outputs/cluster_data \
    --cluster-num-images 999999 \
    --use-pca-targets
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest-only` | `false` | Skip clustering, only generate manifest.json (for QSACL mode) |
| `--save-cluster-data` | `false` | Save manifest.json (and targets.npy if clustering) |
| `--cluster-data-dir` | `outputs/cluster_data` | Output directory for cluster data |
| `--use-pca-targets` | `false` | Generate 256-d PCA targets (requires clustering) |
| `--clusterer` | `sklearn_kmeans` | Clustering method: `sklearn_kmeans`, `bisecting_kmeans`, `gmm`, `hdbscan` |
| `--k` | `40` | Number of clusters (ignored for HDBSCAN) |
| `--pca-dim` | `256` | PCA dimensions before clustering |
| `--cluster-num-images` | `35000` | Number of images to process (use 999999 for all) |

> **Important:** `--cluster-num-images`, `--seed`, `--data-root`, `--split`,
> and all patching parameters (`--large-size`, `--small-size`, `--small-stride`,
> `--small-stride-x/y`) must be identical between Stage 1 and Stage 2 for the
> cache keys to match. The shell scripts use matching defaults.

---

## Composition-Aware Training (QSACL)

Trains DynamicVis backbone using **Two-View Query-Slot Attention Contrastive Learning**:

1. **Multi-View Augmentation** — 8 views per cell (2 global + 6 local crops)
2. **Slot Contrastive (InfoNCE)** — positives are same cell under different augmentations
3. **Slot Variance** — VICReg-style variance hinge per slot position
4. **Auxiliary Classification** — label-guided CE loss on fMoW categories

### Architecture

| File | Purpose |
|------|---------|
| `losses/composition_loss.py` | `CompositionAwareLoss` — multi-term loss module |
| `models/composition_head.py` | `CompositionHead` + `CompositionAwareDynamicVis` + `MultiViewDataPreprocessor` |
| `models/query_slot_decoder.py` | `QuerySlotDecoder` — cross-attention over DINOv3 patch embeddings |
| `datasets/fmow_composition_dataset.py` | `FMoWCompositionDataset` — multi-view augmentation support |
| `configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py` | MMEngine config |
| `train_dynamicvis_composition.py` | Training script |
| `run_gpu_composition_training.sh` | SLURM script |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `4` | Per-GPU batch size (low due to 8 views per sample) |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `5e-4` | Learning rate |
| `--num-gpus` | `8` | MIG slices for DDP |
| `--cluster-data-dir` | `outputs/cluster_data` | Pre-computed cluster data directory |
| `--resume` | — | Resume from latest checkpoint |
| `--no-wandb` | — | Disable WandB logging |
| `--debug` | — | 2 epochs, batch size 8 |