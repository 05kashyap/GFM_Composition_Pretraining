#!/bin/bash
#SBATCH --job-name=AryanKashyapN_dinov3
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00

# Standalone preprocessing + clustering + visualization using local DINOv3 SAT weights.
# Usage:
#   sbatch run_gpu_preprocess_viz_dinov3.sh --data-root ./data/fmow

set -euo pipefail

echo "=============================================="
echo "FMoW Preprocess+Cluster Viz - DINOv3 SAT (local)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "PWD: $(pwd)"
echo "=============================================="

mkdir -p logs

# Conda bootstrap (mirrors existing repo scripts)
CONDA_BASE=""
if [ -d "/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3" ]; then
  CONDA_BASE="/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
  CONDA_BASE="$HOME/anaconda3"
fi

if [ -n "$CONDA_BASE" ]; then
  __conda_setup="$($CONDA_BASE/bin/conda shell.bash hook 2> /dev/null || true)"
  if [ -n "$__conda_setup" ]; then
    eval "$__conda_setup"
  elif [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
  else
    export PATH="$CONDA_BASE/bin:$PATH"
  fi
  unset __conda_setup
fi

conda activate dynamicvis

DATA_ROOT="data/fmow"
SPLIT="train"
SIZE_STATS_N=1000
CLUSTER_NUM_IMAGES=35000
VIZ_NUM_IMAGES=100

# Patch/standardization defaults
PAD_SIZE=1024
LARGE_SIZE=512
SMALL_SIZE=128

# DINOv3 SAT local weights
WEIGHTS_PATH="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# Clustering defaults: kmeans with k=100
CLUSTERER="sklearn_kmeans"
K=40

# Fit subset and PCA defaults
FIT_SMALL_PATCHES_PER_IMAGE=64
PCA_DIM=128

# HDBSCAN defaults (only used if CLUSTERER=hdbscan)
MIN_CLUSTER_SIZE=15
MIN_SAMPLES=0
HDBSCAN_JOBS=8

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)
      DATA_ROOT="$2"; shift 2 ;;
    --split)
      SPLIT="$2"; shift 2 ;;
    --cluster-num-images)
      CLUSTER_NUM_IMAGES="$2"; shift 2 ;;
    --viz-num-images)
      VIZ_NUM_IMAGES="$2"; shift 2 ;;
    --clusterer)
      CLUSTERER="$2"; shift 2 ;;
    --k)
      K="$2"; shift 2 ;;
    --weights-path)
      WEIGHTS_PATH="$2"; shift 2 ;;
    --pca-dim)
      PCA_DIM="$2"; shift 2 ;;
    --fit-small-patches-per-image)
      FIT_SMALL_PATCHES_PER_IMAGE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; shift ;;
  esac
done

echo ""
echo "Config:"
echo "  data_root: $DATA_ROOT"
echo "  split: $SPLIT"
echo "  cluster_num_images: $CLUSTER_NUM_IMAGES"
echo "  viz_num_images: $VIZ_NUM_IMAGES"
echo "  pad_size: $PAD_SIZE"
echo "  large_size: $LARGE_SIZE"
echo "  small_size: $SMALL_SIZE"
echo "  weights_path: $WEIGHTS_PATH"
echo "  clusterer: $CLUSTERER"
echo "  k: $K"
echo "  fit_small_patches_per_image: $FIT_SMALL_PATCHES_PER_IMAGE"
echo "  pca_dim: $PCA_DIM"
echo ""

# Unbuffered output for continuous Slurm logs
python -u scripts/viz_fmow_patch_embed_cluster_dinov3.py \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --size-stats-n "$SIZE_STATS_N" \
  --cluster-num-images "$CLUSTER_NUM_IMAGES" \
  --viz-num-images "$VIZ_NUM_IMAGES" \
  --max-edge "$PAD_SIZE" \
  --pad-size "$PAD_SIZE" \
  --large-size "$LARGE_SIZE" --large-stride "$LARGE_SIZE" \
  --small-size "$SMALL_SIZE" --small-stride "$SMALL_SIZE" \
  --weights-path "$WEIGHTS_PATH" \
  --device cuda \
  --embed-batch 64 \
  --clusterer "$CLUSTERER" \
  --k "$K" \
  --fit-small-patches-per-image "$FIT_SMALL_PATCHES_PER_IMAGE" \
  --pca-dim "$PCA_DIM" \
  --hdbscan-min-cluster-size "$MIN_CLUSTER_SIZE" \
  --hdbscan-min-samples "$MIN_SAMPLES" \
  --hdbscan-jobs "$HDBSCAN_JOBS" \
  --cache-embeddings \
  --save-embeddings

echo "Done: $(date)"
