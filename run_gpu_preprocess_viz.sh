#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Standalone preprocessing + clustering + visualization (NO DynamicVis training)
# Usage:
#   sbatch run_gpu_preprocess_viz.sh --data-root ./data/fmow

set -euo pipefail

echo "=============================================="
echo "FMoW Preprocess+Cluster Viz - SLURM Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "PWD: $(pwd)"
echo "=============================================="

mkdir -p logs

# Conda bootstrap (mirrors your existing training script style)
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
CLUSTER_NUM_IMAGES=10000
VIZ_NUM_IMAGES=50

# Patch/standardization defaults per your request
PAD_SIZE=1024
LARGE_SIZE=256
SMALL_SIZE=64
# Sliding window strides (default: 50% overlap → stride = size / 2)
SMALL_STRIDE=32
SMALL_STRIDE_X=""  # empty = use SMALL_STRIDE for both axes
SMALL_STRIDE_Y=""
LARGE_STRIDE="$LARGE_SIZE"
LARGE_STRIDE_X=""
LARGE_STRIDE_Y=""

# HDBSCAN recipe defaults
FIT_SMALL_PATCHES_PER_IMAGE=16  # 5000 images -> 80k fit points
PCA_DIM=128
MIN_CLUSTER_SIZE=15
MIN_SAMPLES=5
HDBSCAN_JOBS=8

# Clustering algorithm: sklearn_kmeans | bisecting_kmeans | gmm | hdbscan | stream_kmeans
CLUSTERER="sklearn_kmeans"
K=100

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
    --min-cluster-size)
      MIN_CLUSTER_SIZE="$2"; shift 2 ;;
    --k)
      K="$2"; shift 2 ;;
    --clusterer)
      CLUSTERER="$2"; shift 2 ;;
    --pca-dim)
      PCA_DIM="$2"; shift 2 ;;
    --fit-small-patches-per-image)
      FIT_SMALL_PATCHES_PER_IMAGE="$2"; shift 2 ;;
    --small-stride)
      SMALL_STRIDE="$2"; shift 2 ;;
    --small-stride-x)
      SMALL_STRIDE_X="$2"; shift 2 ;;
    --small-stride-y)
      SMALL_STRIDE_Y="$2"; shift 2 ;;
    --large-stride)
      LARGE_STRIDE="$2"; shift 2 ;;
    --large-stride-x)
      LARGE_STRIDE_X="$2"; shift 2 ;;
    --large-stride-y)
      LARGE_STRIDE_Y="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; shift ;;
  esac
done

echo ""
echo "Config:"
echo "  data_root: $DATA_ROOT"
echo "  split: $SPLIT"
echo "  size_stats_n: $SIZE_STATS_N"
echo "  cluster_num_images: $CLUSTER_NUM_IMAGES"
echo "  viz_num_images: $VIZ_NUM_IMAGES"
echo "  pad_size: $PAD_SIZE"
echo "  large_size: $LARGE_SIZE"
echo "  small_size: $SMALL_SIZE"
echo "  small_stride: $SMALL_STRIDE (stride_x=${SMALL_STRIDE_X:-$SMALL_STRIDE} stride_y=${SMALL_STRIDE_Y:-$SMALL_STRIDE})"
echo "  embedder: dinov2"
echo "  clusterer: $CLUSTERER"
echo "  k: $K"
echo "  hdbscan_min_cluster_size: $MIN_CLUSTER_SIZE"
echo "  pca_dim: $PCA_DIM"
echo "  hdbscan_min_cluster_size: $MIN_CLUSTER_SIZE"
echo ""

# Unbuffered output is important for continuous Slurm logs
PYTHON_ARGS="\
  --data-root $DATA_ROOT \
  --split $SPLIT \
  --size-stats-n $SIZE_STATS_N \
  --cluster-num-images $CLUSTER_NUM_IMAGES \
  --viz-num-images $VIZ_NUM_IMAGES \
  --pad-size $PAD_SIZE \
  --large-size $LARGE_SIZE --large-stride $LARGE_STRIDE \
  --small-size $SMALL_SIZE --small-stride $SMALL_STRIDE \
  --embedder dinov2 \
  --device cuda \
  --clusterer $CLUSTERER \
  --k $K \
  --fit-small-patches-per-image $FIT_SMALL_PATCHES_PER_IMAGE \
  --pca-dim $PCA_DIM \
  --hdbscan-min-cluster-size $MIN_CLUSTER_SIZE \
  --hdbscan-min-samples $MIN_SAMPLES \
  --hdbscan-jobs $HDBSCAN_JOBS \
  --assign-noise-to-nearest-centroid"

# Append per-axis stride overrides if specified
if [ -n "$SMALL_STRIDE_X" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --small-stride-x $SMALL_STRIDE_X"
fi
if [ -n "$SMALL_STRIDE_Y" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --small-stride-y $SMALL_STRIDE_Y"
fi
if [ -n "$LARGE_STRIDE_X" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --large-stride-x $LARGE_STRIDE_X"
fi
if [ -n "$LARGE_STRIDE_Y" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --large-stride-y $LARGE_STRIDE_Y"
fi

python -u scripts/viz_fmow_patch_embed_cluster.py $PYTHON_ARGS

echo "Done: $(date)"
