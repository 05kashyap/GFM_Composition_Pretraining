#!/bin/bash
#SBATCH --job-name=cluster_viz
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #NEVER CHANGE THIS
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Cluster and visualize cached FMoW patch embeddings.
# This is Phase 2 of the preprocessing pipeline (Phase 1 = embed_patches.py).
# Reads from the cache directory populated by run_gpu_embed_patches.sh.
#
# Usage:
#   sbatch run_gpu_cluster_viz.sh
#   sbatch run_gpu_cluster_viz.sh --k 60 --pca-dim 256 --viz-num-images 100

set -euo pipefail

echo "=============================================="
echo "FMoW Clustering + Visualization"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "PWD: $(pwd)"
echo "=============================================="

mkdir -p logs

# Conda bootstrap
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

# Defaults (patching params MUST match embed_patches.py)
DATA_ROOT="data/fmow"
SPLIT="train"
CLUSTER_NUM_IMAGES=35000
VIZ_NUM_IMAGES=100
SIZE_STATS_N=1000

LARGE_SIZE=512
LARGE_STRIDE="$LARGE_SIZE"
LARGE_STRIDE_X=""
LARGE_STRIDE_Y=""
SMALL_SIZE=128
SMALL_STRIDE=64
SMALL_STRIDE_X=""
SMALL_STRIDE_Y="128"

WEIGHTS_PATH="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
POOL_MODE="cls_avg"
CACHE_DIR="outputs/preprocess_cache_dinov3"

CLUSTERER="sklearn_kmeans"
K=40
PCA_DIM=256
FIT_SMALL_PATCHES_PER_IMAGE=64

MIN_CLUSTER_SIZE=15
MIN_SAMPLES=0
HDBSCAN_JOBS=8
SAVE_CLUSTER_DATA=false
CLUSTER_DATA_DIR="outputs/cluster_data"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)          DATA_ROOT="$2"; shift 2 ;;
    --split)              SPLIT="$2"; shift 2 ;;
    --cluster-num-images) CLUSTER_NUM_IMAGES="$2"; shift 2 ;;
    --viz-num-images)     VIZ_NUM_IMAGES="$2"; shift 2 ;;
    --clusterer)          CLUSTERER="$2"; shift 2 ;;
    --k)                  K="$2"; shift 2 ;;
    --pca-dim)            PCA_DIM="$2"; shift 2 ;;
    --pool-mode)          POOL_MODE="$2"; shift 2 ;;
    --cache-dir)          CACHE_DIR="$2"; shift 2 ;;
    --weights-path)       WEIGHTS_PATH="$2"; shift 2 ;;
    --fit-small-patches-per-image) FIT_SMALL_PATCHES_PER_IMAGE="$2"; shift 2 ;;
    --small-stride)       SMALL_STRIDE="$2"; shift 2 ;;
    --small-stride-x)     SMALL_STRIDE_X="$2"; shift 2 ;;
    --small-stride-y)     SMALL_STRIDE_Y="$2"; shift 2 ;;
    --large-stride)       LARGE_STRIDE="$2"; shift 2 ;;
    --large-stride-x)     LARGE_STRIDE_X="$2"; shift 2 ;;
    --large-stride-y)     LARGE_STRIDE_Y="$2"; shift 2 ;;
    --save-cluster-data)  SAVE_CLUSTER_DATA=true; shift ;;
    --cluster-data-dir)   CLUSTER_DATA_DIR="$2"; shift 2 ;;
    *)                    echo "Unknown arg: $1"; shift ;;
  esac
done

echo ""
echo "Config:"
echo "  data_root: $DATA_ROOT"
echo "  split: $SPLIT"
echo "  cluster_num_images: $CLUSTER_NUM_IMAGES"
echo "  viz_num_images: $VIZ_NUM_IMAGES"
echo "  large_size: $LARGE_SIZE"
echo "  small_size: $SMALL_SIZE"
echo "  small_stride: $SMALL_STRIDE (stride_x=${SMALL_STRIDE_X:-$SMALL_STRIDE} stride_y=${SMALL_STRIDE_Y:-$SMALL_STRIDE})"
echo "  weights_path: $WEIGHTS_PATH"
echo "  pool_mode: $POOL_MODE"
echo "  cache_dir: $CACHE_DIR"
echo "  clusterer: $CLUSTERER"
echo "  k: $K"
echo "  pca_dim: $PCA_DIM"
echo "  fit_small_patches_per_image: $FIT_SMALL_PATCHES_PER_IMAGE"
echo "  save_cluster_data: $SAVE_CLUSTER_DATA"
echo "  cluster_data_dir: $CLUSTER_DATA_DIR"
echo ""

# Build Python args (no GPU needed for clustering — runs on CPU)
PYTHON_ARGS="\
  --data-root $DATA_ROOT \
  --split $SPLIT \
  --size-stats-n $SIZE_STATS_N \
  --cluster-num-images $CLUSTER_NUM_IMAGES \
  --viz-num-images $VIZ_NUM_IMAGES \
  --large-size $LARGE_SIZE --large-stride $LARGE_STRIDE \
  --small-size $SMALL_SIZE --small-stride $SMALL_STRIDE \
  --weights-path $WEIGHTS_PATH \
  --pool-mode $POOL_MODE \
  --cache-dir $CACHE_DIR \
  --clusterer $CLUSTERER \
  --k $K \
  --pca-dim $PCA_DIM \
  --fit-small-patches-per-image $FIT_SMALL_PATCHES_PER_IMAGE \
  --hdbscan-min-cluster-size $MIN_CLUSTER_SIZE \
  --hdbscan-min-samples $MIN_SAMPLES \
  --hdbscan-jobs $HDBSCAN_JOBS \
  --save-embeddings"

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
if [ "$SAVE_CLUSTER_DATA" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --save-cluster-data --cluster-data-dir $CLUSTER_DATA_DIR"
fi

echo "=== Clustering + Visualization (reading from cache) ==="
python -u scripts/cluster_viz.py $PYTHON_ARGS

echo "Done: $(date)"
