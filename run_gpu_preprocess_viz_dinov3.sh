#!/bin/bash
#SBATCH --job-name=AryanKashyapN_dinov3
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #NEVER CHANGE THIS
#SBATCH --mem=48G
#SBATCH --time=16:00:00

# Standalone preprocessing + clustering + visualization using local DINOv3 SAT weights.
# Usage:
#   sbatch run_gpu_preprocess_viz_dinov3.sh --data-root ./data/fmow
#   sbatch run_gpu_preprocess_viz_dinov3.sh --data-root ./data/fmow --num-gpus 4
#   sbatch run_gpu_preprocess_viz_dinov3.sh --data-root ./data/fmow --num-gpus 8

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

# =============================================================================
# MIG GPU Configuration (same layout as run_gpu_dynamicvis_training.sh)
# =============================================================================
# All available MIG slices, round-robin across physical GPUs.
# GPU 0 (PCIe bus 26:00.0) and GPU 1 (PCIe bus 89:00.0)
MIG_GPU0=(
    "MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"  # GPU0 Device 1
    "MIG-a95be726-a3ac-5445-8427-a9fba33402a4"  # GPU0 Device 0
    "MIG-093f7e1b-6f21-5e12-bc9e-435e7f019600"  # GPU0 Device 2
    "MIG-d92daec9-5afd-5a16-adac-861b9216d4f9"  # GPU0 Device 3
)
MIG_GPU1=(
    "MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"  # GPU1 Device 0
    "MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"  # GPU1 Device 1
    "MIG-283697f2-709b-5149-bf8e-30a0abe75dc4"  # GPU1 Device 2
    "MIG-83d74367-8c40-525a-9aab-fb561a0a3daa"  # GPU1 Device 3
)
# Round-robin interleave for best cross-GPU spread
MIG_UUIDS=()
for ((i=0; i<${#MIG_GPU0[@]}; i++)); do
    MIG_UUIDS+=("${MIG_GPU0[$i]}")
    MIG_UUIDS+=("${MIG_GPU1[$i]}")
done

DATA_ROOT="data/fmow"
SPLIT="train"
SIZE_STATS_N=1000
CLUSTER_NUM_IMAGES=35000
VIZ_NUM_IMAGES=100

# Patch/standardization defaults
PAD_SIZE=1024
LARGE_SIZE=512
SMALL_SIZE=128
# Sliding window strides (default: 50% overlap → stride = size / 2)
SMALL_STRIDE=64
SMALL_STRIDE_X=""  # empty = use SMALL_STRIDE for both axes
SMALL_STRIDE_Y=""
LARGE_STRIDE="$LARGE_SIZE"
LARGE_STRIDE_X=""
LARGE_STRIDE_Y=""

# DINOv3 SAT local weights
WEIGHTS_PATH="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# Embedding batch size (patches to accumulate across multiple images)
# With 128px patches + 64px stride on 1024px images, each image produces ~225 patches.
# EMBED_BATCH controls when the MultiImageBatchedEmbedder flushes accumulated patches
# to the GPU.  Setting this to ~9-18× patches-per-image (2048) ensures the GPU gets
# large batches while GPU_BATCH_SIZE controls the actual per-forward-pass sub-batch.
EMBED_BATCH=2048
# GPU batch size (patches per ViT forward pass - increase to use more VRAM)
GPU_BATCH_SIZE=512

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

# Multi-GPU: number of MIG slices for parallel embedding
NUM_GPUS=1

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
    --embed-batch)
      EMBED_BATCH="$2"; shift 2 ;;
    --gpu-batch-size)
      GPU_BATCH_SIZE="$2"; shift 2 ;;
    --pca-dim)
      PCA_DIM="$2"; shift 2 ;;
    --fit-small-patches-per-image)
      FIT_SMALL_PATCHES_PER_IMAGE="$2"; shift 2 ;;
    --num-gpus)
      NUM_GPUS="$2"; shift 2 ;;
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
echo "  cluster_num_images: $CLUSTER_NUM_IMAGES"
echo "  viz_num_images: $VIZ_NUM_IMAGES"
echo "  pad_size: $PAD_SIZE"
echo "  large_size: $LARGE_SIZE"
echo "  small_size: $SMALL_SIZE"
echo "  small_stride: $SMALL_STRIDE (stride_x=${SMALL_STRIDE_X:-$SMALL_STRIDE} stride_y=${SMALL_STRIDE_Y:-$SMALL_STRIDE})"
echo "  weights_path: $WEIGHTS_PATH"
echo "  embed_batch: $EMBED_BATCH"
echo "  gpu_batch_size: $GPU_BATCH_SIZE"
echo "  clusterer: $CLUSTERER"
echo "  k: $K"
echo "  fit_small_patches_per_image: $FIT_SMALL_PATCHES_PER_IMAGE"
echo "  pca_dim: $PCA_DIM"
echo "  num_gpus: $NUM_GPUS"
echo ""

# Cap NUM_GPUS to available slices
if [ "$NUM_GPUS" -gt "${#MIG_UUIDS[@]}" ]; then
    echo "WARNING: Requested $NUM_GPUS GPUs but only ${#MIG_UUIDS[@]} MIG slices available. Capping."
    NUM_GPUS=${#MIG_UUIDS[@]}
fi

# Common args for the Python script
PYTHON_ARGS="\
  --data-root $DATA_ROOT \
  --split $SPLIT \
  --size-stats-n $SIZE_STATS_N \
  --cluster-num-images $CLUSTER_NUM_IMAGES \
  --viz-num-images $VIZ_NUM_IMAGES \
  --max-edge $PAD_SIZE \
  --pad-size $PAD_SIZE \
  --large-size $LARGE_SIZE --large-stride $LARGE_STRIDE \
  --small-size $SMALL_SIZE --small-stride $SMALL_STRIDE \
  --weights-path $WEIGHTS_PATH \
  --embed-batch $EMBED_BATCH \
  --gpu-batch-size $GPU_BATCH_SIZE \
  --clusterer $CLUSTERER \
  --k $K \
  --fit-small-patches-per-image $FIT_SMALL_PATCHES_PER_IMAGE \
  --pca-dim $PCA_DIM \
  --hdbscan-min-cluster-size $MIN_CLUSTER_SIZE \
  --hdbscan-min-samples $MIN_SAMPLES \
  --hdbscan-jobs $HDBSCAN_JOBS \
  --cache-embeddings \
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

if [ "$NUM_GPUS" -gt 1 ]; then
    # ==========================================================================
    # Phase 1: Parallel embedding across N MIG slices
    # Each worker caches embeddings for its shard of images to disk.
    # ==========================================================================
    echo "=== Phase 1: Parallel embedding on $NUM_GPUS MIG slices ==="
    EMBED_PIDS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        MIG_UUID="${MIG_UUIDS[$i]}"
        echo "  Launching shard $i/$NUM_GPUS on MIG $MIG_UUID"
        CUDA_VISIBLE_DEVICES="$MIG_UUID" python -u scripts/viz_fmow_patch_embed_cluster_dinov3.py \
            $PYTHON_ARGS \
            --device cuda \
            --embed-only \
            --shard-index $i \
            --num-shards $NUM_GPUS &
        EMBED_PIDS+=($!)
    done

    echo "  Waiting for ${#EMBED_PIDS[@]} embedding workers..."
    FAILED=0
    for pid in "${EMBED_PIDS[@]}"; do
        if ! wait $pid; then
            echo "ERROR: Embedding worker PID $pid failed!"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "$FAILED embedding worker(s) failed. Aborting."
        exit 1
    fi
    echo "=== Phase 1 complete: all embeddings cached ==="
    echo ""

    # ==========================================================================
    # Phase 2: Full pipeline on single GPU (embeddings served from cache)
    # ==========================================================================
    echo "=== Phase 2: Clustering + visualization (single GPU, reading from cache) ==="
    CUDA_VISIBLE_DEVICES="${MIG_UUIDS[0]}" python -u scripts/viz_fmow_patch_embed_cluster_dinov3.py \
        $PYTHON_ARGS \
        --device cuda
else
    # Single GPU mode
    CUDA_VISIBLE_DEVICES="${MIG_UUIDS[0]}" python -u scripts/viz_fmow_patch_embed_cluster_dinov3.py \
        $PYTHON_ARGS \
        --device cuda
fi

echo "Done: $(date)"
