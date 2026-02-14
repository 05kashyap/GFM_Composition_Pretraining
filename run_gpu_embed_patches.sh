#!/bin/bash
#SBATCH --job-name=embed_patches
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #NEVER CHANGE THIS
#SBATCH --mem=48G
#SBATCH --time=16:00:00

# Embed FMoW patches with DINOv3 on MIG GPUs and cache to disk.
# This is Phase 1 of the preprocessing pipeline (Phase 2 = cluster_viz.py).
#
# Usage:
#   sbatch run_gpu_embed_patches.sh --data-root ./data/fmow
#   sbatch run_gpu_embed_patches.sh --data-root ./data/fmow --num-gpus 8

set -euo pipefail

echo "=============================================="
echo "FMoW Patch Embedding - DINOv3 SAT (local)"
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

# =============================================================================
# MIG GPU Configuration
# =============================================================================
MIG_GPU0=(
    "MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"
    "MIG-a95be726-a3ac-5445-8427-a9fba33402a4"
    "MIG-093f7e1b-6f21-5e12-bc9e-435e7f019600"
    "MIG-d92daec9-5afd-5a16-adac-861b9216d4f9"
)
MIG_GPU1=(
    "MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
    "MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"
    "MIG-283697f2-709b-5149-bf8e-30a0abe75dc4"
    "MIG-83d74367-8c40-525a-9aab-fb561a0a3daa"
)
# Round-robin interleave across physical GPUs
MIG_UUIDS=()
for ((i=0; i<${#MIG_GPU0[@]}; i++)); do
    MIG_UUIDS+=("${MIG_GPU0[$i]}")
    MIG_UUIDS+=("${MIG_GPU1[$i]}")
done

# Defaults
DATA_ROOT="data/fmow"
SPLIT="train"
CLUSTER_NUM_IMAGES=35000

LARGE_SIZE=512
SMALL_SIZE=128
SMALL_STRIDE=64
SMALL_STRIDE_X=""
SMALL_STRIDE_Y="128"

WEIGHTS_PATH="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
POOL_MODE="cls_avg"
EMBED_BATCH=2048
GPU_BATCH_SIZE=512
CACHE_DIR="outputs/preprocess_cache_dinov3"

NUM_GPUS=1

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)          DATA_ROOT="$2"; shift 2 ;;
    --split)              SPLIT="$2"; shift 2 ;;
    --cluster-num-images) CLUSTER_NUM_IMAGES="$2"; shift 2 ;;
    --weights-path)       WEIGHTS_PATH="$2"; shift 2 ;;
    --embed-batch)        EMBED_BATCH="$2"; shift 2 ;;
    --gpu-batch-size)     GPU_BATCH_SIZE="$2"; shift 2 ;;
    --pool-mode)          POOL_MODE="$2"; shift 2 ;;
    --cache-dir)          CACHE_DIR="$2"; shift 2 ;;
    --num-gpus)           NUM_GPUS="$2"; shift 2 ;;
    --small-stride)       SMALL_STRIDE="$2"; shift 2 ;;
    --small-stride-x)     SMALL_STRIDE_X="$2"; shift 2 ;;
    --small-stride-y)     SMALL_STRIDE_Y="$2"; shift 2 ;;
    *)                    echo "Unknown arg: $1"; shift ;;
  esac
done

echo ""
echo "Config:"
echo "  data_root: $DATA_ROOT"
echo "  split: $SPLIT"
echo "  cluster_num_images: $CLUSTER_NUM_IMAGES"
echo "  large_size: $LARGE_SIZE"
echo "  small_size: $SMALL_SIZE"
echo "  small_stride: $SMALL_STRIDE (stride_x=${SMALL_STRIDE_X:-$SMALL_STRIDE} stride_y=${SMALL_STRIDE_Y:-$SMALL_STRIDE})"
echo "  weights_path: $WEIGHTS_PATH"
echo "  pool_mode: $POOL_MODE"
echo "  embed_batch: $EMBED_BATCH"
echo "  gpu_batch_size: $GPU_BATCH_SIZE"
echo "  cache_dir: $CACHE_DIR"
echo "  num_gpus: $NUM_GPUS"
echo ""

# Cap NUM_GPUS to available slices
if [ "$NUM_GPUS" -gt "${#MIG_UUIDS[@]}" ]; then
    echo "WARNING: Requested $NUM_GPUS GPUs but only ${#MIG_UUIDS[@]} MIG slices available. Capping."
    NUM_GPUS=${#MIG_UUIDS[@]}
fi

# Build common Python args
PYTHON_ARGS="\
  --data-root $DATA_ROOT \
  --split $SPLIT \
  --cluster-num-images $CLUSTER_NUM_IMAGES \
  --large-size $LARGE_SIZE \
  --small-size $SMALL_SIZE --small-stride $SMALL_STRIDE \
  --weights-path $WEIGHTS_PATH \
  --pool-mode $POOL_MODE \
  --embed-batch $EMBED_BATCH \
  --gpu-batch-size $GPU_BATCH_SIZE \
  --cache-dir $CACHE_DIR"

if [ -n "$SMALL_STRIDE_X" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --small-stride-x $SMALL_STRIDE_X"
fi
if [ -n "$SMALL_STRIDE_Y" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --small-stride-y $SMALL_STRIDE_Y"
fi

# Launch parallel embedding workers
echo "=== Embedding on $NUM_GPUS MIG slices ==="
EMBED_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    MIG_UUID="${MIG_UUIDS[$i]}"
    echo "  Launching shard $i/$NUM_GPUS on MIG $MIG_UUID"
    CUDA_VISIBLE_DEVICES="$MIG_UUID" python -u scripts/embed_patches.py \
        $PYTHON_ARGS \
        --device cuda \
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

echo "=== All embeddings cached to $CACHE_DIR ==="
echo "Done: $(date)"
