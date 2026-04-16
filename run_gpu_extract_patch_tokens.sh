#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #NEVER CHANGE THIS
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# Extract raw patch tokens from FMoW cells using DINOv3 ViT-L/16.
# This is Phase 1 of the BoVW composition training pipeline.
#
# Usage:
#   sbatch run_gpu_extract_patch_tokens.sh
#   sbatch run_gpu_extract_patch_tokens.sh --num-gpus 8
#   sbatch run_gpu_extract_patch_tokens.sh --manifest data/fmow_manifest_train.json
#   sbatch run_gpu_extract_patch_tokens.sh --data-fraction 0.25  # Process 25% of data

set -euo pipefail

echo "=============================================="
echo "BoVW Phase 1 - Patch Token Extraction"
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

# =============================================================================
# Defaults
# =============================================================================
DATA_ROOT="data/fmow"
MANIFEST="data/fmow_manifest_train.json"
OUTPUT_DIR="outputs/patch_tokens_bovw"
WEIGHTS_PATH="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
BATCH_SIZE=64
NUM_GPUS=8
RESUME="--resume"
MAX_SAMPLES=""
DATA_FRACTION=""

# =============================================================================
# Parse args
# =============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)      DATA_ROOT="$2"; shift 2 ;;
    --manifest)       MANIFEST="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --weights-path)   WEIGHTS_PATH="$2"; shift 2 ;;
    --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
    --num-gpus)       NUM_GPUS="$2"; shift 2 ;;
    --no-resume)      RESUME=""; shift ;;
    --max-samples)    MAX_SAMPLES="$2"; shift 2 ;;
    --data-fraction)  DATA_FRACTION="$2"; shift 2 ;;
    *)                echo "Unknown arg: $1"; shift ;;
  esac
done

# =============================================================================
# Convert --data-fraction to --max-samples
# =============================================================================
if [ -n "$DATA_FRACTION" ]; then
    # Count total entries in manifest
    if [ -f "$MANIFEST" ]; then
        TOTAL_SAMPLES=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
        MAX_SAMPLES=$(python3 -c "import math; print(int(math.ceil($TOTAL_SAMPLES * $DATA_FRACTION)))")
        echo "Data fraction: $DATA_FRACTION → max_samples: $MAX_SAMPLES (of $TOTAL_SAMPLES total)"
    else
        echo "Warning: Cannot apply --data-fraction, manifest not found: $MANIFEST"
    fi
fi

echo ""
echo "Config:"
echo "  data_root:     $DATA_ROOT"
echo "  manifest:      $MANIFEST"
echo "  output_dir:    $OUTPUT_DIR"
echo "  weights_path:  $WEIGHTS_PATH"
echo "  batch_size:    $BATCH_SIZE"
echo "  num_gpus:      $NUM_GPUS"
echo "  resume:        ${RESUME:-disabled}"
echo "  data_fraction: ${DATA_FRACTION:-1.0}"
echo "  max_samples:   ${MAX_SAMPLES:-all}"
echo ""

# Cap NUM_GPUS to available slices
if [ "$NUM_GPUS" -gt "${#MIG_UUIDS[@]}" ]; then
    echo "WARNING: Requested $NUM_GPUS GPUs but only ${#MIG_UUIDS[@]} MIG slices available. Capping."
    NUM_GPUS=${#MIG_UUIDS[@]}
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build common Python args
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

PYTHON_ARGS="\
  --data-root $DATA_ROOT \
  --manifest $MANIFEST \
  --output-dir $OUTPUT_DIR \
  --weights-path $WEIGHTS_PATH \
  --batch-size $BATCH_SIZE \
  $RESUME \
  $MAX_SAMPLES_ARG"

# =============================================================================
# Launch parallel extraction workers
# =============================================================================
echo "=== Extracting patch tokens on $NUM_GPUS MIG slices ==="
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    MIG_UUID="${MIG_UUIDS[$i]}"
    echo "  Launching shard $i/$NUM_GPUS on MIG $MIG_UUID"
    CUDA_VISIBLE_DEVICES="$MIG_UUID" python -u scripts/extract_patch_tokens.py \
        $PYTHON_ARGS \
        --device cuda \
        --shard-index $i \
        --num-shards $NUM_GPUS \
        > "logs/extract_tokens_shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "  Waiting for ${#PIDS[@]} extraction workers..."
echo "  Logs: logs/extract_tokens_shard_*.log"
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        echo "ERROR: Extraction worker PID $pid failed!"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "$FAILED extraction worker(s) failed. Check logs for details."
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== Extraction Complete ==="
NUM_FILES=$(find "$OUTPUT_DIR" -name "*.npz" | wc -l)
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "  Output directory: $OUTPUT_DIR"
echo "  Total .npz files: $NUM_FILES"
echo "  Total size:       $TOTAL_SIZE"
echo ""
echo "Done: $(date)"
