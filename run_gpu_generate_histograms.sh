#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #NEVER CHANGE THIS
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Generate soft-assignment histograms from patch tokens using visual vocabulary.
# This is Phase 3 of the BoVW composition training pipeline.
#
# Usage:
#   sbatch run_gpu_generate_histograms.sh
#   sbatch run_gpu_generate_histograms.sh --beta 15.0
#   sbatch run_gpu_generate_histograms.sh --workers 12
#   sbatch run_gpu_generate_histograms.sh --data-fraction 0.25  # Process 25% of data

set -euo pipefail

echo "=============================================="
echo "BoVW Phase 3 - Histogram Target Generation"
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
# Defaults
# =============================================================================
PATCH_TOKEN_DIR="/mnt/usb/patch_tokens_bovw"
VOCAB_DIR="/mnt/usb/bovw_vocabulary"
MANIFEST="data/fmow_manifest_train.json"
OUTPUT_DIR="/mnt/usb/bovw_histograms"
WEIGHTS_PATH="weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DATA_ROOT="/mnt/usb/fmow"
BETA=10.0
WORKERS=8
RESUME="--resume"
MAX_SAMPLES=""
DATA_FRACTION=""

# =============================================================================
# Parse args
# =============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --patch-token-dir) PATCH_TOKEN_DIR="$2"; shift 2 ;;
    --vocab-dir)       VOCAB_DIR="$2"; shift 2 ;;
    --manifest)        MANIFEST="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --weights-path)    WEIGHTS_PATH="$2"; shift 2 ;;
    --data-root)       DATA_ROOT="$2"; shift 2 ;;
    --beta)            BETA="$2"; shift 2 ;;
    --workers)         WORKERS="$2"; shift 2 ;;
    --no-resume)       RESUME=""; shift ;;
    --max-samples)     MAX_SAMPLES="$2"; shift 2 ;;
    --data-fraction)   DATA_FRACTION="$2"; shift 2 ;;
    *)                 echo "Unknown arg: $1"; shift ;;
  esac
done

# =============================================================================
# Convert --data-fraction to --max-samples
# =============================================================================
if [ -n "$DATA_FRACTION" ]; then
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
echo "  patch_token_dir: $PATCH_TOKEN_DIR"
echo "  vocab_dir:       $VOCAB_DIR"
echo "  manifest:        $MANIFEST"
echo "  output_dir:      $OUTPUT_DIR"
echo "  data_root:       $DATA_ROOT"
echo "  beta:            $BETA"
echo "  workers:         $WORKERS"
echo "  resume:          ${RESUME:-disabled}"
echo "  data_fraction:   ${DATA_FRACTION:-1.0}"
echo "  max_samples:     ${MAX_SAMPLES:-all}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Python args
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

PYTHON_ARGS="\
  --patch-token-dir $PATCH_TOKEN_DIR \
  --vocab-dir $VOCAB_DIR \
  --manifest $MANIFEST \
  --output-dir $OUTPUT_DIR \
  --weights-path $WEIGHTS_PATH \
  --data-root $DATA_ROOT \
  --beta $BETA \
  --workers $WORKERS \
  $RESUME \
  $MAX_SAMPLES_ARG"

# =============================================================================
# Run histogram generation
# =============================================================================
echo "=== Generating histograms ==="

python -u scripts/generate_histograms.py \
    $PYTHON_ARGS \
    2>&1 | tee "logs/generate_histograms.log"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== Histogram Generation Complete ==="
echo "  Histograms: $OUTPUT_DIR/histograms.npy"
echo "  Cell IDs: $OUTPUT_DIR/cell_ids.npy"
echo ""
echo "Done: $(date)"
