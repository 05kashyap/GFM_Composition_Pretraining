#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #NEVER CHANGE THIS
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Build visual vocabulary from extracted patch tokens using FAISS K-means.
# This is Phase 2 of the BoVW composition training pipeline.
#
# Usage:
#   sbatch run_gpu_build_vocabulary.sh
#   sbatch run_gpu_build_vocabulary.sh --K 1024
#   sbatch run_gpu_build_vocabulary.sh --subsample 10000000

set -euo pipefail

echo "=============================================="
echo "BoVW Phase 2 - Visual Vocabulary Construction"
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
# MIG GPU Configuration (use first available MIG slice)
# =============================================================================
MIG_GPU0=(
    "MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"
)

# =============================================================================
# Defaults
# =============================================================================
PATCH_TOKEN_DIR="outputs/patch_tokens_bovw"
OUTPUT_DIR="outputs/bovw_vocabulary"
K=512
SUBSAMPLE=5000000
SEED=42
NITER=100
NREDO=3
DATA_FRACTION=""

# =============================================================================
# Parse args
# =============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --patch-token-dir) PATCH_TOKEN_DIR="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --K)               K="$2"; shift 2 ;;
    --subsample)       SUBSAMPLE="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --niter)           NITER="$2"; shift 2 ;;
    --nredo)           NREDO="$2"; shift 2 ;;
    --no-gpu)          NO_GPU="--no-gpu"; shift ;;
    --data-fraction)   DATA_FRACTION="$2"; shift 2 ;;
    *)                 echo "Unknown arg: $1"; shift ;;
  esac
done

# =============================================================================
# Scale subsample by data-fraction (optional)
# =============================================================================
if [ -n "$DATA_FRACTION" ]; then
    ORIGINAL_SUBSAMPLE=$SUBSAMPLE
    SUBSAMPLE=$(python3 -c "import math; print(int(math.ceil($SUBSAMPLE * $DATA_FRACTION)))")
    echo "Data fraction: $DATA_FRACTION → subsample: $SUBSAMPLE (scaled from $ORIGINAL_SUBSAMPLE)"
fi

echo ""
echo "Config:"
echo "  patch_token_dir: $PATCH_TOKEN_DIR"
echo "  output_dir:      $OUTPUT_DIR"
echo "  K:               $K"
echo "  subsample:       $SUBSAMPLE"
echo "  seed:            $SEED"
echo "  niter:           $NITER"
echo "  nredo:           $NREDO"
echo "  data_fraction:   ${DATA_FRACTION:-1.0}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Python args
PYTHON_ARGS="\
  --patch-token-dir $PATCH_TOKEN_DIR \
  --output-dir $OUTPUT_DIR \
  --K $K \
  --subsample $SUBSAMPLE \
  --seed $SEED \
  --niter $NITER \
  --nredo $NREDO \
  ${NO_GPU:-}"

# =============================================================================
# Run vocabulary building
# =============================================================================
echo "=== Building visual vocabulary ==="
MIG_UUID="${MIG_GPU0[0]}"
echo "  Using MIG slice: $MIG_UUID"

CUDA_VISIBLE_DEVICES="$MIG_UUID" python -u scripts/build_vocabulary.py \
    $PYTHON_ARGS \
    2>&1 | tee "logs/build_vocabulary.log"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== Vocabulary Construction Complete ==="
echo "  Centroids: $OUTPUT_DIR/centroids.npy"
echo "  Ground cost: $OUTPUT_DIR/ground_cost.npy"
echo "  Cluster histogram: $OUTPUT_DIR/cluster_sizes.png"
echo ""
echo "Done: $(date)"
