#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
##SBATCH --time=00:10:00

# =============================================================================
# DynamicVis fMoW Training - SLURM Script
# =============================================================================
# Usage:
#   sbatch run_gpu_dynamicvis_training.sh                     # Default training
#   sbatch run_gpu_dynamicvis_training.sh --epochs 50         # Custom epochs
#   sbatch run_gpu_dynamicvis_training.sh --no-wandb          # Disable wandb
#   sbatch run_gpu_dynamicvis_training.sh --resume            # Resume training
# =============================================================================

echo "=============================================="
echo "DynamicVis fMoW Training - SLURM Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda environment (adjust path if needed)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate dynamicvis

# Load environment variables (AWS credentials, wandb key)
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
fi

# Default values
CONFIG="configs_dynamicvis/fmow_classification/dynamicvis_b_fmow_s3.py"
WORK_DIR="outputs/fmow_dynamicvis_b_s3"
BATCH_SIZE=32
EPOCHS=100
LR=1e-4
RESUME=""
NO_WANDB=""

# Add SLURM job ID to work dir if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    WORK_DIR="${WORK_DIR}_${SLURM_JOB_ID}"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --debug)
            EPOCHS=2
            BATCH_SIZE=8
            shift
            ;;
        --resume)
            RESUME="--resume auto"
            shift
            ;;
        --no-wandb)
            NO_WANDB="--no-wandb"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Print configuration
echo ""
echo "Training Configuration:"
echo "  Config: $CONFIG"
echo "  Work directory: $WORK_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  WandB: $([ -z "$NO_WANDB" ] && echo "Enabled" || echo "Disabled")"
echo ""

# Print GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

echo ""
echo "Starting training..."
echo ""

# Run training
python train_dynamicvis.py $CONFIG \
    --work-dir $WORK_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    $RESUME $NO_WANDB

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

