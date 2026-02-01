#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:1      # 1 MIG slice (torchrun+NCCL has issues with multi-MIG)
#SBATCH --cpus-per-task=4         # More CPUs for data loading
#SBATCH --mem=32G
##SBATCH --time=00:10:00

# =============================================================================
# DynamicVis fMoW Pretraining - SLURM Script (with bounding boxes)
# =============================================================================
# This script trains DynamicVis from scratch using the official pretrain format:
# - Bounding box annotations (detection-style pretraining)
# - FPN neck + RoI extraction
# - Multi-instance learning (MIL) classification
# - Streams data from AWS S3 (no need to download 350GB dataset)
#
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
# Try different possible conda locations
CONDA_BASE=""
if [ -d "/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3" ]; then
    CONDA_BASE="/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi

if [ -n "$CONDA_BASE" ]; then
    # Initialize conda for bash
    __conda_setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            export PATH="$CONDA_BASE/bin:$PATH"
        fi
    fi
    unset __conda_setup
fi

conda activate dynamicvis

# Check if DynamicVis is cloned (it's a separate git repo, not included in main repo)
if [ ! -d "architectures/DynamicVis/dynamicvis" ]; then
    echo "DynamicVis not found. Cloning..."
    mkdir -p architectures
    # Use the correct repo URL from DynamicVis README
    git clone https://github.com/KyanChen/DynamicVis.git architectures/DynamicVis
    echo "DynamicVis cloned successfully."
else
    echo "DynamicVis found at architectures/DynamicVis/"
fi

# Set PYTHONPATH to include DynamicVis architecture (after clone check)
export PYTHONPATH="$(pwd):$(pwd)/architectures/DynamicVis:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Load environment variables (AWS credentials, wandb key)
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
fi

# Default values - Using pretrain config with bounding boxes (like official weights)
CONFIG="configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py"
WORK_DIR="outputs/fmow_dynamicvis_b_s3_pretrain"
BATCH_SIZE=16  # Smaller batch due to larger images + FPN
EPOCHS=200     # Official training uses 200 epochs
LR=4e-4        # Official learning rate
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

# Handle GPU detection and MIG compatibility
# MIG (Multi-Instance GPU) slices don't work well with torchrun + NCCL
# For MIG environments, use single GPU training for stability
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
    echo "Detected CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (${NUM_GPUS} device(s))"
    
    # Check if MIG is enabled (MIG devices cause issues with torchrun + NCCL)
    if nvidia-smi -L 2>/dev/null | grep -q "MIG"; then
        echo "MIG (Multi-Instance GPU) detected - using single GPU for stability"
        echo "Note: torchrun + NCCL has known issues with MIG slices"
        # Use only the first MIG slice
        export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
        NUM_GPUS=1
        echo "Using single MIG slice: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
else
    NUM_GPUS=1
    echo "CUDA_VISIBLE_DEVICES not set, assuming 1 GPU"
fi

echo ""
echo "Starting training..."
echo ""

# Run training - use torchrun for multi-GPU, regular python for single GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using distributed training with $NUM_GPUS GPUs"
    torchrun --nproc_per_node=$NUM_GPUS \
        train_dynamicvis_pretrain.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --launcher pytorch \
        $RESUME $NO_WANDB
else
    echo "Using single GPU training"
    python train_dynamicvis_pretrain.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $RESUME $NO_WANDB
fi

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

