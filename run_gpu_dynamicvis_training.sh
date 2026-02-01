#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
##SBATCH --time=00:10:00

# =============================================================================
# DynamicVis fMoW Pretraining - SLURM Script (with bounding boxes)
# =============================================================================
# This script trains DynamicVis from scratch using the official pretrain format:
# - Bounding box annotations (detection-style pretraining)
# - FPN neck + RoI extraction
# - Multi-instance learning (MIL) classification
# - Can download data locally or stream from AWS S3
#
# Usage:
#   sbatch run_gpu_dynamicvis_training.sh                     # Default (stream from S3)
#   sbatch run_gpu_dynamicvis_training.sh --download-data     # Download then train
#   sbatch run_gpu_dynamicvis_training.sh --data-root /path   # Use existing local data
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
DATA_ROOT=""           # Local data directory (if pre-downloaded)
DOWNLOAD_DATA=false    # Whether to download data before training
DATA_DIR="./data/fmow" # Default download location

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
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --download-data)
            DOWNLOAD_DATA=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
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
echo "  Data source: $([ -n "$DATA_ROOT" ] && echo "Local ($DATA_ROOT)" || echo "S3 streaming")"
echo "  Download data: $DOWNLOAD_DATA"
echo ""

# =============================================================================
# Data Download (if requested)
# =============================================================================
if [ "$DOWNLOAD_DATA" = true ]; then
    echo "=============================================="
    echo "Downloading fMoW dataset..."
    echo "=============================================="
    echo "Target directory: $DATA_DIR"
    echo "This will download ~35GB of msrgb images"
    echo ""
    
    # Run the download script (non-interactive for SLURM)
    python scripts/download_fmow.py \
        --output-dir "$DATA_DIR" \
        --split all \
        --workers 32 <<< "y"
    
    if [ $? -eq 0 ]; then
        echo "Download completed successfully!"
        DATA_ROOT="$DATA_DIR"
    else
        echo "Download failed! Falling back to S3 streaming..."
        DATA_ROOT=""
    fi
    echo ""
fi

# Build data root argument
DATA_ROOT_ARG=""
if [ -n "$DATA_ROOT" ]; then
    if [ -d "$DATA_ROOT" ]; then
        DATA_ROOT_ARG="--data-root $DATA_ROOT"
        echo "Using local data from: $DATA_ROOT"
    else
        echo "WARNING: Data root '$DATA_ROOT' does not exist, using S3 streaming"
    fi
fi

# Print GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

echo ""
echo "Starting training..."
echo ""

# Run training - using pretrain script for bbox-based training
python train_dynamicvis_pretrain.py $CONFIG \
    --work-dir $WORK_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    $RESUME $NO_WANDB $DATA_ROOT_ARG

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

