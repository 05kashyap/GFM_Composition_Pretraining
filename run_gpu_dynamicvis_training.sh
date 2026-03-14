#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
#   sbatch run_gpu_dynamicvis_training.sh --download-data --data-fraction 0.1  # Download 10% per class
#   sbatch run_gpu_dynamicvis_training.sh --data-root /path   # Use existing local data
#   sbatch run_gpu_dynamicvis_training.sh --epochs 50         # Custom epochs
#   sbatch run_gpu_dynamicvis_training.sh --no-wandb          # Disable wandb
#   sbatch run_gpu_dynamicvis_training.sh --resume            # Resume training
#   sbatch run_gpu_dynamicvis_training.sh --num-gpus 2         # DDP on 2 MIG slices
#   sbatch run_gpu_dynamicvis_training.sh --num-gpus 1         # Single MIG slice (default)
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

# =============================================================================
# MIG GPU Configuration for DDP
# =============================================================================
# All available MIG slices, organized by physical GPU.
# GPU 0 (PCIe bus 26:00.0) and GPU 1 (PCIe bus 89:00.0)
# Use: nvidia-smi -L to discover available MIG UUIDs on your node
#
# NCCL constraint: only 1 MIG slice per physical GPU (same bus ID = "Duplicate GPU").
#   → --num-gpus 1-2: uses NCCL backend (fast, 1 slice per physical GPU)
#   → --num-gpus 3-8: uses gloo backend (allows multiple slices per GPU)
#
# Slices are listed in round-robin order (GPU0, GPU1, GPU0, GPU1, ...)
# so the first N slices always maximize cross-GPU spread.
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

# Build round-robin interleaved list: GPU0[0], GPU1[0], GPU0[1], GPU1[1], ...
MIG_UUIDS=()
for ((i=0; i<${#MIG_GPU0[@]}; i++)); do
    MIG_UUIDS+=("${MIG_GPU0[$i]}")
    MIG_UUIDS+=("${MIG_GPU1[$i]}")
done
echo "Available MIG slices: ${#MIG_UUIDS[@]} (${#MIG_GPU0[@]} per GPU x 2 GPUs)"

# NCCL settings for MIG (P2P not supported between MIG instances)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
# Reduce NCCL verbosity (set to INFO for debugging)
export NCCL_DEBUG=WARN

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
BATCH_SIZE=32   # Reduced for full rgb images (larger than msrgb)
EPOCHS=2     # Official training uses 200 epochs
LR=4e-4        # Official learning rate
RESUME=""
NO_WANDB=""
DATA_ROOT=""           # Local data directory (if pre-downloaded)
DOWNLOAD_DATA=false    # Whether to download data before training
DATA_DIR="$(pwd)/data/fmow" # Default download location (absolute path)
DATA_FRACTION=""       # Fraction of data to download per class (empty = all, e.g., 0.1 for 10%)
NUM_GPUS=2             # Number of MIG slices to use for DDP (1 = single GPU, 2+ = DDP)
DIST_BACKEND="auto"    # auto=nccl for <=2 GPUs, gloo for >2. Or force: nccl, gloo

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
        --data-fraction)
            DATA_FRACTION="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dist-backend)
            DIST_BACKEND="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# =============================================================================
# Resolve distributed backend and validate NUM_GPUS
# =============================================================================
# Cap NUM_GPUS to available slices
if [ "$NUM_GPUS" -gt "${#MIG_UUIDS[@]}" ]; then
    echo "WARNING: Requested $NUM_GPUS GPUs but only ${#MIG_UUIDS[@]} MIG slices available. Capping."
    NUM_GPUS=${#MIG_UUIDS[@]}
fi

# Auto-select backend:
#   <=2 GPUs: nccl (1 slice per physical GPU, fast GPU-native collectives)
#   >2 GPUs:  gloo (CPU-based collectives, allows multiple slices per physical GPU)
if [ "$DIST_BACKEND" = "auto" ]; then
    if [ "$NUM_GPUS" -lt 2 ]; then
        DIST_BACKEND="nccl"
    else
        DIST_BACKEND="gloo"
    fi
fi

if [ "$DIST_BACKEND" = "nccl" ] && [ "$NUM_GPUS" -gt 2 ]; then
    echo "WARNING: NCCL with >2 MIG slices will fail (duplicate GPU on same PCIe bus)."
    echo "         Switching to gloo backend automatically."
    DIST_BACKEND="gloo"
fi

echo "Distributed backend: $DIST_BACKEND"

# =============================================================================
# Set CUDA_VISIBLE_DEVICES based on NUM_GPUS
# =============================================================================
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    # Build comma-separated list of MIG UUIDs
    CUDA_DEVICES=""
    for ((i=0; i<NUM_GPUS && i<${#MIG_UUIDS[@]}; i++)); do
        if [ -n "$CUDA_DEVICES" ]; then
            CUDA_DEVICES="${CUDA_DEVICES},${MIG_UUIDS[$i]}"
        else
            CUDA_DEVICES="${MIG_UUIDS[$i]}"
        fi
    done
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo "DDP Mode: Using $NUM_GPUS MIG slices ($DIST_BACKEND backend)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
elif [ "$NUM_GPUS" -eq 1 ] 2>/dev/null; then
    export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[0]}"
    echo "Single GPU Mode: Using 1 MIG slice"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

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
echo "  Num GPUs (MIG slices): $NUM_GPUS"
echo "  Dist backend: $DIST_BACKEND"
echo "  Launcher: $([ "$NUM_GPUS" -gt 1 ] && echo "torchrun DDP" || echo "single process")"
echo ""

# =============================================================================
# Data Download (if requested)
# =============================================================================
if [ "$DOWNLOAD_DATA" = true ]; then
    echo "=============================================="
    echo "Downloading fMoW dataset..."
    echo "=============================================="
    echo "Target directory: $DATA_DIR"
    echo "This will download ~350GB of rgb images"
    echo ""
    
    # Run the download script (non-interactive for SLURM)
    # Using --use-rgb for full resolution images
    FRACTION_ARG=""
    if [ -n "$DATA_FRACTION" ]; then
        FRACTION_ARG="--fraction $DATA_FRACTION"
    fi
    python scripts/download_fmow.py \
        --output-dir "$DATA_DIR" \
        --split all \
        --workers 32 \
        --use-rgb $FRACTION_ARG <<< "y"
    
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
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    # Multi-GPU DDP training via torchrun
    echo "Launching DDP with torchrun (nproc_per_node=$NUM_GPUS)..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$((29500 + RANDOM % 1000)) \
        train_dynamicvis_pretrain.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --launcher pytorch \
        --dist-backend $DIST_BACKEND \
        $RESUME $NO_WANDB $DATA_ROOT_ARG
else
    # Single GPU training
    python train_dynamicvis_pretrain.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $RESUME $NO_WANDB $DATA_ROOT_ARG
fi

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

