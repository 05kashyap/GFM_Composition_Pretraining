#!/bin/bash
# Training script for DynamicVis on fMoW dataset with S3 streaming
#
# Usage:
#   ./run_dynamicvis_training.sh                    # Single GPU, default config
#   ./run_dynamicvis_training.sh --gpus 4           # Multi-GPU training
#   ./run_dynamicvis_training.sh --debug            # Debug mode (few steps)
#   ./run_dynamicvis_training.sh --resume           # Resume from latest checkpoint
#   ./run_dynamicvis_training.sh --no-wandb         # Disable wandb logging

set -e

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
fi

# Default values
CONFIG="configs_dynamicvis/fmow_classification/dynamicvis_b_fmow_s3.py"
WORK_DIR="outputs/fmow_dynamicvis_b_s3"
GPUS=1
BATCH_SIZE=32
EPOCHS=100
LR=1e-4
DEBUG=false
RESUME=""
NO_WANDB=""
EXTRA_ARGS=""

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
        --gpus)
            GPUS="$2"
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
            DEBUG=true
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
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Print configuration
echo "=============================================="
echo "DynamicVis Training on fMoW (S3 Streaming)"
echo "=============================================="
echo "Config: $CONFIG"
echo "Work directory: $WORK_DIR"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Debug mode: $DEBUG"
echo "WandB: $([ -z "$NO_WANDB" ] && echo "Enabled" || echo "Disabled")"
echo "=============================================="

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Training will use CPU."
    GPUS=0
fi

# Build command
CMD="python train_dynamicvis.py $CONFIG \
    --work-dir $WORK_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    $RESUME $NO_WANDB $EXTRA_ARGS"

if [ $GPUS -gt 1 ]; then
    # Multi-GPU training with torchrun
    CMD="torchrun --nproc_per_node=$GPUS train_dynamicvis.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --launcher pytorch \
        $RESUME $NO_WANDB $EXTRA_ARGS"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Run training
eval $CMD

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $WORK_DIR"
