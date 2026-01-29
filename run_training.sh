#!/bin/bash
# Training script for fMoW with DynamicVis

# Load environment variables
set -a
source .env
set +a

# Default training run
python train_fmow.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir outputs/fmow_dynamicvis \
    --num-workers 4

# For quick testing/debugging, use:
# python train_fmow.py --epochs 1 --train-steps 100 --val-steps 50 --batch-size 16