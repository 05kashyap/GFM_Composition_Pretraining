# SatBae

# Basic training
./run_dynamicvis_training.sh

# Debug mode (2 epochs, small batch)
./run_dynamicvis_training.sh --debug

# Multi-GPU training
./run_dynamicvis_training.sh --gpus 4

# Custom configuration
./run_dynamicvis_training.sh --epochs 50 --batch-size 64 --lr 2e-4

# Disable wandb
./run_dynamicvis_training.sh --no-wandb

# Resume training
./run_dynamicvis_training.sh --resume