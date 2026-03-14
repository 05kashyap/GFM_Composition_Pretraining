#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# Composition-Aware DynamicVis Training — SLURM Script
# =============================================================================
# Trains DynamicVis backbone with a projection head mapping to DINOv3
# embedding space, optimised with contrastive + smoothness + consistency loss.
#
# Prerequisites:
#   1. sbatch run_gpu_embed_patches.sh   → cache DINOv3 patch embeddings
#   2. sbatch run_gpu_cluster_viz.sh --save-cluster-data  → outputs/cluster_data/
#
# Usage:
#   sbatch run_gpu_composition_training.sh
#   sbatch run_gpu_composition_training.sh --epochs 50 --batch-size 64
#   sbatch run_gpu_composition_training.sh --num-gpus 2
#   sbatch run_gpu_composition_training.sh --resume
#   sbatch run_gpu_composition_training.sh --debug
# =============================================================================

echo "=============================================="
echo "Composition-Aware DynamicVis Training — SLURM"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=============================================="

mkdir -p logs

# ── Conda ──
CONDA_BASE=""
if [ -d "/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3" ]; then
    CONDA_BASE="/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi

if [ -n "$CONDA_BASE" ]; then
    __conda_setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
    if [ $? -eq 0 ]; then eval "$__conda_setup"; else
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
        else export PATH="$CONDA_BASE/bin:$PATH"; fi
    fi
    unset __conda_setup
fi
conda activate dynamicvis

# ── DynamicVis clone guard ──
if [ ! -d "architectures/DynamicVis/dynamicvis" ]; then
    echo "DynamicVis not found. Cloning..."
    mkdir -p architectures
    git clone https://github.com/KyanChen/DynamicVis.git architectures/DynamicVis
fi

export PYTHONPATH="$(pwd):$(pwd)/architectures/DynamicVis:$PYTHONPATH"

# ── MIG GPU config ──
# All available MIG slices, organized by physical GPU.
# GPU 0 (PCIe bus 26:00.0) and GPU 1 (PCIe bus 89:00.0)
# Use: nvidia-smi -L to discover available MIG UUIDs on your node
#
# NCCL constraint: only 1 MIG slice per physical GPU (same bus ID = "Duplicate GPU").
#   → --num-gpus 1-2: uses NCCL backend (fast, 1 slice per physical GPU)
#   → --num-gpus 3-8: uses gloo backend (allows multiple slices per physical GPU)
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

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── Defaults ──
CONFIG="configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py"
WORK_DIR="outputs/fmow_dynamicvis_b_composition"
BATCH_SIZE=32
EPOCHS=100
LR=1e-3
RESUME=""
NO_WANDB=""
NUM_GPUS=8
DIST_BACKEND="auto"
CLUSTER_DATA_DIR=""
L_COMP=""
L_CONTRAST=""
L_SMOOTH=""

if [ -n "$SLURM_JOB_ID" ]; then
    WORK_DIR="${WORK_DIR}_${SLURM_JOB_ID}"
fi

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)         CONFIG="$2";          shift 2 ;;
        --work-dir)       WORK_DIR="$2";        shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";      shift 2 ;;
        --epochs)         EPOCHS="$2";          shift 2 ;;
        --lr)             LR="$2";              shift 2 ;;
        --debug)          EPOCHS=2; BATCH_SIZE=8; shift ;;
        --resume)
            if [[ -n "${2:-}" && "$2" != --* ]]; then
                RESUME="--resume $2"; shift 2
            else
                RESUME="--resume auto"; shift
            fi
            ;;
        --no-wandb)       NO_WANDB="--no-wandb";  shift ;;
        --num-gpus)       NUM_GPUS="$2";        shift 2 ;;
        --dist-backend)   DIST_BACKEND="$2";    shift 2 ;;
        --cluster-data-dir) CLUSTER_DATA_DIR="$2"; shift 2 ;;
        --l-comp)         L_COMP="$2";           shift 2 ;;
        --l-contrast)     L_CONTRAST="$2";       shift 2 ;;
        --l-smooth)       L_SMOOTH="$2";         shift 2 ;;
        *)                shift ;;
    esac
done

# ── Resolve backend ──
if [ "$NUM_GPUS" -gt "${#MIG_UUIDS[@]}" ]; then
    NUM_GPUS=${#MIG_UUIDS[@]}
fi

if [ "$DIST_BACKEND" = "auto" ]; then
    if [ "$NUM_GPUS" -le 2 ]; then DIST_BACKEND="nccl"; else DIST_BACKEND="gloo"; fi
fi
if [ "$DIST_BACKEND" = "nccl" ] && [ "$NUM_GPUS" -gt 2 ]; then
    DIST_BACKEND="gloo"
fi

# ── CUDA devices ──
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    CUDA_DEVICES=""
    for ((i=0; i<NUM_GPUS && i<${#MIG_UUIDS[@]}; i++)); do
        [ -n "$CUDA_DEVICES" ] && CUDA_DEVICES="${CUDA_DEVICES},"
        CUDA_DEVICES="${CUDA_DEVICES}${MIG_UUIDS[$i]}"
    done
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
elif [ "$NUM_GPUS" -eq 1 ] 2>/dev/null; then
    export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[0]}"
fi

CLUSTER_ARG=""
if [ -n "$CLUSTER_DATA_DIR" ]; then
    CLUSTER_ARG="--cluster-data-dir $CLUSTER_DATA_DIR"
fi

LOSS_ARGS=""
[ -n "$L_COMP" ]     && LOSS_ARGS="$LOSS_ARGS --l-comp $L_COMP"
[ -n "$L_CONTRAST" ] && LOSS_ARGS="$LOSS_ARGS --l-contrast $L_CONTRAST"
[ -n "$L_SMOOTH" ]   && LOSS_ARGS="$LOSS_ARGS --l-smooth $L_SMOOTH"

# ── Summary ──
echo ""
echo "Configuration:"
echo "  Config:          $CONFIG"
echo "  Work dir:        $WORK_DIR"
echo "  Batch size:      $BATCH_SIZE"
echo "  Epochs:          $EPOCHS"
echo "  LR:              $LR"
echo "  Cluster data:    ${CLUSTER_DATA_DIR:-outputs/cluster_data (default)}"
echo "  λ_comp:          ${L_COMP:-config default}"
echo "  λ_contrast:      ${L_CONTRAST:-config default}"
echo "  λ_smooth:        ${L_SMOOTH:-config default}"
echo "  Num GPUs (MIG):  $NUM_GPUS"
echo "  Dist backend:    $DIST_BACKEND"
echo ""

command -v nvidia-smi &>/dev/null && nvidia-smi

echo ""
echo "Starting composition-aware training..."
echo ""

if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$((29500 + RANDOM % 1000)) \
        train_dynamicvis_composition.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --launcher pytorch \
        --dist-backend $DIST_BACKEND \
        $RESUME $NO_WANDB $CLUSTER_ARG $LOSS_ARGS
else
    python train_dynamicvis_composition.py $CONFIG \
        --work-dir $WORK_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $RESUME $NO_WANDB $CLUSTER_ARG $LOSS_ARGS
fi

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
