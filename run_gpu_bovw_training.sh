#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# BoVW DynamicVis Training — SLURM Script
# =============================================================================
# Phase 4 of the BoVW composition training pipeline.
# Trains DynamicVis backbone to predict soft histogram distributions over
# a visual vocabulary using Sinkhorn EMD loss.
#
# Prerequisites:
#   1. sbatch run_gpu_extract_patch_tokens.sh  → outputs/patch_tokens_bovw/
#   2. sbatch run_gpu_build_vocabulary.sh      → outputs/bovw_vocabulary/
#   3. sbatch run_gpu_generate_histograms.sh   → outputs/bovw_histograms/
#
# Usage:
#   sbatch run_gpu_bovw_training.sh
#   sbatch run_gpu_bovw_training.sh --epochs 50 --batch-size 64
#   sbatch run_gpu_bovw_training.sh --num-gpus 4
#   sbatch run_gpu_bovw_training.sh --debug  # 2 epochs, small batch
#   sbatch run_gpu_bovw_training.sh --data-fraction 0.25  # Train on 25% of data
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "BoVW DynamicVis Training — SLURM"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
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

export PYTHONPATH="$(pwd):$(pwd)/architectures/DynamicVis:${PYTHONPATH:-}"

# ── MIG GPU config ──
MIG_GPU0=(
    "MIG-418e4605-20dd-5066-8ba8-ecaa0dcd9e2b"
    "MIG-a95be726-a3ac-5445-8427-a9fba33402a4"
    "MIG-093f7e1b-6f21-5e12-bc9e-435e7f019600"
    "MIG-d92daec9-5afd-5a16-adac-861b9216d4f9"
)
MIG_GPU1=(
    "MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0"
    "MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"
    "MIG-283697f2-709b-5149-bf8e-30a0abe75dc4"
    "MIG-83d74367-8c40-525a-9aab-fb561a0a3daa"
)

# Round-robin interleaved list
MIG_UUIDS=()
for ((i=0; i<${#MIG_GPU0[@]}; i++)); do
    MIG_UUIDS+=("${MIG_GPU0[$i]}")
    MIG_UUIDS+=("${MIG_GPU1[$i]}")
done
echo "Available MIG slices: ${#MIG_UUIDS[@]}"

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── Defaults ──
MANIFEST="data/fmow_manifest_train.json"
HISTOGRAM_DIR="outputs/bovw_histograms"
VOCAB_DIR="outputs/bovw_vocabulary"
CELL_LABELS="outputs/bovw_histograms/cell_labels.npy"
DATA_ROOT="data/fmow"
PRETRAINED_BACKBONE="weights/pretrain_dynamicvis_b_bf16_mamba_best_single-label_f1-score_epoch_170.pth"
OUTPUT_DIR="outputs/bovw_training"
VOCAB_SIZE=512
HIDDEN_DIM=512
BATCH_SIZE=32
NUM_EPOCHS=100
LR=5e-4
LAMBDA_EMD=1.0
LAMBDA_CLS=0#0.5
LAMBDA_MIL=0.25
SINKHORN_EPS=0.05
SINKHORN_ITERS=50
NUM_GPUS=8
DIST_BACKEND="auto"
NUM_VIEWS=2
MAX_SAMPLES=""
DATA_FRACTION=""
NO_PRETRAINED=""

if [ -n "${SLURM_JOB_ID:-}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${SLURM_JOB_ID}"
fi

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --manifest)            MANIFEST="$2";            shift 2 ;;
        --histogram-dir)       HISTOGRAM_DIR="$2";       shift 2 ;;
        --vocab-dir)           VOCAB_DIR="$2";           shift 2 ;;
        --cell-labels)         CELL_LABELS="$2";         shift 2 ;;
        --data-root)           DATA_ROOT="$2";           shift 2 ;;
        --pretrained-backbone) PRETRAINED_BACKBONE="$2"; shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2";          shift 2 ;;
        --vocab-size)          VOCAB_SIZE="$2";          shift 2 ;;
        --hidden-dim)          HIDDEN_DIM="$2";          shift 2 ;;
        --batch-size)          BATCH_SIZE="$2";          shift 2 ;;
        --epochs)              NUM_EPOCHS="$2";          shift 2 ;;
        --lr)                  LR="$2";                  shift 2 ;;
        --lambda-emd)          LAMBDA_EMD="$2";          shift 2 ;;
        --lambda-cls)          LAMBDA_CLS="$2";          shift 2 ;;
        --lambda-mil)          LAMBDA_MIL="$2";          shift 2 ;;
        --sinkhorn-eps)        SINKHORN_EPS="$2";        shift 2 ;;
        --sinkhorn-iters)      SINKHORN_ITERS="$2";      shift 2 ;;
        --num-gpus)            NUM_GPUS="$2";            shift 2 ;;
        --dist-backend)        DIST_BACKEND="$2";        shift 2 ;;
        --num-views)           NUM_VIEWS="$2";           shift 2 ;;
        --max-samples)         MAX_SAMPLES="$2";         shift 2 ;;
        --data-fraction)       DATA_FRACTION="$2";       shift 2 ;;
        --no-pretrained)       NO_PRETRAINED="--no-pretrained"; shift ;;
        --debug)               NUM_EPOCHS=2; BATCH_SIZE=8; shift ;;
        *)                     shift ;;
    esac
done

# ── Convert --data-fraction to --max-samples ──
if [ -n "$DATA_FRACTION" ]; then
    if [ -f "$MANIFEST" ]; then
        TOTAL_SAMPLES=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
        MAX_SAMPLES=$(python3 -c "import math; print(int(math.ceil($TOTAL_SAMPLES * $DATA_FRACTION)))")
        echo "Data fraction: $DATA_FRACTION → max_samples: $MAX_SAMPLES (of $TOTAL_SAMPLES total)"
    else
        echo "Warning: Cannot apply --data-fraction, manifest not found: $MANIFEST"
    fi
fi

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

# ── Summary ──
echo ""
echo "Configuration:"
echo "  Manifest:        $MANIFEST"
echo "  Histogram dir:   $HISTOGRAM_DIR"
echo "  Vocab dir:       $VOCAB_DIR"
echo "  Cell labels:     $CELL_LABELS"
echo "  Data root:       $DATA_ROOT"
echo "  Pretrained:      ${NO_PRETRAINED:-(enabled) $PRETRAINED_BACKBONE}"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Vocab size:      $VOCAB_SIZE"
echo "  Hidden dim:      $HIDDEN_DIM"
echo "  Batch size:      $BATCH_SIZE"
echo "  Epochs:          $NUM_EPOCHS"
echo "  LR:              $LR"
echo "  λ_emd:           $LAMBDA_EMD"
echo "  λ_cls:           $LAMBDA_CLS"
echo "  λ_mil:           $LAMBDA_MIL"
echo "  Sinkhorn eps:    $SINKHORN_EPS"
echo "  Sinkhorn iters:  $SINKHORN_ITERS"
echo "  Num views:       $NUM_VIEWS"
echo "  Num GPUs (MIG):  $NUM_GPUS"
echo "  Dist backend:    $DIST_BACKEND"
echo "  Data fraction:   ${DATA_FRACTION:-1.0}"
echo "  Max samples:     ${MAX_SAMPLES:-all}"
echo ""

command -v nvidia-smi &>/dev/null && nvidia-smi

echo ""
echo "Starting BoVW training..."
echo ""

# ── Build Python args ──
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

PYTHON_ARGS="\
    --manifest $MANIFEST \
    --histogram-dir $HISTOGRAM_DIR \
    --vocab-dir $VOCAB_DIR \
    --cell-labels $CELL_LABELS \
    --data-root $DATA_ROOT \
    --pretrained-backbone $PRETRAINED_BACKBONE \
    --output-dir $OUTPUT_DIR \
    --vocab-size $VOCAB_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --lr $LR \
    --lambda-emd $LAMBDA_EMD \
    --lambda-cls $LAMBDA_CLS \
    --lambda-mil $LAMBDA_MIL \
    --sinkhorn-eps $SINKHORN_EPS \
    --sinkhorn-iters $SINKHORN_ITERS \
    --num-views $NUM_VIEWS \
    --dist-backend $DIST_BACKEND \
    $MAX_SAMPLES_ARG \
    $NO_PRETRAINED"

# ── Launch training ──
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$((29500 + RANDOM % 1000)) \
        train_dynamicvis_bovw.py $PYTHON_ARGS \
        2>&1 | tee "logs/bovw_training.log"
else
    python train_dynamicvis_bovw.py $PYTHON_ARGS \
        2>&1 | tee "logs/bovw_training.log"
fi

echo ""
echo "=============================================="
echo "BoVW training completed at: $(date)"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "  Final model:    $OUTPUT_DIR/final_model.pth"
echo "  Final backbone: $OUTPUT_DIR/final_backbone.pth"
echo "=============================================="
