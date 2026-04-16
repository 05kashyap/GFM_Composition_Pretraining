#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# =============================================================================
# DynamicVis Change Detection Evaluation (LEVIR-CD) - SLURM Script
# =============================================================================
# Usage:
#   sbatch run_gpu_change_detection_eval.sh
#   sbatch run_gpu_change_detection_eval.sh --epochs 5 --batch-size 4
#   sbatch run_gpu_change_detection_eval.sh --eval-only --cd-checkpoint outputs/change_detection/best_cd_model.pth
#   sbatch run_gpu_change_detection_eval.sh --dry-run
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "DynamicVis Change Detection Eval - SLURM"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=============================================="

mkdir -p logs

# -- Conda --
CONDA_BASE=""
if [ -d "/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3" ]; then
    CONDA_BASE="/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi

if [ -n "$CONDA_BASE" ]; then
    __conda_setup="$($CONDA_BASE/bin/conda shell.bash hook 2>/dev/null)"
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

# -- DynamicVis clone guard --
if [ ! -d "architectures/DynamicVis/dynamicvis" ]; then
    echo "DynamicVis not found. Cloning..."
    mkdir -p architectures
    git clone https://github.com/KyanChen/DynamicVis.git architectures/DynamicVis
fi

export DYNAMICVIS_ROOT="$(pwd)/architectures/DynamicVis"
export PYTHONPATH="$(pwd):$DYNAMICVIS_ROOT:${PYTHONPATH:-}"

# -- MIG GPU config --
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

MIG_UUIDS=()
for ((i=0; i<${#MIG_GPU0[@]}; i++)); do
    MIG_UUIDS+=("${MIG_GPU0[$i]}")
    MIG_UUIDS+=("${MIG_GPU1[$i]}")
done

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# -- Defaults --
DATA_ROOT="data/eval/LEVIR CD"
OUTPUT_DIR="outputs/change_detection"
DYNAMICVIS_CONFIG="architectures/DynamicVis/configs_DynamicVis/LEVIR-CD/dynamicvis_b_2X_levircd_mamba.py"
BACKBONE_CHECKPOINT="outputs/bovw_training_8262/epoch_20.pth"
CD_CHECKPOINT=""
PATCH_SIZE=512
STRIDE=512
EPOCHS=2
BATCH_SIZE=32
NUM_WORKERS=4
LR=1e-4
WEIGHT_DECAY=1e-4
MIN_LR=1e-6
FPN_OUT_CHANNELS=256
SEED=42
LOG_EVERY=10
NUM_VISUALIZE=4
TRAIN_BACKBONE=""
NO_AUGMENT=""
EVAL_ONLY=""
DRY_RUN=""
MIG_INDEX=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)            DATA_ROOT="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --dynamicvis-config)    DYNAMICVIS_CONFIG="$2"; shift 2 ;;
        --backbone-checkpoint)  BACKBONE_CHECKPOINT="$2"; shift 2 ;;
        --cd-checkpoint)        CD_CHECKPOINT="$2"; shift 2 ;;
        --patch-size)           PATCH_SIZE="$2"; shift 2 ;;
        --stride)               STRIDE="$2"; shift 2 ;;
        --epochs)               EPOCHS="$2"; shift 2 ;;
        --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)          NUM_WORKERS="$2"; shift 2 ;;
        --lr)                   LR="$2"; shift 2 ;;
        --weight-decay)         WEIGHT_DECAY="$2"; shift 2 ;;
        --min-lr)               MIN_LR="$2"; shift 2 ;;
        --fpn-out-channels)     FPN_OUT_CHANNELS="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --log-every)            LOG_EVERY="$2"; shift 2 ;;
        --num-visualize)        NUM_VISUALIZE="$2"; shift 2 ;;
        --train-backbone)       TRAIN_BACKBONE="--train-backbone"; shift ;;
        --no-augment)           NO_AUGMENT="--no-augment"; shift ;;
        --eval-only)            EVAL_ONLY="--eval-only"; shift ;;
        --dry-run)              DRY_RUN="--dry-run"; shift ;;
        --mig-index)            MIG_INDEX="$2"; shift 2 ;;
        *)                      EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ "$MIG_INDEX" -ge "${#MIG_UUIDS[@]}" ] || [ "$MIG_INDEX" -lt 0 ]; then
    echo "Error: --mig-index must be in [0, $(( ${#MIG_UUIDS[@]} - 1 ))]."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[$MIG_INDEX]}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: LEVIR-CD data directory not found: $DATA_ROOT"
    exit 1
fi

if [ ! -f "$BACKBONE_CHECKPOINT" ]; then
    echo "Error: backbone checkpoint not found: $BACKBONE_CHECKPOINT"
    exit 1
fi

if [ ! -f "$DYNAMICVIS_CONFIG" ]; then
    echo "Error: DynamicVis config not found: $DYNAMICVIS_CONFIG"
    exit 1
fi

if [ -n "$EVAL_ONLY" ] && [ -n "$CD_CHECKPOINT" ] && [ ! -f "$CD_CHECKPOINT" ]; then
    echo "Error: CD checkpoint not found: $CD_CHECKPOINT"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Data root:            $DATA_ROOT"
echo "  Output dir:           $OUTPUT_DIR"
echo "  DynamicVis config:    $DYNAMICVIS_CONFIG"
echo "  Backbone checkpoint:  $BACKBONE_CHECKPOINT"
echo "  CD checkpoint:        ${CD_CHECKPOINT:-$OUTPUT_DIR/best_cd_model.pth}"
echo "  Patch size:           $PATCH_SIZE"
echo "  Stride:               $STRIDE"
echo "  Epochs:               $EPOCHS"
echo "  Batch size:           $BATCH_SIZE"
echo "  Num workers:          $NUM_WORKERS"
echo "  LR:                   $LR"
echo "  Weight decay:         $WEIGHT_DECAY"
echo "  Min LR:               $MIN_LR"
echo "  FPN channels:         $FPN_OUT_CHANNELS"
echo "  Seed:                 $SEED"
echo "  Train backbone:       ${TRAIN_BACKBONE:-no}"
echo "  Eval only:            ${EVAL_ONLY:-no}"
echo "  CUDA device:          $CUDA_VISIBLE_DEVICES"
echo ""

command -v nvidia-smi &>/dev/null && nvidia-smi

PY_ARGS=(
    --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR"
    --dynamicvis-config "$DYNAMICVIS_CONFIG"
    --backbone-checkpoint "$BACKBONE_CHECKPOINT"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --num-epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --min-lr "$MIN_LR"
    --fpn-out-channels "$FPN_OUT_CHANNELS"
    --seed "$SEED"
    --log-every "$LOG_EVERY"
    --num-visualize "$NUM_VISUALIZE"
)

if [ -n "$CD_CHECKPOINT" ]; then
    PY_ARGS+=(--cd-checkpoint-path "$CD_CHECKPOINT")
fi
if [ -n "$TRAIN_BACKBONE" ]; then
    PY_ARGS+=("$TRAIN_BACKBONE")
fi
if [ -n "$NO_AUGMENT" ]; then
    PY_ARGS+=("$NO_AUGMENT")
fi
if [ -n "$EVAL_ONLY" ]; then
    PY_ARGS+=("$EVAL_ONLY")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [ -n "$DRY_RUN" ]; then
    echo "Dry run command:"
    printf 'python eval/change-detection/main.py %q ' "${PY_ARGS[@]}"
    echo ""
    exit 0
fi

LOG_PATH="logs/change_detection_eval_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

echo "Running eval/change-detection/main.py ..."
python eval/change-detection/main.py "${PY_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo ""
echo "=============================================="
echo "Change detection run completed at: $(date)"
echo "Log file: $LOG_PATH"
echo "=============================================="
