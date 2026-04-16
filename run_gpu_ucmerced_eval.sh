#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# =============================================================================
# DynamicVis UC Merced Evaluation - SLURM Script
# =============================================================================
# Usage:
#   sbatch run_gpu_ucmerced_eval.sh
#   sbatch run_gpu_ucmerced_eval.sh --checkpoint outputs/bovw_training_8262/epoch_20.pth
#   sbatch run_gpu_ucmerced_eval.sh --split-mode kfold --num-folds 5
#   sbatch run_gpu_ucmerced_eval.sh --max-train 1000 --max-test 500
#   sbatch run_gpu_ucmerced_eval.sh --head-epochs 50 --mlp-hidden-dim 768
#   sbatch run_gpu_ucmerced_eval.sh --dry-run
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "DynamicVis UC Merced Evaluation - SLURM"
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

# -- MIG GPU config (single-slice default) --
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
DATA_DIR="data/eval/UCMerced_LandUse"
IMAGES_SUBDIR="Images"
SPLIT_MODE="fixed"
TRAIN_LIST="architectures/DynamicVis/datainfo/ucmerced/train.txt"
VAL_LIST="architectures/DynamicVis/datainfo/ucmerced/val.txt"
NUM_FOLDS=5

MODEL_TYPE="dynamicvis"
MODEL_PATH="outputs/bovw_training_8262/epoch_20.pth"
CONFIG_PATH="architectures/DynamicVis/configs_DynamicVis/UCMerced/dynamicvis_b_uc_mamba.py"
EMBEDDING_DIM=768
IMG_SIZE=512
IN_CHANS=3

BATCH_SIZE=32
NUM_WORKERS=4
SEED=42
MAX_TRAIN=""
MAX_TEST=""
HEAD_EPOCHS=30
HEAD_BATCH_SIZE=256
HEAD_LR="1e-3"
HEAD_WEIGHT_DECAY="1e-4"
MLP_HIDDEN_DIM=512
HEAD_DROPOUT=0.1
HEAD_LOG_INTERVAL=5
SAVE_HEAD_DIR=""
NO_STANDARDIZE=""
DRY_RUN=""
MIG_INDEX=3
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)        DATA_DIR="$2"; shift 2 ;;
        --images-subdir)   IMAGES_SUBDIR="$2"; shift 2 ;;
        --split-mode)      SPLIT_MODE="$2"; shift 2 ;;
        --train-list)      TRAIN_LIST="$2"; shift 2 ;;
        --val-list)        VAL_LIST="$2"; shift 2 ;;
        --num-folds)       NUM_FOLDS="$2"; shift 2 ;;
        --model-type)      MODEL_TYPE="$2"; shift 2 ;;
        --checkpoint|--model-path)
                           MODEL_PATH="$2"; shift 2 ;;
        --config|--config-path)
                           CONFIG_PATH="$2"; shift 2 ;;
        --embedding-dim)   EMBEDDING_DIM="$2"; shift 2 ;;
        --img-size)        IMG_SIZE="$2"; shift 2 ;;
        --in-chans)        IN_CHANS="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --max-train)       MAX_TRAIN="$2"; shift 2 ;;
        --max-test)        MAX_TEST="$2"; shift 2 ;;
        --head-epochs)     HEAD_EPOCHS="$2"; shift 2 ;;
        --head-batch-size) HEAD_BATCH_SIZE="$2"; shift 2 ;;
        --head-lr)         HEAD_LR="$2"; shift 2 ;;
        --head-weight-decay)
                   HEAD_WEIGHT_DECAY="$2"; shift 2 ;;
        --mlp-hidden-dim)  MLP_HIDDEN_DIM="$2"; shift 2 ;;
        --head-dropout)    HEAD_DROPOUT="$2"; shift 2 ;;
        --head-log-interval)
                   HEAD_LOG_INTERVAL="$2"; shift 2 ;;
        --save-head-dir)   SAVE_HEAD_DIR="$2"; shift 2 ;;
        --no-standardize-features)
                   NO_STANDARDIZE="--no_standardize_features"; shift ;;
        --dry-run)         DRY_RUN="--dry_run"; shift ;;
        --mig-index)       MIG_INDEX="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ "$SPLIT_MODE" != "fixed" ] && [ "$SPLIT_MODE" != "kfold" ]; then
    echo "Error: --split-mode must be one of: fixed, kfold"
    exit 1
fi

if [ "$MIG_INDEX" -ge "${#MIG_UUIDS[@]}" ] || [ "$MIG_INDEX" -lt 0 ]; then
    echo "Error: --mig-index must be in [0, $(( ${#MIG_UUIDS[@]} - 1 ))]."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[$MIG_INDEX]}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ "$MODEL_TYPE" = "dynamicvis" ] && [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: DynamicVis config not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: data directory not found: $DATA_DIR"
    exit 1
fi

if [ "$SPLIT_MODE" = "fixed" ]; then
    if [ ! -f "$TRAIN_LIST" ]; then
        echo "Error: train split file not found: $TRAIN_LIST"
        exit 1
    fi
    if [ ! -f "$VAL_LIST" ]; then
        echo "Error: val split file not found: $VAL_LIST"
        exit 1
    fi
fi

SAVE_HEAD_STATUS="no"
if [ -n "$SAVE_HEAD_DIR" ]; then
    SAVE_HEAD_STATUS="yes ($SAVE_HEAD_DIR)"
fi

DRY_RUN_STATUS="no"
if [ -n "$DRY_RUN" ]; then
    DRY_RUN_STATUS="yes"
fi

echo ""
echo "Configuration:"
echo "  Data dir:         $DATA_DIR"
echo "  Images subdir:    $IMAGES_SUBDIR"
echo "  Split mode:       $SPLIT_MODE"
echo "  Train list:       $TRAIN_LIST"
echo "  Val list:         $VAL_LIST"
echo "  Num folds:        $NUM_FOLDS"
echo "  Model type:       $MODEL_TYPE"
echo "  Checkpoint:       $MODEL_PATH"
echo "  Config:           $CONFIG_PATH"
echo "  Embedding dim:    $EMBEDDING_DIM"
echo "  Image size:       $IMG_SIZE"
echo "  Batch size:       $BATCH_SIZE"
echo "  Num workers:      $NUM_WORKERS"
echo "  Seed:             $SEED"
echo "  Max train:        ${MAX_TRAIN:-all}"
echo "  Max test:         ${MAX_TEST:-all}"
echo "  Head epochs:      $HEAD_EPOCHS"
echo "  Head batch size:  $HEAD_BATCH_SIZE"
echo "  Head lr:          $HEAD_LR"
echo "  Head wd:          $HEAD_WEIGHT_DECAY"
echo "  MLP hidden dim:   $MLP_HIDDEN_DIM"
echo "  Head dropout:     $HEAD_DROPOUT"
echo "  Head log intvl:   $HEAD_LOG_INTERVAL"
echo "  Save head:        $SAVE_HEAD_STATUS"
if [ -n "$NO_STANDARDIZE" ]; then
    echo "  Standardize feats: no"
else
    echo "  Standardize feats: yes"
fi
echo "  Dry run:          $DRY_RUN_STATUS"
echo "  MIG index:        $MIG_INDEX"
echo "  CUDA device:      $CUDA_VISIBLE_DEVICES"
echo ""

command -v nvidia-smi &>/dev/null && nvidia-smi

PY_ARGS=(
    --data_dir "$DATA_DIR"
    --images_subdir "$IMAGES_SUBDIR"
    --split_mode "$SPLIT_MODE"
    --train_list "$TRAIN_LIST"
    --val_list "$VAL_LIST"
    --num_folds "$NUM_FOLDS"
    --model_type "$MODEL_TYPE"
    --model_path "$MODEL_PATH"
    --config_path "$CONFIG_PATH"
    --embedding_dim "$EMBEDDING_DIM"
    --img_size "$IMG_SIZE"
    --in_chans "$IN_CHANS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --head_epochs "$HEAD_EPOCHS"
    --head_batch_size "$HEAD_BATCH_SIZE"
    --head_lr "$HEAD_LR"
    --head_weight_decay "$HEAD_WEIGHT_DECAY"
    --mlp_hidden_dim "$MLP_HIDDEN_DIM"
    --head_dropout "$HEAD_DROPOUT"
    --head_log_interval "$HEAD_LOG_INTERVAL"
)

if [ -n "$MAX_TRAIN" ]; then
    PY_ARGS+=(--max_train "$MAX_TRAIN")
fi
if [ -n "$MAX_TEST" ]; then
    PY_ARGS+=(--max_test "$MAX_TEST")
fi
if [ -n "$SAVE_HEAD_DIR" ]; then
    PY_ARGS+=(--save_head_dir "$SAVE_HEAD_DIR")
fi
if [ -n "$NO_STANDARDIZE" ]; then
    PY_ARGS+=("$NO_STANDARDIZE")
fi
if [ -n "$DRY_RUN" ]; then
    PY_ARGS+=("$DRY_RUN")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

LOG_PATH="logs/ucmerced_eval_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

echo "Running eval/ucmerced/main.py ..."
python eval/ucmerced/main.py "${PY_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo ""
echo "=============================================="
echo "UC Merced evaluation completed at: $(date)"
echo "Log file: $LOG_PATH"
echo "=============================================="
