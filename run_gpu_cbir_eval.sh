#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# =============================================================================
# DynamicVis CBIR Evaluation (AID / ForestNet) - SLURM Script
# =============================================================================
# Usage:
#   sbatch run_gpu_cbir_eval.sh
#   sbatch run_gpu_cbir_eval.sh --checkpoint outputs/bovw_training_8262/final_backbone.pth
#   sbatch run_gpu_cbir_eval.sh --data-dir data/eval/AID --max-per-class 20
#   sbatch run_gpu_cbir_eval.sh --dataset forestnet --data-dir data/eval/deep/downloads/ForestNetDataset
#   sbatch run_gpu_cbir_eval.sh --model-type prithvi2
#   sbatch run_gpu_cbir_eval.sh --dry-run
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "DynamicVis CBIR Evaluation - SLURM"
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
DATASET="aid"
DATA_DIR="data/eval/AID"
FORESTNET_MODE="both"
MODEL_TYPE="dynamicvis"
MODEL_PATH="outputs/bovw_training_8262/epoch_20.pth"
CONFIG_PATH="architectures/DynamicVis/configs_DynamicVis/AID/dynamicvis_b_aid_mamba.py"
PRITHVI_MODEL_PATH="weights/Prithvi_EO_V2_600M.pt"
EMBEDDING_DIM=768
IMG_SIZE=512
BATCH_SIZE=32
NUM_WORKERS=4
SEED=42
INDEX_TYPE="Flat"
NLIST=100
INDEX_DIR="outputs/cbir_index"
MAX_PER_CLASS=""
MAX_TRAIN=""
MAX_TEST=""
SAVE_INDEX=""
DRY_RUN=""
MIG_INDEX=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)        DATASET="$2"; shift 2 ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --forestnet-mode) FORESTNET_MODE="$2"; shift 2 ;;
        --model-type)     MODEL_TYPE="$2"; shift 2 ;;
        --checkpoint|--model-path)
                          MODEL_PATH="$2"; shift 2 ;;
        --config|--config-path)
                          CONFIG_PATH="$2"; shift 2 ;;
        --embedding-dim)  EMBEDDING_DIM="$2"; shift 2 ;;
        --img-size)       IMG_SIZE="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --index-type)     INDEX_TYPE="$2"; shift 2 ;;
        --nlist)          NLIST="$2"; shift 2 ;;
        --index-dir)      INDEX_DIR="$2"; shift 2 ;;
        --max-per-class)  MAX_PER_CLASS="$2"; shift 2 ;;
        --max-train)      MAX_TRAIN="$2"; shift 2 ;;
        --max-test)       MAX_TEST="$2"; shift 2 ;;
        --save-index)     SAVE_INDEX="--save_index"; shift ;;
        --dry-run)        DRY_RUN="--dry_run"; shift ;;
        --mig-index)      MIG_INDEX="$2"; shift 2 ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ "$MODEL_TYPE" != "dynamicvis" ] && [ "$MODEL_TYPE" != "prithvi" ] && [ "$MODEL_TYPE" != "prithvi2" ] && [ "$MODEL_TYPE" != "prithvi_v2" ]; then
    echo "Error: --model-type must be one of: dynamicvis, prithvi, prithvi2, prithvi_v2"
    exit 1
fi

if [ "$MODEL_TYPE" = "prithvi" ] || [ "$MODEL_TYPE" = "prithvi2" ] || [ "$MODEL_TYPE" = "prithvi_v2" ]; then
    if [ "$MODEL_PATH" = "outputs/bovw_training_8262/epoch_20.pth" ]; then
        MODEL_PATH="$PRITHVI_MODEL_PATH"
    fi
    if [ "$EMBEDDING_DIM" = "768" ]; then
        EMBEDDING_DIM=512
    fi
    CONFIG_PATH=""
fi

if [ "$DATASET" != "aid" ] && [ "$DATASET" != "forestnet" ]; then
    echo "Error: --dataset must be one of: aid, forestnet"
    exit 1
fi

if [ "$FORESTNET_MODE" != "both" ] && [ "$FORESTNET_MODE" != "12" ] && [ "$FORESTNET_MODE" != "4" ]; then
    echo "Error: --forestnet-mode must be one of: both, 12, 4"
    exit 1
fi

# If user selects ForestNet but keeps default AID data_dir, switch to ForestNet default path.
if [ "$DATASET" = "forestnet" ] && [ "$DATA_DIR" = "data/eval/AID" ]; then
    DATA_DIR="data/eval/deep/downloads/ForestNetDataset"
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

if [ "$MODEL_TYPE" != "dynamicvis" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Prithvi v2 checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: data directory not found: $DATA_DIR"
    exit 1
fi

SAVE_INDEX_STATUS="no"
if [ -n "$SAVE_INDEX" ]; then
    SAVE_INDEX_STATUS="yes"
fi

DRY_RUN_STATUS="no"
if [ -n "$DRY_RUN" ]; then
    DRY_RUN_STATUS="yes"
fi

echo ""
echo "Configuration:"
echo "  Dataset:         $DATASET"
echo "  Data dir:        $DATA_DIR"
echo "  ForestNet mode:  $FORESTNET_MODE"
echo "  Model type:      $MODEL_TYPE"
echo "  Checkpoint:      $MODEL_PATH"
echo "  Config:          ${CONFIG_PATH:-n/a}"
echo "  Embedding dim:   $EMBEDDING_DIM"
echo "  Image size:      $IMG_SIZE"
echo "  Batch size:      $BATCH_SIZE"
echo "  Num workers:     $NUM_WORKERS"
echo "  Seed:            $SEED"
echo "  Index type:      $INDEX_TYPE"
echo "  NList:           $NLIST"
echo "  Index dir:       $INDEX_DIR"
echo "  Max per class:   ${MAX_PER_CLASS:-all}"
echo "  Max train:       ${MAX_TRAIN:-all}"
echo "  Max test:        ${MAX_TEST:-all}"
echo "  Save index:      $SAVE_INDEX_STATUS"
echo "  Dry run:         $DRY_RUN_STATUS"
echo "  MIG index:       $MIG_INDEX"
echo "  CUDA device:     $CUDA_VISIBLE_DEVICES"
echo ""

command -v nvidia-smi &>/dev/null && nvidia-smi

PY_ARGS=(
    --dataset "$DATASET"
    --data_dir "$DATA_DIR"
    --model_type "$MODEL_TYPE"
    --model_path "$MODEL_PATH"
    --embedding_dim "$EMBEDDING_DIM"
    --img_size "$IMG_SIZE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --index_type "$INDEX_TYPE"
    --nlist "$NLIST"
    --index_dir "$INDEX_DIR"
)

if [ -n "$CONFIG_PATH" ]; then
    PY_ARGS+=(--config_path "$CONFIG_PATH")
fi

if [ "$DATASET" = "forestnet" ]; then
    PY_ARGS+=(--forestnet_mode "$FORESTNET_MODE")
fi

if [ -n "$MAX_PER_CLASS" ]; then
    PY_ARGS+=(--max_per_class "$MAX_PER_CLASS")
fi
if [ -n "$MAX_TRAIN" ]; then
    PY_ARGS+=(--max_train "$MAX_TRAIN")
fi
if [ -n "$MAX_TEST" ]; then
    PY_ARGS+=(--max_test "$MAX_TEST")
fi
if [ -n "$SAVE_INDEX" ]; then
    PY_ARGS+=("$SAVE_INDEX")
fi
if [ -n "$DRY_RUN" ]; then
    PY_ARGS+=("$DRY_RUN")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

LOG_PATH="logs/cbir_eval_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

echo "Running eval/cbir/main.py ..."
python eval/cbir/main.py "${PY_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo ""
echo "=============================================="
echo "CBIR evaluation completed at: $(date)"
echo "Log file: $LOG_PATH"
echo "=============================================="
