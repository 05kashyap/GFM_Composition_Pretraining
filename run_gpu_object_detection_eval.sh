#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# =============================================================================
# DynamicVis LEVIR-ship Object Detection - SLURM Script
# =============================================================================
# Usage:
#   sbatch run_gpu_object_detection_eval.sh
#   sbatch run_gpu_object_detection_eval.sh --num-epochs 10 --batch-size 2
#   sbatch run_gpu_object_detection_eval.sh --eval-only --detector-checkpoint outputs/object_detection/best_detector.pth
#   sbatch run_gpu_object_detection_eval.sh --max-train 500 --max-val 200 --max-test 200
#   sbatch run_gpu_object_detection_eval.sh --model-type prithvi_v2
#   sbatch run_gpu_object_detection_eval.sh --dry-run
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "DynamicVis LEVIR-ship Detection - SLURM"
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
DATA_ROOT="data/eval/object-det/LEVIR-ship/LEVIR-ship"
OUTPUT_DIR="outputs/object_detection"
MODEL_TYPE="dynamicvis"
DYNAMICVIS_CONFIG="configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py"
BACKBONE_CHECKPOINT="outputs/bovw_training_8262/epoch_20.pth"
PRITHVI_BACKBONE_CHECKPOINT="weights/Prithvi_EO_V2_600M.pt"
DETECTOR_CHECKPOINT=""

IMG_SIZE=512
FPN_OUT_CHANNELS=256
NUM_EPOCHS=5
BATCH_SIZE=4
NUM_WORKERS=4
LR=1e-4
WEIGHT_DECAY=1e-4
LOG_EVERY=10
SEED=42
IOU_THRESHOLDS="0.1,0.2,0.3,0.4,0.5"
SCORE_THRESH=0.3
NUM_VISUALIZE=5
MAX_TRAIN=""
MAX_VAL=""
MAX_TEST=""
TRAIN_BACKBONE=""
NO_AUGMENT=""
EVAL_ONLY=""
DRY_RUN=""
MIG_INDEX=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)            DATA_ROOT="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --model-type)           MODEL_TYPE="$2"; shift 2 ;;
        --dynamicvis-config)    DYNAMICVIS_CONFIG="$2"; shift 2 ;;
        --backbone-checkpoint)  BACKBONE_CHECKPOINT="$2"; shift 2 ;;
        --detector-checkpoint)  DETECTOR_CHECKPOINT="$2"; shift 2 ;;
        --img-size)             IMG_SIZE="$2"; shift 2 ;;
        --fpn-out-channels)     FPN_OUT_CHANNELS="$2"; shift 2 ;;
        --num-epochs)           NUM_EPOCHS="$2"; shift 2 ;;
        --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)          NUM_WORKERS="$2"; shift 2 ;;
        --lr)                   LR="$2"; shift 2 ;;
        --weight-decay)         WEIGHT_DECAY="$2"; shift 2 ;;
        --log-every)            LOG_EVERY="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --iou-thresholds)       IOU_THRESHOLDS="$2"; shift 2 ;;
        --score-thresh)         SCORE_THRESH="$2"; shift 2 ;;
        --num-visualize)        NUM_VISUALIZE="$2"; shift 2 ;;
        --max-train)            MAX_TRAIN="$2"; shift 2 ;;
        --max-val)              MAX_VAL="$2"; shift 2 ;;
        --max-test)             MAX_TEST="$2"; shift 2 ;;
        --train-backbone)       TRAIN_BACKBONE="--train-backbone"; shift ;;
        --no-augment)           NO_AUGMENT="--no-augment"; shift ;;
        --eval-only)            EVAL_ONLY="--eval-only"; shift ;;
        --dry-run)              DRY_RUN="--dry-run"; shift ;;
        --mig-index)            MIG_INDEX="$2"; shift 2 ;;
        *)                      EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ "$MODEL_TYPE" != "dynamicvis" ] && [ "$MODEL_TYPE" != "prithvi" ] && [ "$MODEL_TYPE" != "prithvi2" ] && [ "$MODEL_TYPE" != "prithvi_v2" ]; then
    echo "Error: --model-type must be one of: dynamicvis, prithvi, prithvi2, prithvi_v2"
    exit 1
fi

if [ "$MODEL_TYPE" = "prithvi" ] || [ "$MODEL_TYPE" = "prithvi2" ] || [ "$MODEL_TYPE" = "prithvi_v2" ]; then
    if [ "$BACKBONE_CHECKPOINT" = "outputs/bovw_training_8262/epoch_20.pth" ]; then
        BACKBONE_CHECKPOINT="$PRITHVI_BACKBONE_CHECKPOINT"
    fi
    DYNAMICVIS_CONFIG=""
fi

if [ "$MIG_INDEX" -ge "${#MIG_UUIDS[@]}" ] || [ "$MIG_INDEX" -lt 0 ]; then
    echo "Error: --mig-index must be in [0, $(( ${#MIG_UUIDS[@]} - 1 ))]."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[$MIG_INDEX]}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: data directory not found: $DATA_ROOT"
    exit 1
fi

if [ "$MODEL_TYPE" = "dynamicvis" ] && [ ! -f "$DYNAMICVIS_CONFIG" ]; then
    echo "Error: DynamicVis config not found: $DYNAMICVIS_CONFIG"
    exit 1
fi

if [ ! -f "$BACKBONE_CHECKPOINT" ]; then
    echo "Error: backbone checkpoint not found: $BACKBONE_CHECKPOINT"
    exit 1
fi

if [ -n "$DETECTOR_CHECKPOINT" ] && [ ! -f "$DETECTOR_CHECKPOINT" ]; then
    echo "Error: detector checkpoint not found: $DETECTOR_CHECKPOINT"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Data root:            $DATA_ROOT"
echo "  Output dir:           $OUTPUT_DIR"
echo "  Model type:           $MODEL_TYPE"
echo "  DynamicVis config:    $DYNAMICVIS_CONFIG"
echo "  Backbone checkpoint:  $BACKBONE_CHECKPOINT"
echo "  Detector checkpoint:  ${DETECTOR_CHECKPOINT:-none}"
echo "  Image size:           $IMG_SIZE"
echo "  FPN out channels:     $FPN_OUT_CHANNELS"
echo "  Epochs:               $NUM_EPOCHS"
echo "  Batch size:           $BATCH_SIZE"
echo "  Num workers:          $NUM_WORKERS"
echo "  LR:                   $LR"
echo "  Weight decay:         $WEIGHT_DECAY"
echo "  Log every:            $LOG_EVERY"
echo "  Seed:                 $SEED"
echo "  IoU thresholds:       $IOU_THRESHOLDS"
echo "  Score threshold:      $SCORE_THRESH"
echo "  Num visualize:        $NUM_VISUALIZE"
echo "  Max train:            ${MAX_TRAIN:-all}"
echo "  Max val:              ${MAX_VAL:-all}"
echo "  Max test:             ${MAX_TEST:-all}"
echo "  Train backbone:       ${TRAIN_BACKBONE:-no}"
echo "  Eval only:            ${EVAL_ONLY:-no}"
echo "  CUDA device:          $CUDA_VISIBLE_DEVICES"
echo ""

command -v nvidia-smi &>/dev/null && nvidia-smi

PY_ARGS=(
    --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR"
    --model-type "$MODEL_TYPE"
    --backbone-checkpoint "$BACKBONE_CHECKPOINT"
    --img-size "$IMG_SIZE"
    --fpn-out-channels "$FPN_OUT_CHANNELS"
    --num-epochs "$NUM_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --log-every "$LOG_EVERY"
    --seed "$SEED"
    --iou-thresholds "$IOU_THRESHOLDS"
    --score-thresh "$SCORE_THRESH"
    --num-visualize "$NUM_VISUALIZE"
)

if [ -n "$DYNAMICVIS_CONFIG" ]; then
    PY_ARGS+=(--dynamicvis-config "$DYNAMICVIS_CONFIG")
fi

if [ -n "$DETECTOR_CHECKPOINT" ]; then
    PY_ARGS+=(--detector-checkpoint "$DETECTOR_CHECKPOINT")
fi
if [ -n "$MAX_TRAIN" ]; then
    PY_ARGS+=(--max-train "$MAX_TRAIN")
fi
if [ -n "$MAX_VAL" ]; then
    PY_ARGS+=(--max-val "$MAX_VAL")
fi
if [ -n "$MAX_TEST" ]; then
    PY_ARGS+=(--max-test "$MAX_TEST")
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
if [ -n "$DRY_RUN" ]; then
    PY_ARGS+=("$DRY_RUN")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

LOG_PATH="logs/object_detection_eval_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

echo "Running eval/object-detection/main.py ..."
python eval/object-detection/main.py "${PY_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo ""
echo "=============================================="
echo "Object detection run completed at: $(date)"
echo "Log file: $LOG_PATH"
echo "=============================================="
