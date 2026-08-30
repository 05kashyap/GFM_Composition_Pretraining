#!/bin/bash
#SBATCH --job-name=AryanKashyapN
#SBATCH --partition=small
#SBATCH --gres=gpu:1g.24gb:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 # NEVER CHANGE THIS
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# =============================================================================
# BoVW Visualization Generation (SLURM)
# =============================================================================
# Usage:
#   sbatch run_gpu_generate_bovw_visualizations.sh
#   sbatch run_gpu_generate_bovw_visualizations.sh --skip 5
#   sbatch run_gpu_generate_bovw_visualizations.sh --manifest data/fmow_manifest_val.json
#   sbatch run_gpu_generate_bovw_visualizations.sh --checkpoint-dir outputs/bovw_training_8262
#   sbatch run_gpu_generate_bovw_visualizations.sh --output-dir outputs/visualizations_val
#   sbatch run_gpu_generate_bovw_visualizations.sh --mig-index 3
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "BoVW Visualizations - SLURM"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "PWD:    $(pwd)"
echo "=============================================="

mkdir -p logs

# -- Conda bootstrap --
CONDA_BASE=""
if [ -d "/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3" ]; then
    CONDA_BASE="/data/home/slb1028/work/AryanKashyapN-221AI012/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi

if [ -n "$CONDA_BASE" ]; then
    __conda_setup="$($CONDA_BASE/bin/conda shell.bash hook 2>/dev/null || true)"
    if [ -n "$__conda_setup" ]; then
        eval "$__conda_setup"
    elif [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_BASE/bin:$PATH"
    fi
    unset __conda_setup
fi

conda activate dynamicvis

# -- DynamicVis clone/import guard --
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

# -- Defaults --
MANIFEST="data/fmow_manifest_train.json"
HISTOGRAM_DIR="outputs/bovw_histograms"
VOCAB_DIR="outputs/bovw_vocabulary"
PATCH_TOKEN_DIR="outputs/patch_tokens_bovw"
DINOV3_EMBED_DIR="outputs/preprocess_cache_dinov3"
CHECKPOINT_DIR="outputs/bovw_checkpoints"
OUTPUT_DIR="outputs/visualizations"
CELL_LABELS=""
SEED=42
SKIP=""
MIG_INDEX=0
USE_GPU=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --manifest)         MANIFEST="$2"; shift 2 ;;
        --histogram-dir)    HISTOGRAM_DIR="$2"; shift 2 ;;
        --vocab-dir)        VOCAB_DIR="$2"; shift 2 ;;
        --patch-token-dir)  PATCH_TOKEN_DIR="$2"; shift 2 ;;
        --dinov3-embed-dir) DINOV3_EMBED_DIR="$2"; shift 2 ;;
        --checkpoint-dir)   CHECKPOINT_DIR="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --cell-labels)      CELL_LABELS="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --skip)             SKIP="$2"; shift 2 ;;
        --mig-index)        MIG_INDEX="$2"; shift 2 ;;
        --cpu)              USE_GPU=0; shift ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ "$USE_GPU" -eq 1 ]; then
    if [ "$MIG_INDEX" -ge "${#MIG_UUIDS[@]}" ] || [ "$MIG_INDEX" -lt 0 ]; then
        echo "Error: --mig-index must be in [0, $(( ${#MIG_UUIDS[@]} - 1 ))]."
        exit 1
    fi

    export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[$MIG_INDEX]}"
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=0
    export NCCL_IB_DISABLE=1
    export NCCL_DEBUG=WARN
fi

echo ""
echo "Configuration:"
echo "  Manifest:         $MANIFEST"
echo "  Histogram dir:    $HISTOGRAM_DIR"
echo "  Vocab dir:        $VOCAB_DIR"
echo "  Patch token dir:  $PATCH_TOKEN_DIR"
echo "  DINO cache dir:   $DINOV3_EMBED_DIR"
echo "  Checkpoint dir:   $CHECKPOINT_DIR"
echo "  Output dir:       $OUTPUT_DIR"
echo "  Cell labels:      ${CELL_LABELS:-auto}"
echo "  Seed:             $SEED"
echo "  Skip:             ${SKIP:-none}"
echo "  Use GPU:          $USE_GPU"
if [ "$USE_GPU" -eq 1 ]; then
    echo "  MIG index:        $MIG_INDEX"
    echo "  CUDA device:      $CUDA_VISIBLE_DEVICES"
fi
echo ""

if [ "$USE_GPU" -eq 1 ]; then
    command -v nvidia-smi &>/dev/null && nvidia-smi
fi

PY_ARGS=(
    --manifest "$MANIFEST"
    --histogram-dir "$HISTOGRAM_DIR"
    --vocab-dir "$VOCAB_DIR"
    --patch-token-dir "$PATCH_TOKEN_DIR"
    --dinov3-embed-dir "$DINOV3_EMBED_DIR"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
)

if [ -n "$CELL_LABELS" ]; then
    PY_ARGS+=(--cell-labels "$CELL_LABELS")
fi
if [ -n "$SKIP" ]; then
    PY_ARGS+=(--skip "$SKIP")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

LOG_PATH="logs/generate_bovw_visualizations_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

echo "Running scripts/generate_bovw_visualizations.py ..."
python -u scripts/generate_bovw_visualizations.py "${PY_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo ""
echo "=============================================="
echo "Visualization generation completed at: $(date)"
echo "Log file: $LOG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="
