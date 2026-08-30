#!/bin/bash

# =============================================================================
# BoVW Vocabulary Size (K) Parameter Sweep
# =============================================================================
# Orchestrates the full BoVW pipeline (Phase 2-4) and CBIR evaluation for
# multiple vocabulary sizes: 32, 64, 128, 256, 512, 1024.
#
# For each K:
#   1. Rebuild visual vocabulary using run_gpu_build_vocabulary.sh
#   2. Regenerate histograms using run_gpu_generate_histograms.sh
#   3. Train DynamicVis using run_gpu_bovw_training.sh
#   4. Evaluate on AID using run_gpu_cbir_eval.sh
#   5. Compute silhouette score and extract CBIR metrics
#   6. Append results to consolidated summary file
#
# All outputs stored under: outputs/bovw_k_sweep/
#
# Usage:
#   bash run_bovw_k_sweep.sh
#   bash run_bovw_k_sweep.sh --k-values "256 512 1024"
#   bash run_bovw_k_sweep.sh --dry-run
#   bash run_bovw_k_sweep.sh --data-fraction 0.05  # Use 5% instead of 10%
#   bash run_bovw_k_sweep.sh --start-k 128         # Start from K=128
#   bash run_bovw_k_sweep.sh --epochs 50            # Custom epochs (default 100)
#
# Background execution:
#   nohup bash run_bovw_k_sweep.sh > logs/k_sweep.nohup.log 2>&1 &
#   tail -f logs/k_sweep.nohup.log
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "BoVW K-Sweep Orchestration"
echo "=============================================="
echo "Start: $(date)"
echo "PWD:   $(pwd)"
echo "=============================================="

mkdir -p logs outputs/bovw_k_sweep

# ── Conda bootstrap ──
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

export PYTHONPATH="$(pwd):$(pwd)/architectures/DynamicVis:${PYTHONPATH:-}"

# ── Defaults ──
K_VALUES=(32 64 128 256 512 1024)
DATA_FRACTION=0.1
PATCH_TOKEN_DIR="/mnt/usb/patch_tokens_bovw"
FMOW_DATA_ROOT="/mnt/usb/fmow"
MANIFEST="data/fmow_manifest_train.json"
EPOCHS=20
BATCH_SIZE=32
NUM_GPUS=4
DISABLE_PRETRAINED="false"
DRY_RUN="false"
START_K=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --k-values)          K_VALUES=($2); shift 2 ;;
        --data-fraction)     DATA_FRACTION="$2"; shift 2 ;;
        --epochs)            EPOCHS="$2"; shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; shift 2 ;;
        --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
        --patch-token-dir)   PATCH_TOKEN_DIR="$2"; shift 2 ;;
        --data-root)         FMOW_DATA_ROOT="$2"; shift 2 ;;
        --manifest)          MANIFEST="$2"; shift 2 ;;
        --no-pretrained)     DISABLE_PRETRAINED="true"; shift ;;
        --dry-run)           DRY_RUN="true"; shift ;;
        --start-k)           START_K="$2"; shift 2 ;;
        *)                   echo "Unknown arg: $1"; shift ;;
    esac
done

# ── Configuration summary ──
echo ""
echo "Configuration:"
echo "  K values:         ${K_VALUES[@]}"
echo "  Data fraction:    $DATA_FRACTION"
echo "  Patch token dir:  $PATCH_TOKEN_DIR"
echo "  Data root:        $FMOW_DATA_ROOT"
echo "  Manifest:         $MANIFEST"
echo "  Epochs:           $EPOCHS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Num GPUs:         $NUM_GPUS"
echo "  Pretrained:       $([ "$DISABLE_PRETRAINED" = "true" ] && echo "disabled" || echo "enabled")"
echo "  Dry run:          $DRY_RUN"
echo "  Start K:          ${START_K:-all}"
echo ""

# ── Create summary file ──
SUMMARY_DIR="outputs/bovw_k_sweep"
SUMMARY_FILE="$SUMMARY_DIR/k_sweep_results.json"
mkdir -p "$SUMMARY_DIR"

# Initialize summary if not exists
if [ ! -f "$SUMMARY_FILE" ]; then
    python3 << EOF
import json
from datetime import datetime, timezone

summary = {
    "sweep_metadata": {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "k_values": [],
        "data_fraction": $DATA_FRACTION,
        "epochs": $EPOCHS,
        "batch_size": $BATCH_SIZE,
    },
    "results": []
}
with open("$SUMMARY_FILE", "w") as f:
    json.dump(summary, f, indent=2)
EOF
fi

# ── Helper function: wait for SLURM job ──
wait_for_slurm_job() {
    local job_id=$1
    local max_wait=$((37200))  # 2 hours max
    local elapsed=0
    local check_interval=10

    while [ $elapsed -lt $max_wait ]; do
        if ! squeue -j "$job_id" &>/dev/null; then
            # Job finished
            local status=$(sacct -j "$job_id" --format=State --noheader | head -1 | xargs)
            if [ "$status" = "COMPLETED" ]; then
                return 0
            else
                echo "  Job $job_id finished with status: $status"
                return 1
            fi
        fi
        echo "  Job $job_id running... (elapsed: ${elapsed}s)"
        sleep $check_interval
        ((elapsed += check_interval))
    done

    echo "  Job $job_id timed out after ${max_wait}s"
    return 1
}

# ── Main sweep loop ──
START_INDEX=0
if [ -n "$START_K" ]; then
    for i in "${!K_VALUES[@]}"; do
        if [ "${K_VALUES[$i]}" = "$START_K" ]; then
            START_INDEX=$i
            break
        fi
    done
    echo "Starting from K=${K_VALUES[$START_INDEX]} (index $START_INDEX)"
fi

for ((i = START_INDEX; i < ${#K_VALUES[@]}; i++)); do
    K="${K_VALUES[$i]}"
    K_DIR="$SUMMARY_DIR/k_${K}"
    mkdir -p "$K_DIR"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║ K = $K ($(( i + 1 ))/${#K_VALUES[@]}) - $(date)    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2: Build vocabulary
    # ─────────────────────────────────────────────────────────────────────
    echo "[Phase 2] Building vocabulary for K=$K..."
    VOCAB_DIR="$K_DIR/vocab"
    mkdir -p "$VOCAB_DIR"

    PHASE2_LOG="$K_DIR/phase2_build_vocabulary.log"
    PHASE2_CMD="sbatch run_gpu_build_vocabulary.sh \
        --K $K \
        --patch-token-dir $PATCH_TOKEN_DIR \
        --output-dir $VOCAB_DIR \
        --data-fraction $DATA_FRACTION"

    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would execute:"
        echo "  $PHASE2_CMD"
        sleep 2
        continue
    fi

    echo "  Command: $PHASE2_CMD"
    PHASE2_JID=$(eval "$PHASE2_CMD" | grep -oP 'Submitted batch job \K\d+')
    echo "  Job ID: $PHASE2_JID"

    if wait_for_slurm_job "$PHASE2_JID"; then
        echo "  ✓ Phase 2 completed"
    else
        echo "  ✗ Phase 2 failed for K=$K, skipping remaining phases"
        continue
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3: Generate histograms
    # ─────────────────────────────────────────────────────────────────────
    echo "[Phase 3] Generating histograms for K=$K..."
    HIST_DIR="$K_DIR/histograms"
    mkdir -p "$HIST_DIR"

    PHASE3_LOG="$K_DIR/phase3_generate_histograms.log"
    PHASE3_CMD="sbatch run_gpu_generate_histograms.sh \
        --patch-token-dir $PATCH_TOKEN_DIR \
        --vocab-dir $VOCAB_DIR \
        --manifest $MANIFEST \
        --output-dir $HIST_DIR \
        --data-fraction $DATA_FRACTION"

    echo "  Command: $PHASE3_CMD"
    PHASE3_JID=$(eval "$PHASE3_CMD" | grep -oP 'Submitted batch job \K\d+')
    echo "  Job ID: $PHASE3_JID"

    if wait_for_slurm_job "$PHASE3_JID"; then
        echo "  ✓ Phase 3 completed"
    else
        echo "  ✗ Phase 3 failed for K=$K, skipping remaining phases"
        continue
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3b: Extract cell labels (once per sweep, reusable)
    # ─────────────────────────────────────────────────────────────────────
    if [ ! -f "$HIST_DIR/cell_labels.npy" ]; then
        echo "[Phase 3b] Extracting cell labels..."
        python scripts/extract_manifest_labels.py \
            --manifest $MANIFEST \
            --output-dir $HIST_DIR
        echo "  ✓ Phase 3b completed"
    else
        echo "[Phase 3b] Cell labels already exist, skipping..."
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Phase 4: Train BoVW model
    # ─────────────────────────────────────────────────────────────────────
    echo "[Phase 4] Training BoVW model for K=$K..."
    TRAIN_DIR="$K_DIR/training"
    mkdir -p "$TRAIN_DIR"

    NO_PRETRAINED_ARG=""
    if [ "$DISABLE_PRETRAINED" = "true" ]; then
        NO_PRETRAINED_ARG="--no-pretrained"
    fi

    PHASE4_LOG="$K_DIR/phase4_training.log"
    PHASE4_CMD="sbatch run_gpu_bovw_training.sh \
        --vocab-size $K \
        --histogram-dir $HIST_DIR \
        --vocab-dir $VOCAB_DIR \
        --cell-labels $HIST_DIR/cell_labels.npy \
        --data-root $FMOW_DATA_ROOT \
        --manifest $MANIFEST \
        --output-dir $TRAIN_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-gpus $NUM_GPUS \
        --data-fraction $DATA_FRACTION \
        $NO_PRETRAINED_ARG"

    echo "  Command: $PHASE4_CMD"
    PHASE4_JID=$(eval "$PHASE4_CMD" | grep -oP 'Submitted batch job \K\d+')
    echo "  Job ID: $PHASE4_JID"

    if wait_for_slurm_job "$PHASE4_JID"; then
        echo "  ✓ Phase 4 completed"
    else
        echo "  ✗ Phase 4 failed for K=$K, skipping evaluation"
        continue
    fi

    # Find the best checkpoint
    CHECKPOINT=$(find "$TRAIN_DIR" -name "final_backbone.pth" -o -name "final_model.pth" -o -name "epoch_*.pth" | tail -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "  ✗ No checkpoint found in $TRAIN_DIR"
        continue
    fi
    echo "  Using checkpoint: $CHECKPOINT"

    # ─────────────────────────────────────────────────────────────────────
    # Phase 5: CBIR Evaluation on AID
    # ─────────────────────────────────────────────────────────────────────
    echo "[Phase 5] Evaluating on AID dataset for K=$K..."
    EVAL_DIR="$K_DIR/eval_aid"
    mkdir -p "$EVAL_DIR"

    PHASE5_LOG="$K_DIR/phase5_cbir_eval.log"
    PHASE5_CMD="sbatch run_gpu_cbir_eval.sh \
        --checkpoint $CHECKPOINT \
        --dataset aid \
        --index-dir $EVAL_DIR"

    echo "  Command: $PHASE5_CMD"
    PHASE5_JID=$(eval "$PHASE5_CMD" | grep -oP 'Submitted batch job \K\d+')
    echo "  Job ID: $PHASE5_JID"

    if wait_for_slurm_job "$PHASE5_JID"; then
        echo "  ✓ Phase 5 completed"
    else
        echo "  ✗ Phase 5 failed for K=$K"
        continue
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Phase 6: Collect results and compute metrics
    # ─────────────────────────────────────────────────────────────────────
    echo "[Phase 6] Collecting results for K=$K..."

    python3 - "$K" "$K_DIR" "$VOCAB_DIR" "$HIST_DIR" "$TRAIN_DIR" "$SUMMARY_FILE" << 'PYSCRIPT'
import sys
import json
import numpy as np
from pathlib import Path

K = sys.argv[1]
K_DIR = sys.argv[2]
VOCAB_DIR = sys.argv[3]
HIST_DIR = sys.argv[4]
TRAIN_DIR = sys.argv[5]
SUMMARY_FILE = sys.argv[6]

sys.path.insert(0, str(Path.cwd()))

result = {"k": int(K)}

# 1. Silhouette score from centroids and cell embeddings
try:
    from scripts.pipeline_utils import silhouette_optional
    centroids_path = Path(VOCAB_DIR) / "centroids.npy"
    histograms_path = Path(HIST_DIR) / "histograms.npy"

    if centroids_path.exists() and histograms_path.exists():
        centroids = np.load(centroids_path)
        histograms = np.load(histograms_path)

        # Use histogram assignments as pseudo-labels (argmax)
        if histograms.shape[0] > 0:
            labels = np.argmax(histograms, axis=1)
            silhouette = silhouette_optional(centroids, labels, seed=42)
            result["silhouette_score"] = silhouette if silhouette is not None else None
        else:
            result["silhouette_score"] = None
    else:
        result["silhouette_score"] = None
        print(f"  Warning: Could not compute silhouette (missing centroids or histograms)")

except Exception as e:
    result["silhouette_score"] = None
    print(f"  Warning: Silhouette computation failed: {e}")

# 2. Parse CBIR evaluation log for AID metrics
try:
    from scripts.k_sweep_utilities import parse_cbir_log
    cbir_log_path = Path(f"logs/cbir_eval_*.log")
    # Find the most recent CBIR eval log
    import glob
    logs = sorted(glob.glob(str(Path("logs") / "cbir_eval_*.log")))
    if logs:
        cbir_metrics = parse_cbir_log(logs[-1])
        result["cbir_metrics"] = cbir_metrics
    else:
        result["cbir_metrics"] = None
        print(f"  Warning: No CBIR log found")
except Exception as e:
    result["cbir_metrics"] = None
    print(f"  Warning: CBIR metric extraction failed: {e}")

# 3. Append to summary
try:
    with open(SUMMARY_FILE, "r") as f:
        summary = json.load(f)

    # Add K value if not already present
    if int(K) not in summary["sweep_metadata"]["k_values"]:
        summary["sweep_metadata"]["k_values"].append(int(K))

    # Append or replace result for this K
    existing_idx = next((i for i, r in enumerate(summary["results"]) if r.get("k") == int(K)), None)
    if existing_idx is not None:
        summary["results"][existing_idx] = result
    else:
        summary["results"].append(result)

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Results appended to summary: {SUMMARY_FILE}")
except Exception as e:
    print(f"  Error writing summary: {e}")

PYSCRIPT

done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ K-Sweep Complete!                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "To view results:"
echo "  cat $SUMMARY_FILE | python3 -m json.tool"
echo ""
echo "Completed: $(date)"
