#!/bin/bash
# =============================================================================
# BoVW Loss Ablation Study
# =============================================================================
# Runs 6 training configurations to compare different loss combinations:
#   1. EMD only
#   2. EMD + MIL
#   3. EMD + MIL + CLS (full)
#   4. MIL + CLS
#   5. MIL only
#   6. CLS only
#
# All runs use:
#   - 10 epochs
#   - No pretrained backbone (training from scratch)
#   - Same hyperparameters otherwise
#
# Usage:
#   bash run_bovw_ablation.sh              # Run all 6 configs sequentially
#   bash run_bovw_ablation.sh --parallel   # Submit all as separate SLURM jobs
#   bash run_bovw_ablation.sh --dry-run    # Print commands without executing
# =============================================================================

set -euo pipefail

# ── Configuration ──
NUM_EPOCHS=20
DATA_FRACTION=0.5  # Use 50% of data to speed up ablation
BATCH_SIZE=32
NUM_GPUS=8

# Parse args
PARALLEL=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)   PARALLEL=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --epochs)     NUM_EPOCHS="$2"; shift 2 ;;
        --data-fraction) DATA_FRACTION="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-gpus)   NUM_GPUS="$2"; shift 2 ;;
        *)            shift ;;
    esac
done

echo "=============================================="
echo "BoVW Loss Ablation Study"
echo "=============================================="
echo "Epochs:        $NUM_EPOCHS"
echo "Data fraction: $DATA_FRACTION"
echo "Batch size:    $BATCH_SIZE"
echo "Num GPUs:      $NUM_GPUS"
echo "Parallel:      $PARALLEL"
echo "Dry run:       $DRY_RUN"
echo "=============================================="
echo ""

# ── Define ablation configurations ──
# Format: "name|lambda_emd|lambda_cls|lambda_mil"
CONFIGS=(
    # "emd_cls|1.0|0.5|0"
    # "emd_only|1.0|0|0"
    # "emd_mil|1.0|0|0.25"
    "emd_mil_cls|1.0|0.5|0.25"
    "mil_cls|0|0.5|0.25"
    # "mil_only|0|0|0.25"
    # "cls_only|0|0.5|0"
)

# ── Run each configuration ──
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name lambda_emd lambda_cls lambda_mil <<< "$config"

    echo "----------------------------------------------"
    echo "Configuration: $name"
    echo "  λ_emd = $lambda_emd"
    echo "  λ_cls = $lambda_cls"
    echo "  λ_mil = $lambda_mil"
    echo "----------------------------------------------"

    # Build command
    CMD="bash run_gpu_bovw_training.sh \
        --epochs $NUM_EPOCHS \
        --data-fraction $DATA_FRACTION \
        --batch-size $BATCH_SIZE \
        --num-gpus $NUM_GPUS \
        --lambda-emd $lambda_emd \
        --lambda-cls $lambda_cls \
        --lambda-mil $lambda_mil \
        --output-dir outputs/bovw_ablation_${name}"
#        --no-pretrained \

    if $DRY_RUN; then
        echo "[DRY RUN] Would execute:"
        echo "  $CMD"
        echo ""
    elif $PARALLEL; then
        # Submit as separate SLURM job
        echo "Submitting SLURM job for: $name"
        sbatch run_gpu_bovw_training.sh \
            --epochs $NUM_EPOCHS \
            --data-fraction $DATA_FRACTION \
            --batch-size $BATCH_SIZE \
            --num-gpus $NUM_GPUS \
            --lambda-emd $lambda_emd \
            --lambda-cls $lambda_cls \
            --lambda-mil $lambda_mil \
            --output-dir "outputs/bovw_ablation_${name}"
        echo ""
    else
        # Run sequentially
        echo "Running: $name"
        bash run_gpu_bovw_training.sh \
            --epochs $NUM_EPOCHS \
            --data-fraction $DATA_FRACTION \
            --batch-size $BATCH_SIZE \
            --num-gpus $NUM_GPUS \
            --lambda-emd $lambda_emd \
            --lambda-cls $lambda_cls \
            --lambda-mil $lambda_mil \
                --output-dir "outputs/bovw_ablation_${name}"
        echo ""
    fi
done

echo "=============================================="
echo "Ablation study complete!"
echo ""
echo "Results saved to:"
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name _ _ _ <<< "$config"
    echo "  outputs/bovw_ablation_${name}/"
done
echo ""
echo "To compare results, check W&B or run:"
echo "  python -c \"import json; [print(f'{d}: {json.load(open(d+\"/metrics.json\"))[\"best_loss\"]}') for d in sorted(glob.glob('outputs/bovw_ablation_*'))]\""
echo "=============================================="
