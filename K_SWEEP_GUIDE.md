# BoVW K-Sweep Orchestration Guide

This document describes how to run the full BoVW pipeline sweep across multiple vocabulary sizes using the new orchestration scripts.

## Overview

The K-sweep workflow automates the process of:
1. Building visual vocabularies for different K values (32, 64, 128, 256, 512, 1024)
2. Generating histogram targets for each vocabulary
3. Training BoVW models with each vocabulary size
4. Evaluating trained models on the AID CBIR dataset
5. Computing silhouette scores for clustering quality
6. Aggregating results into a single summary file

All outputs are stored under `outputs/bovw_k_sweep/`.

## Quick Start

### Dry-Run (Recommended First Step)

Test the orchestration logic without submitting jobs:

```bash
bash run_bovw_k_sweep.sh --dry-run --k-values "256 512" --epochs 5
```

This will print all the commands that would be executed.

### Single K Test

Run a single K value (e.g., K=512) on a small subset:

```bash
bash run_bovw_k_sweep.sh --k-values "512" --data-fraction 0.05 --epochs 20
```

### Full Sweep

Run the complete sweep with default settings (10% data, all K values, 100 epochs):

```bash
bash run_bovw_k_sweep.sh
```

This will sequentially process:
- K = 32, 64, 128, 256, 512, 1024

### Custom Parameters

```bash
# Smaller subset and faster training
bash run_bovw_k_sweep.sh \
  --data-fraction 0.05 \
  --epochs 50 \
  --batch-size 16 \
  --num-gpus 4

# Resume from a specific K (useful if sweep was interrupted)
bash run_bovw_k_sweep.sh --start-k 256

# Custom K values
bash run_bovw_k_sweep.sh --k-values "128 256 512 1024"

# Skip pretrained backbone (train from scratch)
bash run_bovw_k_sweep.sh --no-pretrained
```

### Background Execution

If you want it to keep running after you disconnect, use:

```bash
nohup bash run_bovw_k_sweep.sh > logs/k_sweep.nohup.log 2>&1 &
tail -f logs/k_sweep.nohup.log
```

## Output Structure

```
outputs/bovw_k_sweep/
├── k_sweep_results.json          # Main results file (machine-readable)
├── k_sweep_results.csv           # Results in CSV format (if exported)
├── k_sweep_plots.png             # Visualization plots (if generated)
│
├── k_32/                          # Per-K subdirectory
│   ├── vocab/                     # Phase 2: vocabulary
│   │   ├── centroids.npy          # (K, 1024) cluster centroids
│   │   ├── ground_cost.npy        # (K, K) distance matrix for EMD
│   │   └── cluster_sizes.png      # K-means histogram
│   ├── histograms/                # Phase 3: histogram targets
│   │   ├── histograms.npy         # (N, K) soft histogram targets
│   │   ├── cell_ids.npy           # (N,) cell indices
│   │   └── cell_labels.npy        # (N,) fMoW category labels
│   ├── training/                  # Phase 4: model training
│   │   ├── epoch_*.pth            # Checkpoints
│   │   ├── final_model.pth        # Final trained model
│   │   └── final_backbone.pth     # Final backbone (for evaluation)
│   ├── eval_aid/                  # Phase 5: AID evaluation
│   │   ├── faiss_index_fold_*.idx # FAISS indices per fold
│   │   └── eval.log               # Evaluation output
│   ├── phase2_build_vocabulary.log
│   ├── phase3_generate_histograms.log
│   ├── phase4_training.log
│   └── phase5_cbir_eval.log
│
├── k_64/
├── k_128/
├── ...
└── k_1024/
```

## Results Summary Format

The `k_sweep_results.json` file has the following structure:

```json
{
  "sweep_metadata": {
    "created_at": "2026-05-06T07:43:17.105972Z",
    "k_values": [32, 64, 128, 256, 512, 1024],
    "data_fraction": 0.1,
    "epochs": 100,
    "batch_size": 32
  },
  "results": [
    {
      "k": 256,
      "silhouette_score": 0.4521,
      "cbir_metrics": {
        "recall_at_1": 0.3456,
        "recall_at_5": 0.5678,
        "recall_at_10": 0.7890,
        "map_at_1": 0.2111,
        "map_at_5": 0.3222,
        "map_at_10": 0.4333,
        "recall_at_1_std": 0.0123,
        "map_at_1_std": 0.0456
      }
    },
    ...
  ]
}
```

## Post-Sweep Analysis

After the sweep completes, generate analysis artifacts:

```bash
# Print summary table
python3 scripts/analyze_k_sweep.py

# Export to CSV (for Excel/analysis)
python3 scripts/analyze_k_sweep.py --export-csv

# Generate plots (requires matplotlib)
python3 scripts/analyze_k_sweep.py --plot
```

This will generate:
- Console table with formatted metrics
- `k_sweep_results.csv` for spreadsheet analysis
- `k_sweep_plots.png` with three plots:
  - Silhouette score vs K
  - CBIR Recall@1 vs K
  - CBIR mAP@1 vs K

## Key Concepts

### Silhouette Score
- Measures clustering quality of the visual vocabulary
- Computed from vocabulary centroids and histogram soft-assignments
- Range: [-1, 1], higher is better
- May be N/A if insufficient data or clusters

### CBIR Metrics
- **Recall@K**: Fraction of relevant items in top-K results
- **mAP@K**: Mean Average Precision at top-K results
- Computed via stratified K-fold cross-validation on AID dataset
- Reported as mean ± std across folds
- Higher is better

### Data Fraction
- Percentage of fMoW training data to use
- Default: 0.1 (10%)
- Applies consistently across Phase 2 (vocabulary), Phase 3 (histograms), and Phase 4 (training)
- Allows faster iteration during development

## Troubleshooting

### Job Stuck or Timed Out

If a job doesn't complete within 2 hours, the orchestration script will report a timeout and skip that K value.

To resume from a specific K:

```bash
bash run_bovw_k_sweep.sh --start-k 256
```

### Missing CBIR Metrics

The AID evaluation logs are searched in `logs/cbir_eval_*.log`. If no log is found:

1. Verify `run_gpu_cbir_eval.sh` completed successfully
2. Check that logs are written to `logs/` directory
3. Run evaluation manually:
   ```bash
   sbatch run_gpu_cbir_eval.sh --checkpoint outputs/bovw_k_sweep/k_512/training/final_backbone.pth
   ```

### Silhouette Score is N/A

This occurs when:
- Insufficient data points (< 3 samples)
- Too few unique clusters (< 2 classes)
- Centroid or histogram file is corrupted

Verify Phase 2 and 3 outputs are present:

```bash
ls outputs/bovw_k_sweep/k_512/vocab/centroids.npy
ls outputs/bovw_k_sweep/k_512/histograms/histograms.npy
```

## Data Source Notes

The sweep assumes the following directory structure:

```
/mnt/usb/
├── patch_tokens_bovw/       # Phase 1 DINOv3 patch tokens (required)
├── bovw_vocabulary/         # NOT used by sweep (per-K vocabs created)
├── bovw_histograms/         # NOT used by sweep (per-K histograms created)
├── fmow/                     # fMoW dataset for training (required)
│   ├── train/<category>/<location>/*.jpg
│   └── val/<category>/<location>/*.jpg
```

The sweep will NOT overwrite or use the existing `/mnt/usb/bovw_vocabulary` or `/mnt/usb/bovw_histograms` directories; instead, it creates K-specific outputs under `outputs/bovw_k_sweep/k_<K>/`.

## Advanced Usage

### Parallel Submission (Not Recommended)

The current orchestration is sequential and waits for each job to complete before moving to the next phase. For parallel submission, you can modify the script to submit all K jobs at once (requires external job coordination).

### Resuming Partial Sweeps

If the sweep was interrupted, resume from a specific K:

```bash
bash run_bovw_k_sweep.sh --start-k 256
```

This will skip K values that have already been processed (32, 64, 128) and resume from K=256.

### Custom Manifest

Use a different fMoW manifest for training (e.g., validation split):

```bash
bash run_bovw_k_sweep.sh --manifest data/fmow_manifest_val.json
```

## Performance Estimates

With default settings (10% data, 8 GPUs, 100 epochs):

| Phase | Time per K | Notes |
|-------|-----------|-------|
| Phase 2 (Vocabulary) | ~15-30 min | FAISS K-means, parallelizable |
| Phase 3 (Histograms) | ~30-60 min | Soft-assignment computation |
| Phase 4 (Training) | ~2-4 hours | Depends on epochs and batch size |
| Phase 5 (CBIR Eval) | ~30 min | AID dataset, k-fold cross-validation |
| **Per-K Total** | ~4-6 hours | Cumulative time |
| **Full Sweep (6 K)** | ~24-36 hours | Sequential execution |

Actual times depend on hardware, data I/O, and system load.

## See Also

- [CONTEXT.md](CONTEXT.md) — Detailed BoVW pipeline architecture
- [README.md](README.md) — Main project README
- [run_bovw_ablation.sh](run_bovw_ablation.sh) — Similar orchestration for loss ablation studies
