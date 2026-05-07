# BoVW K-Sweep: Quick Start Reference

## Files Created

| File | Purpose |
|------|---------|
| `run_bovw_k_sweep.sh` | Main orchestration script (executable) |
| `scripts/k_sweep_utilities.py` | CBIR log parsing + result aggregation |
| `scripts/analyze_k_sweep.py` | Post-sweep analysis (CSV export, plots) |
| `K_SWEEP_GUIDE.md` | Comprehensive guide with examples |

## Recommended Setup

### 1. Verify Data Location

Ensure your data is at `/mnt/usb/`:

```bash
ls -la /mnt/usb/patch_tokens_bovw  # DINOv3 patch tokens (Phase 1 output)
ls -la /mnt/usb/fmow               # fMoW dataset images
```

### 2. Quick Validation (Optional)

Test the orchestration on a single K value with 5% data and 10 epochs:

```bash
bash run_bovw_k_sweep.sh \
    --k-values "256" \
    --data-fraction 0.05 \
    --epochs 10
```

Expect: ~30 min per phase, watch the log for any issues.

### 3. Run Full Sweep

Production sweep with 10% data, 100 epochs, all K values:

```bash
bash run_bovw_k_sweep.sh
```

Expect: ~24-36 hours total (sequential), or run in a `tmux`/`screen` session.

To detach it from your shell, use:

```bash
nohup bash run_bovw_k_sweep.sh > logs/k_sweep.nohup.log 2>&1 &
tail -f logs/k_sweep.nohup.log
```

### 4. Monitor Progress

While the sweep runs, check the summary file periodically:

```bash
# View current results
cat outputs/bovw_k_sweep/k_sweep_results.json | python3 -m json.tool

# Watch logs for a specific K
tail -f outputs/bovw_k_sweep/k_512/phase4_training.log
```

### 5. Analyze Results

After the sweep completes:

```bash
# Print formatted table
python3 scripts/analyze_k_sweep.py

# Export to CSV + generate plots
python3 scripts/analyze_k_sweep.py --export-csv --plot

# View plots
open outputs/bovw_k_sweep/k_sweep_plots.png  # or `display` if on Linux
```

## Common Adjustments

```bash
# Faster iteration (5% data, 30 epochs)
bash run_bovw_k_sweep.sh --data-fraction 0.05 --epochs 30

# Subset of K values
bash run_bovw_k_sweep.sh --k-values "128 256 512 1024"

# Resume if interrupted
bash run_bovw_k_sweep.sh --start-k 512

# Train without pretrained backbone
bash run_bovw_k_sweep.sh --no-pretrained

# Dry run to see what will execute
bash run_bovw_k_sweep.sh --dry-run
```

## Expected Outputs

After completion, you will have:

```
outputs/bovw_k_sweep/
├── k_sweep_results.json       ← Main results file
├── k_sweep_results.csv        ← For spreadsheet viewing
├── k_sweep_plots.png          ← Silhouette & CBIR metrics plots
├── k_32/   ├── k_64/   ├── k_128/   ├── k_256/   ├── k_512/   └── k_1024/
│   ├── vocab/                 (vocabulary artifacts)
│   ├── histograms/            (histogram targets)
│   ├── training/              (trained checkpoints)
│   └── eval_aid/              (CBIR evaluation logs)
```

## Key Results to Examine

The `k_sweep_results.json` contains, for each K:
- **silhouette_score**: Clustering quality (-1 to 1, higher is better)
- **recall_at_1/5/10**: CBIR retrieval recall (0 to 1, higher is better)
- **map_at_1/5/10**: CBIR retrieval mAP (0 to 1, higher is better)

## Next Steps

1. **Identify Best K**: Look for the K that balances silhouette score + CBIR metrics
2. **Train Production Model**: Run a final training on 100% data with the chosen K
3. **Downstream Evaluation**: Evaluate on other datasets (UC Merced, LEVIR-CD, etc.)

## Need Help?

- **For detailed guide**: See `K_SWEEP_GUIDE.md`
- **For troubleshooting**: See `K_SWEEP_GUIDE.md` → Troubleshooting section
- **For CONTEXT**: See `CONTEXT.md` for BoVW architecture details
