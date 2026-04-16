# SatBae

SatBae is a satellite foundation model project built on the DynamicVis backbone.
The current primary training method is a BoVW-style pipeline on fMoW that distills
structure from DINOv3 patch tokens into DynamicVis using Sinkhorn EMD supervision.

- Primary method: BoVW DynamicVis training on fMoW (implemented and actively used).
- In-domain support: fMoW evaluation scripts for both simplified and pretrain-style heads.
- Downstream support: retrieval, scene classification, and change detection under eval/.
- Experimental/legacy paths still present: vanilla bbox pretraining and QSACL composition work.

## Methodology Diagram

fMoW cells -> DINOv3 patch tokens -> visual vocabulary -> soft histograms ->
DynamicVis BoVW training -> downstream transfer (CBIR / UC Merced / LEVIR-CD).

## fMoW Dataset

SatBae is centered on fMoW with 63 land-use categories. The BoVW pipeline uses
cell-level training targets built from DINOv3 patch embeddings.

Expected local layout (after download):

data/fmow/train/<category>/<location>/*.jpg
data/fmow/val/<category>/<location>/*.jpg

Common manifests already in this repository:

- data/fmow_manifest_train.json
- data/fmow_manifest_val.json

Download examples:

```bash
# 10% stratified sample by class (fast smoke test)
python scripts/download_fmow.py --output-dir data/fmow --split train --fraction 0.1

# 50% stratified sample (recommended for iterative work)
python scripts/download_fmow.py --output-dir data/fmow --split train --fraction 0.5

# Full RGB download (large)
python scripts/download_fmow.py --output-dir data/fmow --split train --use-rgb

# Validation split
python scripts/download_fmow.py --output-dir data/fmow --split val --fraction 0.5
```

Notes:

- download_fmow.py defaults to msrgb unless --use-rgb is provided.
- Keep manifest and local data layout consistent across BoVW phases.

## Novel BoVW Pipeline

The implemented pipeline learns a DynamicVis backbone to predict BoVW histogram
targets produced from DINOv3 patch tokens.

Loss used in training:

L_total = lambda_emd * L_emd + lambda_cls * L_cls + lambda_mil * L_mil

Where:

- L_emd: Sinkhorn EMD between predicted and target histograms.
- L_cls: auxiliary label-smoothed classification loss.
- L_mil: CLIP-style bidirectional MIL contrastive loss.

### End-to-End Quick Start (BoVW)

Run the following in order.

```bash
# Phase 1: extract DINOv3 patch tokens from fMoW cells
sbatch run_gpu_extract_patch_tokens.sh \
    --manifest data/fmow_manifest_train.json \
    --num-gpus 8

# Phase 2: build visual vocabulary with FAISS k-means
sbatch run_gpu_build_vocabulary.sh \
    --patch-token-dir outputs/patch_tokens_bovw \
    --K 512 \
    --subsample 5000000

# Phase 3: generate soft histogram targets
sbatch run_gpu_generate_histograms.sh \
    --manifest data/fmow_manifest_train.json \
    --output-dir outputs/bovw_histograms

# Phase 3b: derive per-cell class labels from manifest paths
python scripts/extract_manifest_labels.py \
    --manifest data/fmow_manifest_train.json \
    --output-dir outputs/bovw_histograms

# Phase 4: train DynamicVis with BoVW objective
sbatch run_gpu_bovw_training.sh \
    --manifest data/fmow_manifest_train.json \
    --histogram-dir outputs/bovw_histograms \
    --vocab-dir outputs/bovw_vocabulary \
    --cell-labels outputs/bovw_histograms/cell_labels.npy \
    --epochs 100 \
    --batch-size 32 \
    --num-gpus 8
```

### Phase Outputs

| Phase | Script | Main Outputs |
|---|---|---|
| 1 | run_gpu_extract_patch_tokens.sh | outputs/patch_tokens_bovw/*.npz |
| 2 | run_gpu_build_vocabulary.sh | outputs/bovw_vocabulary/centroids.npy, ground_cost.npy |
| 3 | run_gpu_generate_histograms.sh | outputs/bovw_histograms/histograms.npy, cell_ids.npy |
| 3b | scripts/extract_manifest_labels.py | outputs/bovw_histograms/cell_labels.npy |
| 4 | run_gpu_bovw_training.sh | outputs/bovw_training*/final_model.pth, final_backbone.pth |

### Optional Ablation Runner

```bash
# Sequential ablations
bash run_bovw_ablation.sh --epochs 20

# Parallel job submission
bash run_bovw_ablation.sh --parallel --epochs 20
```

## Downstream Evaluation (eval/)

The repository includes three downstream evaluation tracks for transferred
DynamicVis representations.

### 1) CBIR Retrieval (AID / ForestNet)

Entry point:

- eval/cbir/main.py
- Wrapper: run_gpu_cbir_eval.sh

What it does:

- Extracts embeddings with the model adapter in eval/adapters/.
- Builds or loads a FAISS index.
- Reports Recall@K and mAP@K.

Example commands:

```bash
# AID retrieval (stratified k-fold)
sbatch run_gpu_cbir_eval.sh \
    --dataset aid \
    --data-dir data/eval/AID \
    --checkpoint outputs/bovw_training_8262/epoch_20.pth

# ForestNet retrieval
sbatch run_gpu_cbir_eval.sh \
    --dataset forestnet \
    --data-dir data/eval/deep/downloads/ForestNetDataset \
    --checkpoint outputs/bovw_training_8262/epoch_20.pth
```

### 2) UC Merced Scene Classification

Entry point:

- eval/ucmerced/main.py
- Wrapper: run_gpu_ucmerced_eval.sh

What it does:

- Freezes the foundation backbone.
- Extracts pooled features.
- Trains a lightweight linear/MLP head.
- Reports Top-1/Top-5, precision, recall, F1.

Example commands:

```bash
# Fixed split
sbatch run_gpu_ucmerced_eval.sh \
    --checkpoint outputs/bovw_training_8262/epoch_20.pth \
    --split-mode fixed

# Stratified k-fold
sbatch run_gpu_ucmerced_eval.sh \
    --checkpoint outputs/bovw_training_8262/epoch_20.pth \
    --split-mode kfold \
    --num-folds 5
```

### 3) LEVIR-CD Change Detection

Entry point:

- eval/change-detection/main.py
- Wrapper: run_gpu_change_detection_eval.sh

What it does:

- Uses DynamicVis as a Siamese feature backbone.
- Adds an FPN-style fusion neck and CD prediction head.
- Trains/evaluates on LEVIR-CD with precision/recall/F1/IoU metrics.

Example commands:

```bash
# Train + evaluate
sbatch run_gpu_change_detection_eval.sh \
    --backbone-checkpoint outputs/bovw_training_8262/epoch_20.pth \
    --epochs 5

# Evaluate only from saved CD checkpoint
sbatch run_gpu_change_detection_eval.sh \
    --eval-only \
    --backbone-checkpoint outputs/bovw_training_8262/epoch_20.pth \
    --cd-checkpoint outputs/change_detection/best_cd_model.pth
```

Typical change-detection artifacts:

- outputs/change_detection/best_cd_model.pth
- outputs/change_detection/training_curves.png
- outputs/change_detection/predictions.png

## In-Domain fMoW Evaluation Scripts

```bash
# Evaluate a DynamicVis checkpoint on fMoW
python evaluate_dynamicvis.py --checkpoint /path/to/checkpoint.pth

# Evaluate pretrain-style model with RoI/FPN flow
python evaluate_dynamicvis_pretrain.py --checkpoint /path/to/checkpoint.pth
```

## Other Training Modes In Repo

- Vanilla pretrain (bbox-supervised DynamicVis): run_gpu_dynamicvis_training.sh
- Composition/QSACL experiments: train_dynamicvis_composition.py and related configs

These remain available, but the BoVW pipeline above is the current main path.

## Infrastructure Notes

- SLURM wrappers auto-activate the dynamicvis conda environment.
- Wrappers also auto-check/clone architectures/DynamicVis when missing.
- PYTHONPATH is set to include both repository root and architectures/DynamicVis.

## More Detail

For deeper architecture and loss documentation, see CONTEXT.md.