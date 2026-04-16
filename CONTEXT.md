# SatBae — Codebase Context

> A foundation model for satellite imagery built on the DynamicVis backbone with two primary training methodologies: **vanilla pretraining** (region-based classification) and **BoVW training** (Bag of Visual Words histogram prediction using Sinkhorn EMD).

---

## 1. Project Structure

```
SatBae/
├── architectures/DynamicVis/          # Upstream DynamicVis fork
│   └── dynamicvis/models/models.py    # Backbone, MambaBlock, SpatialSparseMixer, PretrainClsHead
├── configs_dynamicvis/
│   ├── fmow_pretrain/                 # Vanilla pretrain configs
│   └── fmow_composition/              # Experimental composition configs
├── models/
│   ├── bovw_head.py                   # BoVWDynamicVis + BoVWHead (primary training method)
│   ├── composition_head.py            # CompositionHead (experimental QSACL)
│   ├── query_slot_decoder.py          # QuerySlotDecoder (experimental)
│   └── dynamicvis_classifier.py       # Classification wrapper
├── losses/
│   ├── bovw_loss.py                   # BoVWLoss: Sinkhorn EMD + MIL contrastive + CLS
│   └── composition_loss.py            # CompositionAwareLoss (experimental)
├── datasets/
│   ├── fmow_s3_pretrain.py            # FMoWS3PretrainDataset (vanilla pretrain)
│   ├── fmow_bovw_dataset.py           # FMoWBoVWDataset (BoVW training)
│   └── fmow_composition_dataset.py    # FMoWCompositionDataset (experimental)
├── scripts/
│   ├── extract_patch_tokens.py        # BoVW Phase 1: DINOv3 token extraction
│   ├── build_vocabulary.py            # BoVW Phase 2: K-means vocabulary
│   ├── generate_histograms.py         # BoVW Phase 3: Soft histogram targets
│   ├── extract_manifest_labels.py     # BoVW Phase 3b: Cell label extraction
│   └── ...
├── utils/
│   ├── training_utils.py              # Device setup, checkpointing, meters
│   └── ema.py                         # EMAModel (for experimental BYOL training)
├── train_dynamicvis_pretrain.py       # Vanilla pretrain entry point
├── train_dynamicvis_bovw.py           # BoVW training entry point (primary)
├── train_dynamicvis_composition.py    # Experimental composition training
├── run_gpu_dynamicvis_training.sh     # SLURM: vanilla pretrain
├── run_gpu_bovw_training.sh           # SLURM: BoVW Phase 4 training
├── run_bovw_ablation.sh               # Ablation study script for BoVW loss components
├── run_gpu_extract_patch_tokens.sh    # SLURM: BoVW Phase 1
├── run_gpu_build_vocabulary.sh        # SLURM: BoVW Phase 2
└── run_gpu_generate_histograms.sh     # SLURM: BoVW Phase 3
```

---

## 2. DynamicVis Backbone

**File:** `architectures/DynamicVis/dynamicvis/models/models.py`
**Class:** `DynamicVisBackbone`

A 4-stage hierarchical vision backbone using **Mamba SSMs** (State Space Models) with **sparse token routing**. This backbone forms the foundation for both training methodologies.

### 2.1 Architecture — `arch='b'` (Base)

| Stage | Embed Dim | Mamba Layers | Patch Size | Stride | Resolution (512 input) | Tokens | Keep Ratio |
|-------|-----------|-------------|------------|--------|------------------------|--------|------------|
| 0     | 96        | 2           | 7          | 4      | 128×128                | 16384  | 1/8 (2048) |
| 1     | 192       | 4           | 3          | 2      | 64×64                  | 4096   | 1/4 (1024) |
| 2     | 384       | 16          | 3          | 2      | 32×32                  | 1024   | 1/2 (512)  |
| 3     | 768       | 4           | 3          | 2      | 16×16                  | 256    | 1/1 (all)  |

**Total trainable parameters:** 36,745,536 (100%)

### 2.2 Key Components

**SpatialSparseMixer** — the core block at each stage:

1. **Token importance scoring:** `nn.Linear(embed_dims, 1)` → softmax → per-token weights
2. **Gumbel noise** (training only): Gumbel(0, 0.1) noise on router logits for stochastic exploration
3. **Top-k selection:** Only `N / keep_ratio` tokens enter the Mamba layer
4. **Global tokens:** `adaptive_avg_pool1d` compresses all tokens into `H` summary tokens
5. **Bidirectional Mamba** (`path_type='forward_reverse_mean'`): Runs both forward and reversed token sequences through Mamba, averages results
6. **Weighted scatter-back:** Processed tokens are weighted by routing weights, added back to full residual stream

**MambaBlock** wraps HuggingFace `MambaMixer`:
- `state_size=16`, `intermediate_size=2*embed_dims`, `conv_kernel=4`
- `time_step_rank=ceil(embed_dims/16)`, `hidden_act="silu"`

### 2.3 Output Modes

| `out_type` | `out_indices` | Output | Used By |
|-----------|---------------|--------|---------|
| `'featmap'` | `(0,1,2,3)` | 4 spatial feature maps: `(B,96,128,128)`, `(B,192,64,64)`, `(B,384,32,32)`, `(B,768,16,16)` | Vanilla Pretrain (FPN + RoI) |
| `'avg_featmap'` | `(3,)` | Single `(B, 768)` global vector (avg pool + LayerNorm) | BoVW Training |

---

## 3. Vanilla DynamicVis Pretraining

### 3.1 Overview

A **region-based classification** pretrain on fMoW's 63 land-use categories. The backbone produces multi-scale feature maps → FPN → RoI features per bounding box → classification head with label-smoothed cross-entropy + MIL (Multi-Instance Learning) contrastive loss.

This method uses dense supervision from fMoW's bounding box annotations, where each image contains multiple labeled regions.

### 3.2 Config

**File:** `configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py`

```python
default_scope = 'mmdet'
num_classes   = 63
img_size      = 512
```

### 3.3 Model Architecture

```
DynamicVisPretrainClassifier (inherits ImageClassifier)
  ├── backbone: DynamicVisBackbone(arch='b', out_type='featmap', out_indices=(0,1,2,3))
  │     → 4 feature maps: [96@128², 192@64², 384@32², 768@16²]
  ├── pre_neck: FPN(in=[96,192,384,768], out=256, num_outs=5)
  │     → 5 feature levels at 256 channels
  ├── neck: GenericRoIExtractor
  │     RoIAlign(7×7) + Conv(5×5) pre-processing + GeneralizedAttention post
  │     → (N_rois, 256, 7, 7)
  └── head: DynamicVisPretrainClsHead(in=256, num_classes=63, with_mil=True)
        fc(256 → 63) + category_embedding(63, 256)
```

### 3.4 Loss Functions

**Total loss = L_cls + 0.25 × L_mil**

#### 3.4.1 Classification Loss — LabelSmoothLoss

```python
smoothed_target = one_hot × 0.9 + 0.1/63
L_cls = -sum(smoothed_target × log_softmax(logits))
```

Applied per RoI box with `avg_factor = N_rois`. Label smoothing (0.1) prevents overconfident predictions and improves generalization.

#### 3.4.2 MIL Contrastive Loss — CLIP-style Bidirectional

The MIL loss aligns RoI feature vectors with learned category embeddings using bidirectional contrastive learning:

```python
# L2 normalize features and class embeddings
object_feats = L2_norm(avg_pool(roi_feats))           # (N_rois, 256)
class_embeds = L2_norm(category_embedding.weight)      # (63, 256)
logit_scale  = exp(learnable_param)                    # init ~14.3

# Compute scaled cosine similarities
logits_f2c = logit_scale × object_feats @ class_embeds.T   # (N_rois, 63)
logits_c2f = logits_f2c.T                                   # (63, N_rois)

# Bidirectional MIL cross-entropy
mil_loss = (MILCrossEntropy(logits_f2c, labels) + MILCrossEntropy(logits_c2f, labels.T)) / 2
L_mil = 0.5 × mil_loss
```

**MILCrossEntropy**: Sums softmax probabilities over all positive class positions, then takes negative log:
```python
probs = F.softmax(pred_logits, dim=-1)
loss = -torch.log(torch.sum(target * probs, dim=-1) + 1e-8)
```

This allows multiple positive classes per sample and handles the case where different RoIs in an image may have different labels.

**Key components:**
- `category_embedding`: Learnable 63×256 matrix mapping classes to embedding space
- `logit_scale`: Learnable temperature parameter, initialized to `log(1/0.07) ≈ 2.66`, clamped to max 100

### 3.5 Data Pipeline

```
FMoWS3PretrainDataset (63 classes, S3 or local)
  → LoadImageFromS3WithBbox (decode, limit max_edge=1024, parse bboxes from JSON)
  → RandomFlip(H, 0.5) → RandomFlip(V, 0.5)
  → RandomResize(ratio_range=[0.1, 2.0]) → RandomCrop(512×512) → Pad(512×512)
  → FilterAnnotations(min_bbox=8×8)
  → PackDetInputs → DetDataSample with gt_instances (bboxes + labels)
  → DetDataPreprocessor (ImageNet mean/std, pad to divisor=32)
```

**Label source:** Bounding box annotations from fMoW JSON sidecar files (e.g., `airport_123_0_rgb_msrgb.json`). Each JSON contains bounding box coordinates and category labels for all objects in the image.

### 3.6 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW(lr=4e-4, betas=(0.9, 0.999), weight_decay=0.05) |
| AMP | bfloat16, dynamic loss scaling |
| Grad clip | max_norm=5.0 |
| Warmup | LinearLR: 0.001× → 1× over 5 epochs |
| Decay | CosineAnnealing: lr → lr×0.01 over remaining epochs |
| Auto-scale LR | base_batch_size=1184, enabled |
| Val interval | Every 10 epochs |
| Val metric | SingleLabelMetric (precision, recall, F1 over 63 classes) |

### 3.7 Training Results (Reference)

| Metric | Value |
|--------|-------|
| Training loss (epoch 10) | ~4.73 (cls 2.99 + mil 1.74) |
| Val F1 | 19.13% |
| Val Precision | 23.11% |
| Val Recall | 20.00% |
| Memory | ~16,581 MiB per GPU |
| Time per iter | ~1.46 sec |

### 3.8 SLURM Script

**File:** `run_gpu_dynamicvis_training.sh`

```bash
# Default training
sbatch run_gpu_dynamicvis_training.sh

# Override options
sbatch run_gpu_dynamicvis_training.sh --epochs 100 --batch-size 32
```

---

## 4. BoVW Training Pipeline (Primary Method)

### 4.1 Overview

A **Bag of Visual Words (BoVW)** approach that trains the DynamicVis backbone to predict soft histogram distributions over a learned visual vocabulary. The vocabulary is constructed via K-means clustering of DINOv3 patch tokens extracted from satellite imagery.

**Key advantages over vanilla pretraining:**
- No bounding box annotations required — uses weak image-level supervision
- Learns compositional visual patterns via histogram prediction
- Combines EMD loss (structural similarity) with MIL contrastive and classification losses

### 4.2 Pipeline Overview

```
Phase 1: Extract Patch Tokens    → outputs/patch_tokens_bovw/*.npz
    ↓
Phase 2: Build Visual Vocabulary → outputs/bovw_vocabulary/centroids.npy
    ↓                            → outputs/bovw_vocabulary/ground_cost.npy
Phase 3: Generate Histograms     → outputs/bovw_histograms/histograms.npy
    ↓
Phase 3b: Extract Cell Labels    → outputs/bovw_histograms/cell_labels.npy
    ↓
Phase 4: Train BoVW Model        → outputs/bovw_training/final_model.pth
```

---

### 4.3 Phase 1 — Raw Patch Token Extraction

**Script:** `scripts/extract_patch_tokens.py`

Extracts raw patch tokens from each 512×512 fMoW cell using DINOv3 ViT-L/16.

**Processing:**
1. Feed 512×512 cell through DINOv3 ViT-L/16 (pretrained on 493M satellite images)
2. Extract 32×32 = 1024 spatial tokens (each corresponds to a 16×16 pixel patch)
3. Average pool every 2×2 region → 16×16 = 256 tokens representing 32×32 pixel regions
4. Output per cell: `(256, 1024)` float32 tensor

**DINOv3 ViT-L/16 Architecture:**
- 24 transformer layers, dim=1024, 16 heads, MLP ratio=4
- Patch size 16, RoPE 2D positional encoding
- 1 CLS token + 4 register/storage tokens
- Pretrained on 493M satellite images (`dinov3_vitl16_pretrain_sat493m`)

**Output format:** `.npz` files with keys:
- `patch_tokens`: `(256, 1024)` float32
- `img_path`: original image path

**SLURM Script:** `run_gpu_extract_patch_tokens.sh`

```bash
# Default: 8 MIG GPUs, ~363k cells
sbatch run_gpu_extract_patch_tokens.sh

# Override options
sbatch run_gpu_extract_patch_tokens.sh --num-gpus 4
sbatch run_gpu_extract_patch_tokens.sh --manifest data/fmow_manifest_val.json
sbatch run_gpu_extract_patch_tokens.sh --no-resume  # force re-extraction
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `data/fmow` | Path to fMoW dataset |
| `--manifest` | `data/fmow_manifest_train.json` | Path to manifest.json |
| `--output-dir` | `outputs/patch_tokens_bovw` | Output directory |
| `--weights-path` | `weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | DINOv3 weights |
| `--batch-size` | 32 | Cells per GPU batch |
| `--num-gpus` | 8 | Number of MIG slices |
| `--no-resume` | flag | Force re-extraction |

**Expected output:**
- ~363k `.npz` files for full fMoW train set
- Each file: ~1MB (256×1024×4 bytes)
- Total size: ~350-400 GB

---

### 4.4 Phase 2 — Visual Vocabulary Construction

**Script:** `scripts/build_vocabulary.py`

Builds K-means visual vocabulary from extracted patch tokens using FAISS.

**Processing:**
1. Load random subsample of patch tokens from Phase 1 (default 5M tokens)
2. Sample uniformly across cells for representativeness
3. L2-normalize all tokens before clustering
4. Run FAISS K-means with K=512 clusters
5. Save centroids and precompute ground cost matrix for EMD loss

**Output files:**
- `centroids.npy`: `(K, 1024)` cluster centroids (L2-normalized)
- `ground_cost.npy`: `(K, K)` pairwise cosine distances for EMD
- `cluster_sizes.npy`: `(K,)` number of tokens per cluster
- `cluster_sizes.png`: histogram visualization

**SLURM Script:** `run_gpu_build_vocabulary.sh`

```bash
# Default: K=512, 5M token subsample
sbatch run_gpu_build_vocabulary.sh

# Larger vocabulary
sbatch run_gpu_build_vocabulary.sh --K 1024

# More tokens for clustering
sbatch run_gpu_build_vocabulary.sh --subsample 10000000
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch-token-dir` | `outputs/patch_tokens_bovw` | Phase 1 output |
| `--output-dir` | `outputs/bovw_vocabulary` | Output directory |
| `--K` | 512 | Vocabulary size |
| `--subsample` | 5000000 | Tokens to sample |
| `--seed` | 42 | Random seed |
| `--niter` | 100 | K-means iterations |
| `--nredo` | 3 | K-means restarts |

**Verification:**
```bash
python -c "
import numpy as np
c = np.load('outputs/bovw_vocabulary/centroids.npy')
g = np.load('outputs/bovw_vocabulary/ground_cost.npy')
print('Centroids:', c.shape)  # (512, 1024)
print('Ground cost:', g.shape, 'symmetric:', np.allclose(g, g.T))
"
```

---

### 4.5 Phase 3 — Histogram Target Generation

**Script:** `scripts/generate_histograms.py`

Converts each cell's patch tokens into soft-assignment histograms using the vocabulary from Phase 2.

**Processing:**
1. Load centroids from Phase 2 and L2-normalize them
2. For each cell, load `(256, 1024)` patch tokens from Phase 1
3. L2-normalize tokens
4. Compute soft assignment using RBF kernel:
   ```python
   def soft_assign(tokens, centroids, beta=10.0):
       sim = tokens @ centroids.T              # (N, K) cosine similarities
       dist_sq = 1.0 - sim                     # (N, K) cosine distances
       weights = np.exp(-beta * dist_sq)       # RBF kernel
       weights = weights / weights.sum(axis=1, keepdims=True)
       histogram = weights.sum(axis=0)         # (K,)
       histogram = histogram / histogram.sum() # L1 normalize
       return histogram.astype(np.float32)
   ```
5. Save all histograms as single numpy array for fast random access

**Output files:**
- `histograms.npy`: `(N_cells, K)` float32 histogram distributions
- `cell_ids.npy`: `(N_cells,)` int64 mapping to manifest entries

**SLURM Script:** `run_gpu_generate_histograms.sh`

```bash
# Default: beta=10.0, 8 workers
sbatch run_gpu_generate_histograms.sh

# Override RBF temperature
sbatch run_gpu_generate_histograms.sh --beta 15.0
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch-token-dir` | `outputs/patch_tokens_bovw` | Phase 1 output |
| `--vocab-dir` | `outputs/bovw_vocabulary` | Phase 2 output |
| `--manifest` | `data/fmow_manifest_train.json` | Manifest path |
| `--output-dir` | `outputs/bovw_histograms` | Output directory |
| `--beta` | 10.0 | RBF temperature (higher = sharper) |
| `--workers` | 8 | Parallel workers |

---

### 4.6 Phase 3b — Cell Label Extraction

**Script:** `scripts/extract_manifest_labels.py`

Extracts fMoW class labels from image paths for classification losses. Unlike vanilla pretraining which uses bounding box annotations, BoVW training derives labels from the directory structure.

**Label assignment mechanism:**
1. fMoW images follow this path pattern: `train/<category>/<category>_<id>/<category>_<id>_<seq>_rgb.jpg`
2. The script extracts the category name using regex: `r'(?:train|val)/([^/]+)/'`
3. Maps category string to integer label (0-62) using the 63 fMoW classes
4. **All 512×512 cells from the same image inherit the same category label**

This is **weak supervision** — the entire image is labeled "airport" even though individual cells might show runways, terminals, parking lots, or grass areas.

```bash
# Generate cell labels
python scripts/extract_manifest_labels.py \
    --manifest data/fmow_manifest_train.json \
    --output-dir outputs/bovw_histograms
```

**Output:** `cell_labels.npy` — `(N_cells,)` int64 array of class labels (0-62, or -1 for unlabeled)

**fMoW Categories (63 classes):**
```
zoo, wind_farm, water_treatment_facility, waste_disposal, tunnel_opening,
tower, toll_booth, swimming_pool, surface_mine, storage_tank, stadium,
space_facility, solar_farm, smokestack, single-unit_residential,
shopping_mall, shipyard, runway, road_bridge, recreational_facility,
railway_bridge, race_track, prison, port, police_station,
place_of_worship, parking_lot_or_garage, park, oil_or_gas_facility,
office_building, nuclear_powerplant, multi-unit_residential,
military_facility, lighthouse, lake_or_pond, interchange,
impoverished_settlement, hospital, helipad, ground_transportation_station,
golf_course, gas_station, fountain, flooded_road, fire_station,
factory_or_powerplant, electric_substation, educational_institution,
debris_or_rubble, dam, crop_field, construction_site, car_dealership,
burial_site, border_checkpoint, barn, archaeological_site, aquaculture,
amusement_park, airport_terminal, airport_hangar, airport, false_detection
```

---

### 4.7 Phase 4 — BoVW Training

**Script:** `train_dynamicvis_bovw.py`

Trains the DynamicVis backbone to predict histogram distributions using a combination of losses.

#### 4.7.1 Model Architecture

```
BoVWDynamicVis
  ├── backbone: DynamicVisBackbone(arch='b', out_type='avg_featmap')
  │     → global_feat (B, 768)
  ├── head: BoVWHead(in=768, hidden=512, vocab_size=512)
  │     prediction_head:
  │       Linear(768, 512) → LayerNorm → GELU
  │       Linear(512, 512) → Softmax
  │     → pred_hist (B, 512) normalized probability distribution
  ├── aux_cls_head: Linear(768, 63)  [receives global_feat.detach()]
  └── loss_fn: BoVWLoss(sinkhorn_eps=0.05, sinkhorn_iters=50)
```

#### 4.7.2 Loss Function — BoVWLoss

**File:** `losses/bovw_loss.py`

```
L_total = λ_emd × L_emd + λ_cls × L_cls + λ_mil × L_mil
```

| Term | Lambda | Formula | Purpose |
|------|--------|---------|---------|
| **L_emd** (Sinkhorn EMD) | 1.0 | Sinkhorn-Knopp optimal transport | Histogram distribution alignment |
| **L_cls** (classification) | 0.5 | Label-smoothed CE on aux head | Direct discriminative signal |
| **L_mil** (MIL contrastive) | 0.25 | CLIP-style bidirectional alignment | Feature-class contrastive learning |

##### Sinkhorn EMD Loss

The Sinkhorn-Knopp algorithm computes a differentiable approximation to Earth Mover's Distance (optimal transport):

```python
def sinkhorn(pred, target, ground_cost, eps=0.05, iters=50):
    # Precompute log kernel: log K = -C / eps
    log_K = -ground_cost / eps

    # Initialize scaling vectors (in log domain for numerical stability)
    log_u, log_v = zeros_like(pred), zeros_like(pred)

    for _ in range(iters):
        # Update u: u = target / (K @ v)
        log_u = log(target) - logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        # Update v: v = pred / (K.T @ u)
        log_v = log(pred) - logsumexp(log_K.T + log_u.unsqueeze(1), dim=2)

    # Compute transport plan T = diag(u) @ K @ diag(v)
    log_T = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
    T = exp(log_T)

    # EMD = <T, C> (Frobenius inner product)
    return (T * ground_cost).sum(dim=[1, 2]).mean()
```

The ground cost matrix `C` is the precomputed pairwise cosine distance between K vocabulary centroids from Phase 2. Unlike naive histogram comparison (L1/L2), EMD respects the geometric structure of the vocabulary — moving mass between similar visual words costs less than between dissimilar ones.

##### MIL Contrastive Loss

Identical to vanilla pretraining's MIL loss, but operating on backbone global features instead of RoI features:

```python
# L2 normalize features and class embeddings
feats = F.normalize(backbone_feats, p=2, dim=-1)          # (B, 768)
class_emb = F.normalize(category_embedding.weight, p=2, dim=-1)  # (63, 768)

# Compute scaled cosine similarity with learnable temperature
logit_scale = logit_scale.exp().clamp(max=100.0)
logits_f2c = logit_scale * feats @ class_emb.t()  # (B, 63)
logits_c2f = logits_f2c.t()                        # (63, B)

# Bidirectional MIL cross-entropy
L_mil = (MILCrossEntropy(logits_f2c, labels) + MILCrossEntropy(logits_c2f, labels.t())) / 2
```

**Key components:**
- `category_embedding`: Learnable 63×768 embedding matrix
- `logit_scale`: Learnable temperature, initialized to `log(1/0.07) ≈ 2.66`

##### Classification Loss

Label-smoothed cross-entropy on the auxiliary classification head (same as vanilla pretraining).

#### 4.7.3 Data Pipeline

```
FMoWBoVWDataset
  → Load 512×512 cell image from manifest
  → RandomFlip(H, 0.5) → RandomFlip(V, 0.5) → Resize(512)
  → ToTensor → Normalize(ImageNet mean/std)
  → Load histogram target from histograms.npy
  → Load cell label from cell_labels.npy
```

#### 4.7.4 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW(lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05) |
| AMP | bfloat16 |
| Grad clip | max_norm=5.0 |
| Warmup | 5 epochs linear (lr: 5e-7 → 5e-4) |
| Decay | Cosine: 5e-4 → 5e-6 over remaining epochs |

#### 4.7.5 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | `data/fmow_manifest_train.json` | Path to manifest |
| `--histogram-dir` | `outputs/bovw_histograms` | Phase 3 output |
| `--vocab-dir` | `outputs/bovw_vocabulary` | Phase 2 output |
| `--cell-labels` | `outputs/bovw_histograms/cell_labels.npy` | Cell labels |
| `--pretrained-backbone` | (see below) | Pretrained backbone weights |
| `--no-pretrained` | flag | Train from scratch |
| `--vocab-size` | 512 | Visual vocabulary size (K) |
| `--hidden-dim` | 512 | Hidden dimension in head |
| `--batch-size` | 32 | Per-GPU batch size |
| `--num-epochs` | 100 | Training epochs |
| `--lr` | 5e-4 | Base learning rate |
| `--lambda-emd` | 1.0 | EMD loss weight |
| `--lambda-cls` | 0.5 | Classification loss weight |
| `--lambda-mil` | 0.25 | MIL contrastive loss weight |
| `--sinkhorn-eps` | 0.05 | Sinkhorn regularization |
| `--sinkhorn-iters` | 50 | Sinkhorn iterations |
| `--wandb-project` | `satbae-bovw` | W&B project |
| `--no-wandb` | flag | Disable W&B |

**Default pretrained backbone:**
```
weights/pretrain_dynamicvis_b_bf16_mamba_best_single-label_f1-score_epoch_170.pth
```

#### 4.7.6 SLURM Script

**File:** `run_gpu_bovw_training.sh`

```bash
# Default training (100 epochs, 8 MIG GPUs)
sbatch run_gpu_bovw_training.sh

# Override options
sbatch run_gpu_bovw_training.sh --epochs 50 --batch-size 64
sbatch run_gpu_bovw_training.sh --lambda-emd 2.0 --lambda-cls 0.25 --lambda-mil 0.5

# Train from scratch (no pretrained backbone)
sbatch run_gpu_bovw_training.sh --no-pretrained

# Smoke test (2 epochs)
sbatch run_gpu_bovw_training.sh --debug
```

#### 4.7.7 W&B Logging

Training metrics logged to Weights & Biases:
- `train/loss`: Total loss
- `train/loss_emd`: EMD component
- `train/loss_cls`: Classification component
- `train/loss_mil`: MIL contrastive component
- `train/lr`: Learning rate
- `train/best_loss`: Best loss so far

#### 4.7.8 Expected Metrics

- `loss_emd`: Start ~0.3-0.6, should decrease
- `loss_cls`: Start ~4.0, should decrease
- `loss_mil`: Start ~4.0-5.0, should decrease
- `grad_norm`: Should stay below 20
- No NaN or Inf in any loss

#### 4.7.9 Checkpointing

Post-training saves:
1. **`final_model.pth`** — Full model excluding aux_cls_head
2. **`final_backbone.pth`** — Backbone only (for downstream tasks)

---

### 4.8 Ablation Study

**Script:** `run_bovw_ablation.sh`

Tests different loss component combinations to understand their individual and joint contributions.

**Configurations:**

| Name | λ_emd | λ_cls | λ_mil | Description |
|------|-------|-------|-------|-------------|
| `emd_only` | 1.0 | 0 | 0 | Pure histogram matching |
| `emd_mil` | 1.0 | 0 | 0.25 | + contrastive |
| `emd_mil_cls` | 1.0 | 0.5 | 0.25 | Full pipeline |
| `mil_cls` | 0 | 0.5 | 0.25 | No EMD |
| `mil_only` | 0 | 0 | 0.25 | Pure contrastive |
| `cls_only` | 0 | 0.5 | 0 | Pure classification |

```bash
# Run all ablations sequentially
./run_bovw_ablation.sh --epochs 10

# Run ablations as parallel SLURM jobs
./run_bovw_ablation.sh --parallel --epochs 10

# Dry run (show commands without executing)
./run_bovw_ablation.sh --dry-run

# Use subset of data for faster testing
./run_bovw_ablation.sh --data-fraction 0.1 --epochs 5
```

---

### 4.9 Full Pipeline Commands

```bash
# ========================================
# PHASE 1: Extract Patch Tokens
# ========================================
sbatch run_gpu_extract_patch_tokens.sh

# Check progress
find outputs/patch_tokens_bovw -name "*.npz" | wc -l  # should reach ~363k

# ========================================
# PHASE 2: Build Visual Vocabulary
# ========================================
sbatch run_gpu_build_vocabulary.sh

# Check output
ls -la outputs/bovw_vocabulary/

# ========================================
# PHASE 3: Generate Histograms
# ========================================
sbatch run_gpu_generate_histograms.sh

# Check output
ls -la outputs/bovw_histograms/

# ========================================
# PHASE 3b: Extract Cell Labels
# ========================================
python scripts/extract_manifest_labels.py \
    --manifest data/fmow_manifest_train.json \
    --output-dir outputs/bovw_histograms

# ========================================
# PHASE 4: Train BoVW Model
# ========================================
sbatch run_gpu_bovw_training.sh

# Check output
ls -la outputs/bovw_training/
```

### 4.10 Storage Requirements

| Phase | Output | Size (estimate) |
|-------|--------|-----------------|
| Phase 1 | `outputs/patch_tokens_bovw/*.npz` | ~350-400 GB |
| Phase 2 | `outputs/bovw_vocabulary/` | ~10 MB |
| Phase 3 | `outputs/bovw_histograms/` | ~700 MB |
| Phase 4 | `outputs/bovw_training/` | ~500 MB |

---

## 5. Other Experimental Methods

### 5.1 Composition-Aware Training (QSACL)

An experimental self-supervised distillation approach using DINOv3 features and Query-Slot Attention Contrastive Learning. This method encountered persistent slot collapse issues and was superseded by the BoVW approach.

**Key files:**
- `train_dynamicvis_composition.py`
- `models/composition_head.py`
- `models/query_slot_decoder.py`
- `losses/composition_loss.py`

**Approach:** Uses learnable query vectors with cross-attention over frozen DINOv3 patch embeddings. Includes BYOL-style EMA target network and multi-view contrastive learning (2 global + 6 local crops).

### 5.2 Direct Composition Targets

An earlier variant that attempted to directly regress DINOv3-derived composition targets using MSE loss with VICReg-style variance/covariance regularization. Suffered from target collapse due to high correlation in DINOv3 embeddings.

---

## 6. Infrastructure

### 6.1 Hardware (SLURM)

- 2× NVIDIA H100 NVL (95,830 MiB each), MIG enabled
- 4 MIG slices per GPU × 2 GPUs = 8 MIG slices (22,144 MiB each)
- Round-robin interleaving across GPUs for cross-GPU spread
- NCCL P2P disabled (MIG doesn't support P2P)
- Backend: **gloo** for >2 MIG slices, nccl for ≤2

### 6.2 Entry Point Comparison

| Feature | Vanilla Pretrain | BoVW Training |
|---------|------------------|---------------|
| Script | `train_dynamicvis_pretrain.py` | `train_dynamicvis_bovw.py` |
| Scope | mmdet | Custom |
| Data source | S3/local fMoW images + bbox JSON | Pre-computed histograms + cells |
| Supervision | Per-bbox labels | Per-cell weak labels |
| Output mode | `featmap` (4 scales) | `avg_featmap` (global) |
| Head | FPN + RoIAlign + fc | MLP histogram predictor |
| Losses | CLS + MIL | EMD + CLS + MIL |

### 6.3 Key Differences Summary

| Aspect | Vanilla Pretrain | BoVW Training |
|--------|------------------|---------------|
| Task | 63-class region classification | Histogram distribution prediction |
| Teacher | Ground-truth fMoW bbox labels | DINOv3-derived vocabulary |
| Backbone output | 4 feature maps (FPN input) | Single 768-d global vector |
| Head | FPN + RoIAlign + fc(256→63) | MLP(768→512→K) + Softmax |
| Primary loss | Label-smooth CE | Sinkhorn EMD |
| Data unit | Full image with bounding boxes | 512×512 grid cells |
| Label source | JSON sidecar bbox annotations | Directory path extraction |
| Validation | F1/precision/recall | Loss metrics only |
