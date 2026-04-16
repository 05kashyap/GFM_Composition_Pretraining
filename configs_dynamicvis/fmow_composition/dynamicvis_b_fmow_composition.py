"""
DynamicVis Base — Composition-Aware Training on fMoW with Two-View QSACL.

Trains DynamicVisBackbone (arch='b') with a projection head and multi-view
contrastive learning (QSACL: Query-Slot Attention Contrastive Learning).

The key innovation is two-view contrastive learning where:
  - Each cell produces N augmented views (8 = 2 global + 6 local crops)
  - Positives: same cell across different views (view correspondence)
  - Negatives: different cells within the batch
  - No labels needed for contrastive loss (labels only used for aux_cls)

Loss components:
    L = λ_cls          * aux classification CE      label-guided
      + λ_slot_contrast * QSACL InfoNCE             two-view slot contrastive
      + λ_slot_var      * per-slot variance hinge   slot anti-collapse
      + λ_var           * proj variance hinge       backbone anti-collapse (VICReg)
      + λ_cov           * proj covariance penalty   backbone decorrelation (VICReg)

Note: λ_var and λ_cov provide gradients to the backbone through the projection head.
Without them, the backbone would receive no gradients since:
  - aux classification uses detached features (intentionally no backbone gradients)
  - slot losses use external DINOv3 embeddings (no backbone gradients)

Prerequisites:
    1. Run ``embed_patches.py`` to cache DINOv3 small-patch embeddings.
    2. Run ``cluster_viz.py --save-cluster-data`` to produce
       ``outputs/cluster_data/{manifest.json, cell_labels.npy}``.
       (targets.npy is optional — not needed when loss_comp=0)

Usage:
    python train_dynamicvis_composition.py \\
        configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py
"""

# ==================== Custom Imports ====================
custom_imports = dict(
    imports=[
        'dynamicvis',                       # backbone registration
        'models.composition_head',          # CompositionAwareDynamicVis, CompositionHead
        'models.query_slot_decoder',        # QuerySlotDecoder
        'losses.composition_loss',          # CompositionAwareLoss
        'datasets.fmow_composition_dataset',  # FMoWCompositionDataset
        'datasets.group_sampler',           # ImageGroupSampler
    ],
    allow_failed_imports=False,
)

default_scope = 'mmpretrain'

# ==================== Paths ====================
cluster_data_dir = 'outputs/cluster_data'
patch_embed_dir = 'outputs/preprocess_cache_dinov3'  # DINOv3 patch embedding cache

# ==================== Hooks ====================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        max_keep_ckpts=5,
        save_last=True,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# ==================== Environment ====================
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# DDP: register tokens may not receive gradients when target branch is stop-gradiented
find_unused_parameters = True

# ==================== Logging ====================
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)

# ==================== Work Directory ====================
work_dir = 'outputs/fmow_dynamicvis_b_composition'

# ==================== Training Knobs ====================
batch_size = 64       # per GPU; increased from 40 since we reduced views from 8 to 4
num_workers = 4
persistent_workers = True
non_blocking = True
prefetch_factor = 2
pin_memory = True

img_size = 512
num_epochs = 100
num_views = 8         # 2 global + 6 local crops per cell (reduced from 8 for memory)

# ==================== Visualisation (wandb) ====================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='satbae-fmow',
            group='dynamicvis-composition',
            name='dynamicvis_b_composition',
            tags=['dynamicvis', 'fmow', 'composition', 'contrastive'],
        ),
    ),
]

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends,
)

# ==================== Schedule ====================
train_cfg = dict(by_epoch=True, max_epochs=num_epochs)

# ==================== Data Preprocessor ====================
# Custom preprocessor handles multi-view QSACL inputs:
#   - Receives list of dicts from pseudo_collate
#   - Stacks single-view tensors OR list of multi-view tensors per sample
#   - No normalization (already done in dataset transforms)
data_preprocessor = dict(
    type='MultiViewDataPreprocessor',
    non_blocking=True,
)

# ==================== Model ====================
model = dict(
    type='CompositionAwareDynamicVis',
    data_preprocessor=data_preprocessor,
    num_classes=63,                          # activates aux_cls_head
    num_queries=16,                          # QuerySlotDecoder queries
    slot_dim=256,                            # QuerySlotDecoder output dim
    patch_dim=2048,                          # DINOv3 patch embedding dim
    conditioned=True,                        # backbone-conditioned residual queries
    backbone_dim=768,                        # backbone global feature dimension
    num_registers=4,                         # register tokens in slot decoder
    ema_tau=0.996,                           # EMA decay for target slot decoder (BYOL-style)
    backbone=dict(
        type='DynamicVisBackbone',
        arch='b',
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='fixed', val=0.1),
        global_token_cfg=dict(pos='head', num=-1),
        is_softmax_on_x=True,
        img_size=img_size,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        spatial_token_keep_ratios=[8, 4, 2, 1],
        out_indices=(3,),
        out_type='avg_featmap',              # → (B, 768) global embedding
    ),
    head=dict(
        type='CompositionHead',
        in_channels=768,                     # backbone last-stage dim (arch='b')
        proj_dim=256,                        # PCA target dimension (was 2048 for raw DINOv3)
        hidden_dim=512,                      # reduced for smaller output dim (was 1536)
        loss_type='mse',                     # MSE on standardised targets — stronger gradients
        standardise_targets=True,            # z-score targets → equal per-dim contribution
        tau=0.5,                             # temperature for InfoNCE (unused when lambda_contrast=0)
        lambda_comp=0,                       # MSE alignment (disabled — no targets needed)
        lambda_cosine=0,                     # cosine direction alignment (disabled)
        lambda_var=5.0,                      # variance regularization (ENABLED for backbone gradients)
        lambda_cov=1.0,                      # covariance regularization (ENABLED for decorrelation)
        var_gamma=1.0,                       # target std for variance hinge
        lambda_contrast=0,                   # disabled — not needed with QSACL
        lambda_smooth=0,                     # spatial smoothness (disabled)
        lambda_cls=0.5,                      # aux classification CE (label-guided)
        lambda_slot_contrast=0.5,            # BYOL-style slot loss (asymmetric)
        lambda_slot_var=0.25,                # per-slot variance hinge
        lambda_slot_diversity=1.0,           # slot diversity (orthogonality) loss — increased from 0.1
        slot_var_gamma=1.0,                  # target std for slot variance
        slot_contrast_tau=0.1,               # temperature for slot InfoNCE
    ),
)

# ==================== Data Loaders ====================
# NOTE: Using pseudo_collate returns list of dicts. Multi-view collation is
# handled in train_dynamicvis_composition.py batch loop.
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    drop_last=True,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(
        type='ImageGroupSampler',
        cells_per_group=16,   # up to 16 cells per image per epoch
        shuffle=True,
    ),
    dataset=dict(
        type='FMoWCompositionDataset',
        cluster_data_dir=cluster_data_dir,
        patch_embed_dir=patch_embed_dir,
        img_size=img_size,
        split='train',
        val_ratio=0.1,
        num_views=num_views,                  # 2 global + 2 local crops per cell
    ),
)

# Validation is disabled — the composition model outputs embeddings, not class
# predictions, so classification-based evaluators (Accuracy) are not applicable.
# The real evaluation metrics are the loss components logged during training.
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

# ==================== Optimizer ====================
base_lr = 5e-4

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    dtype='bfloat16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2),   # relaxed from 1.0 — MSE has stronger gradients
)

# ==================== LR Scheduler ====================
param_scheduler = [
    # 5-epoch linear warmup: 0.001 * base_lr → base_lr
    dict(
        type='LinearLR',
        start_factor=0.001,
        end_factor=1.0,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    # Cosine decay to base_lr / 100
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        by_epoch=True,
        begin=5,
        end=num_epochs,
        convert_to_iter_based=True,
    ),
]

# ==================== Auto Scale LR ====================
auto_scale_lr = dict(base_batch_size=64, enable=False)

# ==================== Runner ====================
runner_type = 'Runner'
