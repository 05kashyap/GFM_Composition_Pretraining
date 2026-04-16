# Ablation: DynamicVis stage-3 tokens as slot decoder keys/values (no DINOv3 patch cache)
"""
DynamicVis Base — Composition-Aware Training on fMoW (DynamicVis Keys Ablation).

This is an ablation variant that uses DynamicVis stage-3 spatial tokens (B, 256, 768)
as keys/values for the QuerySlotDecoder instead of cached DINOv3 patch embeddings.
This makes the contrastive component fully end-to-end and removes the dependency
on the offline DINOv3 patch embedding cache.

Key differences from the baseline:
  - use_dynamicvis_keys=True: backbone outputs spatial feature map instead of pooled
  - patch_dim=768: DynamicVis stage-3 dim instead of DINOv3's 2048
  - No patch_embed_dir: .npz files not loaded from disk

Uses the same loss formulation:
    L = λ_cls          * aux classification CE      label-guided
      + λ_slot_contrast * supervised contrastive    slot contrastive (label-based)
      + λ_slot_var      * per-slot variance hinge   slot anti-collapse

Prerequisites:
    1. Run ``cluster_viz.py --save-cluster-data --use-pca-targets`` to produce
       ``outputs/cluster_data/{manifest.json, targets.npy (256-d), centroids.npy}``.
    2. NO need to run embed_patches.py — slot decoder keys come from the backbone.

Usage:
    python train_dynamicvis_composition.py \\
        configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition_dvkeys.py \\
        --use-dynamicvis-keys
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
# No patch_embed_dir needed — slot decoder uses DynamicVis stage-3 tokens

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

# ==================== Logging ====================
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)

# ==================== Work Directory ====================
work_dir = 'outputs/fmow_dynamicvis_b_composition_dvkeys'

# ==================== Training Knobs ====================
batch_size = 32       # per GPU; SLURM script sets total batch (default 256 = 32 * 8 GPUs)
num_workers = 4
persistent_workers = True
non_blocking = True
prefetch_factor = 2
pin_memory = True

img_size = 512
num_epochs = 100

# ==================== Visualisation (wandb) ====================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='satbae-fmow',
            group='dynamicvis-composition',
            name='dynamicvis_b_composition_dvkeys',
            tags=['dynamicvis', 'fmow', 'composition', 'contrastive', 'dvkeys-ablation'],
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
# ClsDataPreprocessor normalises images and stacks the batch.
# Mean/std match ImageNet (same as DynamicVis pretrain).
data_preprocessor = dict(
    type='ClsDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ==================== Model ====================
model = dict(
    type='CompositionAwareDynamicVis',
    data_preprocessor=data_preprocessor,
    num_classes=63,                          # activates aux_cls_head
    num_queries=16,                          # QuerySlotDecoder queries
    slot_dim=256,                            # QuerySlotDecoder output dim
    patch_dim=768,                           # DynamicVis stage-3 dim (overridden when use_dynamicvis_keys=True)
    use_dynamicvis_keys=True,                # Use backbone stage-3 tokens as slot decoder keys
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
        # NOTE: out_type will be overridden to 'featmap' by use_dynamicvis_keys=True
        out_type='avg_featmap',
    ),
    head=dict(
        type='CompositionHead',
        in_channels=768,                     # backbone last-stage dim (arch='b')
        proj_dim=256,                        # PCA target dimension (was 2048 for raw DINOv3)
        hidden_dim=512,                      # reduced for smaller output dim (was 1536)
        loss_type='mse',                     # MSE on standardised targets — stronger gradients
        standardise_targets=True,            # z-score targets → equal per-dim contribution
        tau=0.5,                             # temperature for InfoNCE (unused when lambda_contrast=0)
        lambda_comp=0,                       # MSE alignment disabled
        lambda_cosine=0,                     # cosine direction alignment disabled
        lambda_var=0,                        # variance regularization disabled
        lambda_cov=0,                        # covariance regularization disabled
        var_gamma=1.0,                       # target std for variance hinge
        lambda_contrast=0,                   # InfoNCE disabled
        lambda_smooth=0,                     # spatial smoothness disabled
        lambda_cls=0.5,                      # aux classification CE (label-guided)
        lambda_slot_contrast=0.5,            # per-slot supervised contrastive (requires labels)
        lambda_slot_var=0.25,                # per-slot variance hinge
        slot_var_gamma=1.0,                  # target std for slot variance
        slot_contrast_tau=0.1,               # temperature for slot contrastive
    ),
)

# ==================== Data Loaders ====================
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
        # No patch_embed_dir — slot decoder uses DynamicVis stage-3 tokens
        img_size=img_size,
        split='train',
        val_ratio=0.1,
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
    ),
]

# ==================== Auto Scale LR ====================
auto_scale_lr = dict(base_batch_size=64, enable=False)

# ==================== Runner ====================
runner_type = 'Runner'
