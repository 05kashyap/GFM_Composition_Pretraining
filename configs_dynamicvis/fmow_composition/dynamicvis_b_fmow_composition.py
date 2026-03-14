"""
DynamicVis Base — Composition-Aware Training on fMoW.

Trains DynamicVisBackbone (arch='b') with a projection head that maps
the backbone's 768-d global embedding into DINOv3's 2048-d space, then
optimises a three-part composition-aware loss:

    L = λ_comp     * (1 − cos(f_i, t_i))         alignment
      + λ_contrast * InfoNCE(f, t)                discrimination
      + λ_smooth   * L2(adjacent embeddings)      regularisation

Prerequisites:
    1. Run ``embed_patches.py`` to cache DINOv3 small-patch embeddings.
    2. Run ``cluster_viz.py --save-cluster-data`` to produce
       ``outputs/cluster_data/{manifest.json, targets.npy, centroids.npy}``.

Usage:
    python train_dynamicvis_composition.py \\
        configs_dynamicvis/fmow_composition/dynamicvis_b_fmow_composition.py
"""

# ==================== Custom Imports ====================
custom_imports = dict(
    imports=[
        'dynamicvis',                       # backbone registration
        'models.composition_head',          # CompositionAwareDynamicVis, CompositionHead
        'losses.composition_loss',          # CompositionAwareLoss
        'datasets.fmow_composition_dataset',  # FMoWCompositionDataset
        'datasets.group_sampler',           # ImageGroupSampler
    ],
    allow_failed_imports=False,
)

default_scope = 'mmpretrain'

# ==================== Paths ====================
cluster_data_dir = 'outputs/cluster_data'

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
work_dir = 'outputs/fmow_dynamicvis_b_composition'

# ==================== Training Knobs ====================
batch_size = 32       # per GPU (22 GiB MIG limit)
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
train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=10)

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
        proj_dim=2048,                       # match DINOv3 cls_avg
        hidden_dim=768,
        loss_type='mse',                     # MSE on standardised targets — stronger gradients
        standardise_targets=True,            # z-score targets → equal per-dim contribution
        tau=0.5,                             # temperature for InfoNCE (unused when lambda_contrast=0)
        lambda_comp=1.0,                     # MSE alignment (primary distillation signal)
        lambda_contrast=0.0,                 # disabled — targets too correlated for InfoNCE
        lambda_smooth=0.1,                   # spatial smoothness regulariser
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
        img_size=img_size,
        split='train',
        val_ratio=0.1,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='FMoWCompositionDataset',
        cluster_data_dir=cluster_data_dir,
        img_size=img_size,
        split='val',
        val_ratio=0.1,
    ),
)

test_dataloader = val_dataloader

# ==================== Evaluator (composition — log losses only) ====================
# The composition model outputs embeddings, not class predictions.
# We use a dummy Accuracy evaluator with compatibility shims in predict().
# The real evaluation metrics are the loss components logged during training.
val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator
val_cfg = dict()
test_cfg = dict()

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
