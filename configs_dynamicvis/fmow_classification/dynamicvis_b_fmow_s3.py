"""
DynamicVis Base configuration for fMoW classification with S3 streaming.
Uses AWS S3 to stream images directly without downloading the full dataset.
Includes wandb logging for experiment tracking.
"""

# Import custom modules
custom_imports = dict(
    imports=[
        'dynamicvis',
        'datasets.fmow_s3_mmpretrain',
    ],
    allow_failed_imports=False
)

default_scope = 'mmpretrain'

# ==================== Hooks Configuration ====================
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
        # Save best based on F1-score (same as pretrained model)
        save_best='single-label/f1-score',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# ==================== Environment Configuration ====================
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ==================== Logging Configuration ====================
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)

# ==================== Work Directory ====================
work_dir = 'outputs/fmow_dynamicvis_b_s3'

# ==================== AWS S3 Configuration ====================
s3_bucket = 'spacenet-dataset'
s3_prefix = 'Hosted-Datasets/fmow/fmow-rgb'
manifest_key = 'Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2'
local_manifest = 'data/manifest.json.bz2'

# ==================== Training Configuration ====================
batch_size = 32  # Per GPU batch size
num_workers = 4
persistent_workers = True
pin_memory = True

num_classes = 63
img_size = 224  # Use 224 for S3 streaming to reduce bandwidth

# ==================== Visualization Backends (including wandb) ====================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='satbae-fmow',
            group='dynamicvis',
            name='dynamicvis_b_s3_streaming',
            tags=['dynamicvis', 'fmow', 's3-streaming'],
        )
    ),
]

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends
)

# ==================== Training Schedule ====================
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# ==================== Data Preprocessor ====================
data_preprocessor = dict(
    type='ClsDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# ==================== Model Configuration ====================
model = dict(
    type='ImageClassifier',
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
        out_type='avg_featmap',
    ),
    neck=None,
    head=dict(
        type='DynamicVisClsHead',
        num_classes=num_classes,
        in_channels=768,  # Base model embed dim
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
)

# ==================== Data Pipelines ====================
train_pipeline = [
    dict(type='LoadImageFromS3', to_float32=True),
    dict(type='RandomResizedCrop', scale=img_size, crop_ratio_range=(0.8, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromS3', to_float32=True),
    dict(type='ResizeEdge', scale=int(img_size * 1.14), edge='short'),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='PackInputs'),
]

test_pipeline = val_pipeline

# ==================== Data Loaders ====================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='FMoWS3Dataset',
        bucket=s3_bucket,
        s3_prefix=s3_prefix,
        manifest_key=manifest_key,
        local_manifest=local_manifest,
        split='train',
        pipeline=train_pipeline,
        # Prefetching for efficient GPU utilization
        # Buffer should be 3-4x batch_size to keep GPU fed
        enable_prefetch=True,
        prefetch_size=1024,  # Prefetch ~4 batches ahead (for batch_size=256)
        num_prefetch_workers=16,  # More threads for parallel S3 fetches
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='FMoWS3Dataset',
        bucket=s3_bucket,
        s3_prefix=s3_prefix,
        manifest_key=manifest_key,
        local_manifest=local_manifest,
        split='val',
        pipeline=val_pipeline,
        enable_prefetch=True,
        prefetch_size=512,  # ~2 batches ahead for validation
        num_prefetch_workers=8,
    ),
)

test_dataloader = val_dataloader

# ==================== Evaluators ====================
# Use multiple metrics to match pretrained model evaluation
# This reports: accuracy (top1/top5), precision, recall, f1-score
val_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(
        type='SingleLabelMetric',
        items=('precision', 'recall', 'f1-score'),
        average='macro',
        num_classes=num_classes,
    ),
]
test_evaluator = val_evaluator

# ==================== Validation/Test Configuration ====================
val_cfg = dict()
test_cfg = dict()

# ==================== Optimizer ====================
base_lr = 1e-4

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

# ==================== Learning Rate Scheduler ====================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        by_epoch=True,
        begin=5,
        end=100,
    ),
]

# ==================== Runner ====================
runner_type = 'Runner'
