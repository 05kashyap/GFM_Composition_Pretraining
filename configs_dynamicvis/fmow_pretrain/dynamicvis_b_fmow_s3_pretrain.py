"""
DynamicVis Base configuration for fMoW pretraining with S3 streaming.

This config matches the official DynamicVis pretrained model format:
- Uses bounding box annotations (detection-style pretraining)
- DynamicVisPretrainClassifier with FPN neck and RoI extraction
- Multi-instance learning (MIL) classification head
- Streams from AWS S3 without downloading the full 350GB dataset

Usage:
    python train_dynamicvis.py configs_dynamicvis/fmow_pretrain/dynamicvis_b_fmow_s3_pretrain.py
"""

# Import custom modules
custom_imports = dict(
    imports=[
        'dynamicvis',
        'datasets.fmow_s3_pretrain',
    ],
    allow_failed_imports=False
)

default_scope = 'mmdet'

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
        save_best='single-label/f1-score',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmpretrain.VisualizationHook', enable=False),
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
work_dir = 'outputs/fmow_dynamicvis_b_s3_pretrain'

# ==================== AWS S3 Configuration ====================
s3_bucket = 'spacenet-dataset'
s3_prefix = 'Hosted-Datasets/fmow/fmow-rgb'

# ==================== Training Configuration ====================
batch_size = 16  # Per GPU batch size - smaller due to larger images/FPN
num_workers = 4
persistent_workers = True
non_blocking = True
prefetch_factor = 2
pin_memory = True

num_classes = 63
img_size = 512  # Same as original DynamicVis config
val_interval = 10

# ==================== Visualization Backends (including wandb) ====================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='satbae-fmow',
            group='dynamicvis-pretrain',
            name='dynamicvis_b_s3_pretrain',
            tags=['dynamicvis', 'fmow', 's3-streaming', 'pretrain'],
        )
    ),
]

visualizer = dict(
    type='mmpretrain.UniversalVisualizer',
    vis_backends=vis_backends
)

# ==================== Training Schedule ====================
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=val_interval)

# ==================== Data Preprocessor ====================
# Using DetDataPreprocessor for detection-style training
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    non_blocking=non_blocking,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# ==================== Model Configuration ====================
# This matches the official DynamicVis pretrained model architecture
model = dict(
    type='mmpretrain.DynamicVisPretrainClassifier',
    backbone=dict(
        type='mmpretrain.DynamicVisBackbone',
        arch='b',
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='fixed', val=0.1),
        global_token_cfg=dict(pos='head', num=-1),
        is_softmax_on_x=True,
        img_size=img_size,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        spatial_token_keep_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        out_type='featmap',
    ),
    pre_neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],  # Base model channel dims
        out_channels=256,
        num_outs=5,
    ),
    neck=dict(
        type='GenericRoIExtractor',
        aggregation='sum',
        roi_layer=dict(
            type='RoIAlign',
            output_size=7,
            sampling_ratio=2,
            use_torchvision=True,
        ),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
        pre_cfg=dict(
            type='ConvModule',
            in_channels=256,
            out_channels=256,
            kernel_size=5,
            padding=2,
            inplace=False,
        ),
        post_cfg=dict(
            type='GeneralizedAttention',
            in_channels=256,
            spatial_range=-1,
            num_heads=6,
            attention_type='0100',
            kv_stride=2,
        ),
    ),
    head=dict(
        type='mmpretrain.DynamicVisPretrainClsHead',
        num_classes=num_classes,
        with_mil=True,
        in_channels=256,
        loss=dict(type='mmpretrain.LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
)

# ==================== Data Pipelines ====================
# Training pipeline with augmentation
train_pipeline = [
    dict(type='LoadImageFromS3WithBbox', to_float32=True, max_edge=1024),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # Large scale jittering
    dict(
        type='RandomResize',
        scale=(img_size, img_size),
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True,
    ),
    dict(
        type='RandomCrop',
        crop_size=(img_size, img_size),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=False,
    ),
    dict(type='Pad', size=(img_size, img_size), pad_val=dict(img=tuple(bgr_mean))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=True),
    dict(type='PackDetInputs'),
]

# Test/validation pipeline
test_pipeline = [
    dict(type='LoadImageFromS3WithBbox', to_float32=True, max_edge=1024),
    dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
    dict(type='Pad', size=(img_size, img_size), pad_val=dict(img=tuple(bgr_mean))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=True),
    dict(type='PackDetInputs'),
]

# ==================== Data Loaders ====================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='FMoWS3PretrainDataset',
        bucket=s3_bucket,
        s3_prefix=s3_prefix,
        split='train',
        pipeline=train_pipeline,
        use_msrgb=True,  # Use smaller msrgb images
        enable_prefetch=True,
        prefetch_size=1024,
        num_prefetch_workers=16,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='FMoWS3PretrainDataset',
        bucket=s3_bucket,
        s3_prefix=s3_prefix,
        split='val',
        pipeline=test_pipeline,
        use_msrgb=True,
        enable_prefetch=True,
        prefetch_size=512,
        num_prefetch_workers=8,
    ),
)

test_dataloader = val_dataloader

# ==================== Evaluators ====================
val_evaluator = dict(
    type='mmpretrain.SingleLabelMetric',
    num_classes=num_classes,
)
test_evaluator = val_evaluator

# ==================== Validation/Test Configuration ====================
val_cfg = dict()
test_cfg = dict()

# ==================== Optimizer ====================
base_lr = 0.0004  # Same as original DynamicVis config

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
        end=200,
    ),
]

# ==================== Auto Scale LR ====================
# Automatically scale LR based on batch size
# Base: batch_size=148 * 8 GPUs = 1184 total
# Scale factor = actual_batch_size / base_batch_size
auto_scale_lr = dict(base_batch_size=1184, enable=True)

# ==================== Runner ====================
runner_type = 'Runner'
