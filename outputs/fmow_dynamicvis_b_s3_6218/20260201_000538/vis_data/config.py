base_lr = 0.0001
batch_size = 32
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'dynamicvis',
        'datasets.fmow_s3_mmpretrain',
    ])
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True,
    type='ClsDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        rule='greater',
        save_best='accuracy/top1',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_size = 224
launcher = 'none'
load_from = None
local_manifest = 'data/manifest.json.bz2'
log_level = 'INFO'
manifest_key = 'Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2'
model = dict(
    backbone=dict(
        arch='b',
        global_token_cfg=dict(num=-1, pos='head'),
        img_size=224,
        is_softmax_on_x=True,
        out_indices=(3, ),
        out_type='avg_featmap',
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='fixed', val=0.1),
        spatial_token_keep_ratios=[
            8,
            4,
            2,
            1,
        ],
        strides=[
            4,
            2,
            2,
            2,
        ],
        type='DynamicVisBackbone'),
    head=dict(
        in_channels=768,
        loss=dict(
            label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
        num_classes=63,
        type='DynamicVisClsHead'),
    neck=None,
    type='ImageClassifier')
num_classes = 63
num_workers = 4
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0, norm_type=2),
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=5,
        by_epoch=True,
        end=100,
        eta_min=1.0000000000000002e-06,
        type='CosineAnnealingLR'),
]
persistent_workers = True
pin_memory = True
randomness = dict(deterministic=False, seed=42)
resume = False
runner_type = 'Runner'
s3_bucket = 'spacenet-dataset'
s3_prefix = 'Hosted-Datasets/fmow/fmow-rgb'
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        bucket='spacenet-dataset',
        enable_prefetch=True,
        local_manifest='data/manifest.json.bz2',
        manifest_key='Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2',
        num_prefetch_workers=4,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromS3'),
            dict(edge='short', scale=255, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        prefetch_size=64,
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='val',
        type='FMoWS3Dataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(to_float32=True, type='LoadImageFromS3'),
    dict(edge='short', scale=255, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        bucket='spacenet-dataset',
        enable_prefetch=True,
        local_manifest='data/manifest.json.bz2',
        manifest_key='Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2',
        num_prefetch_workers=8,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromS3'),
            dict(
                crop_ratio_range=(
                    0.8,
                    1.0,
                ),
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        prefetch_size=128,
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='train',
        type='FMoWS3Dataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadImageFromS3'),
    dict(crop_ratio_range=(
        0.8,
        1.0,
    ), scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        bucket='spacenet-dataset',
        enable_prefetch=True,
        local_manifest='data/manifest.json.bz2',
        manifest_key='Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2',
        num_prefetch_workers=4,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromS3'),
            dict(edge='short', scale=255, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        prefetch_size=64,
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='val',
        type='FMoWS3Dataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
val_pipeline = [
    dict(to_float32=True, type='LoadImageFromS3'),
    dict(edge='short', scale=255, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            group='dynamicvis',
            name='dynamicvis_b_s3_streaming',
            project='satbae-fmow',
            tags=[
                'dynamicvis',
                'fmow',
                's3-streaming',
            ]),
        type='WandbVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                group='dynamicvis',
                name='dynamicvis_b_s3_streaming',
                project='satbae-fmow',
                tags=[
                    'dynamicvis',
                    'fmow',
                    's3-streaming',
                ]),
            type='WandbVisBackend'),
    ])
work_dir = 'outputs/fmow_dynamicvis_b_s3_6218'
