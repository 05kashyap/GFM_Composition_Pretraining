auto_scale_lr = dict(base_batch_size=1184, enable=True)
base_lr = 0.0004
batch_size = 16
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
        'datasets.fmow_s3_pretrain',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    non_blocking=True,
    pad_size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        rule='greater',
        save_best='single-label/f1-score',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='mmpretrain.VisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_size = 512
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='b',
        global_token_cfg=dict(num=-1, pos='head'),
        img_size=512,
        is_softmax_on_x=True,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        out_type='featmap',
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
        type='mmpretrain.DynamicVisBackbone'),
    head=dict(
        in_channels=256,
        loss=dict(
            label_smooth_val=0.1,
            mode='original',
            type='mmpretrain.LabelSmoothLoss'),
        num_classes=63,
        type='mmpretrain.DynamicVisPretrainClsHead',
        with_mil=True),
    neck=dict(
        aggregation='sum',
        featmap_strides=[
            4,
            8,
            16,
            32,
        ],
        out_channels=256,
        post_cfg=dict(
            attention_type='0100',
            in_channels=256,
            kv_stride=2,
            num_heads=6,
            spatial_range=-1,
            type='GeneralizedAttention'),
        pre_cfg=dict(
            in_channels=256,
            inplace=False,
            kernel_size=5,
            out_channels=256,
            padding=2,
            type='ConvModule'),
        roi_layer=dict(
            output_size=7,
            sampling_ratio=2,
            type='RoIAlign',
            use_torchvision=True),
        type='GenericRoIExtractor'),
    pre_neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    type='mmpretrain.DynamicVisPretrainClassifier')
non_blocking = True
num_classes = 63
num_workers = 4
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0, norm_type=2),
    dtype='bfloat16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0004, type='AdamW', weight_decay=0.05),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=1, factor=1.0, type='ConstantLR'),
]
persistent_workers = True
pin_memory = True
prefetch_factor = 2
randomness = dict(deterministic=False, seed=42)
resume = False
runner_type = 'Runner'
s3_bucket = 'spacenet-dataset'
s3_prefix = 'Hosted-Datasets/fmow/fmow-rgb'
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        bucket='spacenet-dataset',
        enable_prefetch=True,
        num_prefetch_workers=8,
        pipeline=[
            dict(
                max_edge=1024, to_float32=True,
                type='LoadImageFromS3WithBbox'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(
                keep_empty=True,
                min_gt_bbox_wh=(
                    8,
                    8,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        prefetch_size=512,
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='val',
        type='FMoWS3PretrainDataset',
        use_msrgb=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(num_classes=63, type='mmpretrain.SingleLabelMetric')
test_pipeline = [
    dict(max_edge=1024, to_float32=True, type='LoadImageFromS3WithBbox'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        pad_val=dict(img=(
            103.53,
            116.28,
            123.675,
        )),
        size=(
            512,
            512,
        ),
        type='Pad'),
    dict(keep_empty=True, min_gt_bbox_wh=(
        8,
        8,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=10)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        bucket='spacenet-dataset',
        enable_prefetch=True,
        num_prefetch_workers=16,
        pipeline=[
            dict(
                max_edge=1024, to_float32=True,
                type='LoadImageFromS3WithBbox'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='Resize',
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=False,
                crop_size=(
                    512,
                    512,
                ),
                crop_type='absolute',
                recompute_bbox=True,
                type='RandomCrop'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(
                keep_empty=True,
                min_gt_bbox_wh=(
                    8,
                    8,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        prefetch_size=1024,
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='train',
        type='FMoWS3PretrainDataset',
        use_msrgb=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(max_edge=1024, to_float32=True, type='LoadImageFromS3WithBbox'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='Resize',
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=False,
        crop_size=(
            512,
            512,
        ),
        crop_type='absolute',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(
        pad_val=dict(img=(
            103.53,
            116.28,
            123.675,
        )),
        size=(
            512,
            512,
        ),
        type='Pad'),
    dict(keep_empty=True, min_gt_bbox_wh=(
        8,
        8,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        bucket='spacenet-dataset',
        enable_prefetch=True,
        num_prefetch_workers=8,
        pipeline=[
            dict(
                max_edge=1024, to_float32=True,
                type='LoadImageFromS3WithBbox'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(
                keep_empty=True,
                min_gt_bbox_wh=(
                    8,
                    8,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        prefetch_size=512,
        s3_prefix='Hosted-Datasets/fmow/fmow-rgb',
        split='val',
        type='FMoWS3PretrainDataset',
        use_msrgb=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(num_classes=63, type='mmpretrain.SingleLabelMetric')
val_interval = 10
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            group='dynamicvis-pretrain',
            name='dynamicvis_b_s3_pretrain',
            project='satbae-fmow',
            tags=[
                'dynamicvis',
                'fmow',
                's3-streaming',
                'pretrain',
            ]),
        type='WandbVisBackend'),
]
visualizer = dict(
    type='mmpretrain.UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                group='dynamicvis-pretrain',
                name='dynamicvis_b_s3_pretrain',
                project='satbae-fmow',
                tags=[
                    'dynamicvis',
                    'fmow',
                    's3-streaming',
                    'pretrain',
                ]),
            type='WandbVisBackend'),
    ])
work_dir = 'outputs/fmow_dynamicvis_b_s3_pretrain_6265'
