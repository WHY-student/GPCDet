checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='SetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='TOOD',
    backbone=dict(
        type='ResNet_ATTENTION',
        depth=50,
        num_stages=4,
        attention='NAMAttention',
        device=2,
        chann=True,
        spatial=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='PAFPN_ATTENTION',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        num_dcn=3,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/'
classes = ('sword', 'Ruyi Bao', 'Precious mirror', 'Pipa', 'Bowl',
           "Buddhist abbot's staff", 'Scripture', 'pagoda', 'beads',
           'vajra Pestle', 'vajra bell', 'Karma pestle', 'Yak tail dusting',
           'Peacock feather fan and mirror', 'canopy', 'flag')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=('sword', 'Ruyi Bao', 'Precious mirror', 'Pipa', 'Bowl',
                 "Buddhist abbot's staff", 'Scripture', 'pagoda', 'beads',
                 'vajra Pestle', 'vajra bell', 'Karma pestle',
                 'Yak tail dusting', 'Peacock feather fan and mirror',
                 'canopy', 'flag'),
        ann_file=
        '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/annotations/instances_train.json',
        img_prefix=
        '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('sword', 'Ruyi Bao', 'Precious mirror', 'Pipa', 'Bowl',
                 "Buddhist abbot's staff", 'Scripture', 'pagoda', 'beads',
                 'vajra Pestle', 'vajra bell', 'Karma pestle',
                 'Yak tail dusting', 'Peacock feather fan and mirror',
                 'canopy', 'flag'),
        ann_file=
        '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/annotations/instances_val.json',
        img_prefix=
        '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('sword', 'Ruyi Bao', 'Precious mirror', 'Pipa', 'Bowl',
                 "Buddhist abbot's staff", 'Scripture', 'pagoda', 'beads',
                 'vajra Pestle', 'vajra bell', 'Karma pestle',
                 'Yak tail dusting', 'Peacock feather fan and mirror',
                 'canopy', 'flag'),
        ann_file=
        '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/annotations/instances_val.json',
        img_prefix=
        '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox', save_best='auto')
optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[10, 16])
runner = dict(type='EpochBasedRunner', max_epochs=25)
work_dir = 'work_dirs_tood_panfpn/namchannel_cbamspatial/channel_spatial_bing_x1*sig(w)+x2*(1-sig(w)'
auto_resume = False
gpu_ids = [2]
