_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN_CBAM_FUSE',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output', # 与Retinanet不同，这里是从p5生成
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
optimizer = dict(type='SGD',# 用于构建优化器的配置文件
                 lr=0.008, momentum=0.9, weight_decay=0.0001)# SGD 的衰减权重
optimizer_config = dict(grad_clip=None) # optimizer hook 的配置文件
# learning policy
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook
    policy='step',   # 调度流程(scheduler)的策略
    warmup='linear', # warmup策略
    warmup_iters=500,  # warmup的迭代次数
    warmup_ratio=0.001, # 用于warmup的起始学习率的比率
    step=[12,40,60]) # 衰减学习率的起止回合数
runner = dict(type='EpochBasedRunner', max_epochs=80)
data = dict(
    samples_per_gpu=4,  # 单个 GPU 的 Batch size
    workers_per_gpu=1) # 单个 GPU 分配的数据加载线程数