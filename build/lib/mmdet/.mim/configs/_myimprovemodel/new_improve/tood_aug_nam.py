_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='TOOD',
    backbone=dict(
        type='ResNet_ATTENTION',
        depth=50,
        num_stages=4,
        attention='NAMAttention',
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
        gsconv=False,
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
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
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
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco/'
#data_root='/home/pengxl/DataSet/tangka/update/coco/'
classes=("sword","Ruyi Bao","Precious mirror","Pipa","Bowl","Buddhist abbot's staff", "Scripture","pagoda","beads", "vajra Pestle",
           "vajra bell","Karma pestle","Yak tail dusting","Peacock feather fan and mirror","canopy","flag")
img_norm_cfg = dict(  # 图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [ # 训练流程
    dict(type='LoadImageFromFile',aug=True,percent=1,light_limt=85),   # 第 1 个流程，从文件路径里加载图像
    dict(type='LoadAnnotations', with_bbox=True),   # 第 2 个流程，对于当前图像，加载它的注释信息
    dict(type='Resize',  # 变化图像和其注释大小的数据增强的流程。
         img_scale=(1333, 800),  # 图像的最大规模
         keep_ratio=True), # 是否保持图像的长宽比
    dict(type='RandomFlip',#  翻转图像和其注释大小的数据增强的流程
         flip_ratio=0.5), # 翻转图像的概率
    dict(type='Normalize', **img_norm_cfg), # 归一化当前图像的数据增强的流程
    dict(type='Pad', size_divisor=32), # 填充当前图像到指定大小的数据增强的流程，填充图像可以被当前值整
    dict(type='DefaultFormatBundle'), # 流程里收集数据的默认格式捆
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), # 决定数据中哪些键应该传递给检测器的流程
]
test_pipeline = [
    dict(type='LoadImageFromFile',aug=True,percent=1,light_limt=85),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# mmdetection中batchsize没有显试定义，这里的batchsize通过公式batchsize=GPUs*samples_per_gpu得到。比如本人gpu数量为1，samples_per_gpu设为2，那么bachsize等于1*2=2。
data = dict(
    samples_per_gpu=4,  # 单个 GPU 的 Batch size
    workers_per_gpu=2,  # 单个 GPU 分配的数据加载线程数
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',   # 标注文件路径
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
# 每 1 轮迭代进行一次测试评估
evaluation = dict(interval=1, # 验证的间隔
                  metric='bbox',# 验证期间使用的指标
                  save_best='auto') #保存最优模型


# optimizer
optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0001)

# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')]
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[10, 16])
runner = dict(type='EpochBasedRunner', max_epochs=25)
checkpoint_config = dict(interval=5)