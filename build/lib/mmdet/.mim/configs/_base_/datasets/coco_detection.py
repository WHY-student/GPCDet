# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/duomeitinrfx/data/tangka_magic_instrument/update/coco_0.8_0.2/'
classes=("sword","Ruyi Bao","Precious mirror","Pipa","Bowl","Buddhist abbot's staff", "Scripture","pagoda","beads", "vajra Pestle",
           "vajra bell","Karma pestle","Yak tail dusting","Peacock feather fan and mirror","canopy","flag")
img_norm_cfg = dict(  # 图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [ # 训练流程
    dict(type='LoadImageFromFile'),   # 第 1 个流程，从文件路径里加载图像
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
    dict(type='LoadImageFromFile'),
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
