# model settings

# 双阶段的有rpn_head （建议框）和roi_head
model = dict(
    type='CascadeRCNN',  # # 检测器(detector)名称
    # 方法定义在/mmdet/models/backbones/resnet.py
    backbone=dict( # 主干网络的配置文件
        type='ResNet',  # 主干网络的类别
        depth=50,    # 主干网络的深度，对于 ResNet 和 ResNext 通常设置为 50 或 101
        num_stages=4,  # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入
        out_indices=(0, 1, 2, 3),  # 每个状态产生的特征图输出的索引。
        frozen_stages=1,  # 第一个状态的权重被冻结
        norm_cfg=dict(  # 归一化层(norm layer)的配置项。
            type='BN', requires_grad=True),  # 归一化层的类别，通常是 BN 或 GN。
        norm_eval=True, # 是否冻结 BN 里的统计项。
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), # 加载通过 ImageNet 预训练的模型

    #这里使用FPN做为neck,并指定了FPN的输入、输出通道数、是否使用relu等参数
    # 方法定义在 /mmdet/models/necks/fpn.py
    neck=dict(
        type='FPN',  # 检测器的 neck 是 FPN，还有 'NASFPN', 'PAFPN' 等
        in_channels=[256, 512, 1024, 2048], # 输入通道数，这与主干网络的输出通道一致 # 这里in_channels对应的长度就是对应于backbone对应融合的长度，与通道数相同的融合
        out_channels=256, # 金字塔特征图每一层的输出通道
        num_outs=5), # 输出的特征层数
     # 方法定义在 /mmdet/models/dense_heads/rpn_head.py
    rpn_head=dict(
        type='RPNHead',  # RPN_head 的类型是 'RPNHead', 还有 'GARPNHead' 等
        in_channels=256,   # 每个输入特征图的输入通道，这与 neck 的输出通道一致。
        feat_channels=256, # head 卷积层的特征通道。
        anchor_generator=dict(  # Anchor生成器的配置。
            # 方法定义在/mmdet/core/anchor/anchor_generator.py
            type='AnchorGenerator',  # 大多是方法使用 AnchorGenerator 作为锚点生成器, SSD 检测器使用 `SSDAnchorGenerator`
            scales=[8],  # 锚框的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0], # anchor高度和宽度之间的比率。
            strides=[4, 8, 16, 32, 64]), # anchor生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        bbox_coder=dict( # 在训练和测试期间对框进行编码和解码。
            type='DeltaXYWHBBoxCoder',  # 框编码器的类别 定义在 /mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
            target_means=[.0, .0, .0, .0],  # 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]), # 用于编码和解码框的标准差

        #loss归属于bbox_head部分,这里指定了检测头的损失函数
        # 损失定义在 /mmdet/models/losses/
        loss_cls=dict(   # 分类分支的损失函数配置
            type='CrossEntropyLoss',
            use_sigmoid=True,   # RPN通常进行二分类，所以通常使用sigmoid函数。
            loss_weight=1.0), # 分类分支的损失权重。
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

    roi_head=dict( # RoIHead 封装了双阶段/级联检测器的第二步。
        type='CascadeRoIHead', # 定义在/mmdet/models/roi_heads/cascade_roi_head.py
        num_stages=3, # 级联次数
        stage_loss_weights=[1, 0.5, 0.25], # 每次级联损失的权重
        bbox_roi_extractor=dict(  # 用于 bbox 回归的 RoI 特征提取器。 提取第一阶段建议框里面的特征
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign',
                           output_size=7, # 特征图的输出大小
                           sampling_ratio=0), # 提取 RoI 特征时的采样率。0 表示自适应比率。
            out_channels=256, # 提取特征的输出通道。
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅，应该与主干的架构保持一致。
        bbox_head=[ # RoIHead 中 box head 的配置.
           # 第一个级联头
            dict(
                type='Shared2FCBBoxHead',   # bbox head 的类别  定义在 mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
                in_channels=256,  # bbox head 的输入通道。 与 roi_extractor 中的 out_channels 一致。
                fc_out_channels=1024,  # FC 层的输出特征通道。
                roi_feat_size=7,   # 候选区域(Region of Interest)特征的大小。
                num_classes=16, # 分类的类别数量。
                bbox_coder=dict( # 第二阶段使用的框编码器。
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],  # 用于编码和解码框的均值
                    target_stds=[0.1, 0.1, 0.2, 0.2]), # 编码和解码的标准差。因为框更准确，所以值更小，常规设置时 [0.1, 0.1, 0.2, 0.2]。
                reg_class_agnostic=True,  # 回归是否与类别有关
                loss_cls=dict(   # 分类分支的损失函数配置
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,  # 回归分支的损失函数配置。
                               loss_weight=1.0)),
            # 第二个级联头
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=16,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            # 第三个级联头
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=16,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    #  训练和测试的配置部分
    train_cfg=dict(   # rpn 和 rcnn 训练超参数的配置
        rpn=dict(
            assigner=dict( # 分配器的配置
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,  # IoU >= 0.7(阈值) 被视为正样本。
                neg_iou_thr=0.3, # IoU < 0.3(阈值) 被视为负样本。
                min_pos_iou=0.3, # 将框作为正样本的最小 IoU 阈值。
                match_low_quality=True, # 是否匹配低质量的框
                ignore_iof_thr=-1), # 忽略 bbox 的 IoF 阈值
            sampler=dict(  # 正/负采样器(sampler)的配置
                type='RandomSampler',
                num=256, # 样本数量
                pos_fraction=0.5, # 正样本占总样本的比例
                neg_pos_ub=-1, # 基于正样本数量的负样本上限
                add_gt_as_proposals=False), # 采样后是否添加 GT 作为 proposal
            allowed_border=0,
            pos_weight=-1, # 训练期间正样本的权重
            debug=False), # 是否设置调试(debug)模式
        rpn_proposal=dict(
            nms_pre=2000, # NMS 前的 box 数
            max_per_img=2000, #  NMS 后要保留的 box 数量
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0), # 允许的最小 box 尺寸
        rcnn=[
            dict(
                assigner=dict(  # 第二阶段分配器的配置 第一个级联
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,  # 样本数量
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(  #第二个级联
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(  #第三个级联
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
