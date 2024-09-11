# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',  # 使用Resnet作为backbone 其一共有4个stage
        depth=50,
        num_stages=4,#图片输入到 ResNet 中进行特征提取，输出 4 个特征图，按照特征图从大到小排列，分别是 C2 C3 C4 C5，stride = 4,8,16,32
        # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要，
        # 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        out_indices=(0, 1, 2, 3),
        # frozen_stages=-1，表示全部可学习
        # frozen_stage=0，表示stem权重固定
        # frozen_stages=1，表示 stem 和第一个 stage 权重固定
        # frozen_stages=2，表示 stem 和前两个 stage 权重固定
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True), # 所有的 BN 层的可学习参数都不需要梯度，也就不会进行参数更新
        norm_eval=True, # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        style='pytorch',  # 使用torch
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        # 4 个特征图输入到 FPN 模块中进行特征融合，输出 5 个通道数相同的特征图,分别是 p2 ~ p6，stride = 4,8,16,32,64
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # FPN 输出特征图个数
        num_outs=5), #FPN 模块虽然是接收 3 个特征图，但是输出 5 个特征图#
    rpn_head=dict(
        # FPN输出的5个特征图，输入到5个相同的 RPN中，每个分支都进行前后景分类和bbox回归，然后就可以和label计算loss
        type='RPNHead', # 这个类还继承了AnchorHead，所以这里面也需要填写anchor的配置
        in_channels=256,
        # 中间特征图通道数
        feat_channels=256,

        # 相比不包括 FPN 的 Faster R-CNN 算法，由于其 RPN Head 是多尺度特征图，
        # 为了适应这种变化，anchor 设置进行了适当修改，FPN 输出的多尺度信息可以帮助区分
        # 不同大小物体识别问题，每一层就不再需要不包括 FPN 的 Faster R-CNN 算法那么多 anchor 了
        anchor_generator=dict(
            type='AnchorGenerator',
            # 表示每个特征图的 base scales
            # 映射到原图的面积还要乘以对应的stride
            scales=[8],
            # 每个特征图有 3 个高宽比例
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须和特征图 stride 一致，不可以随意更改
            strides=[4, 8, 16, 32, 64]),
        # 在 anchor-based 算法中，为了利用 anchor 信息进行更快更好的收敛，
        # 一般会对 head 输出的 bbox 分支 4 个值进行编解码操作，作用有两个：
        # 更好的平衡分类和回归分支 loss，以及平衡 bbox 四个预测值的 loss
        # 训练过程中引入 anchor 信息，加快收敛
        # 对坐标进行回归变换
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(

        type='StandardRoIHead',

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        bbox_head=dict(
            # 2 个共享 FC 模块
            type='Shared2FCBBoxHead',
            # 输入通道数，相等于 FPN 输出通道
            in_channels=256,
            # 中间 FC 层节点个数
            fc_out_channels=1024,
            # RoIAlign 或 RoIPool 输出的特征图大小
            roi_feat_size=7,
            num_classes=16,
            # bbox 编解码策略，除了参数外和 RPN 相同，
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # 影响 bbox 分支的通道数，True 表示 4 通道输出，False 表示 4×num_classes 通道输出
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),

    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            # 正负样本分配
            assigner=dict(
                # 最大IOU原则分配器
                type='MaxIoUAssigner',
                # 正样本阈值 如果 anchor 和所有 gt bbox 的最大 iou 值大于等于 0.7，那么该 anchor 就是高质量正样本
                pos_iou_thr=0.7,
                # 负样本阈值  如果 anchor 和所有 gt bbox 的最大 iou 值小于 0.3，那么该 anchor 就是背景样本
                neg_iou_thr=0.3,
                # 正样本阈值下限 如果 gt bbox 和所有 anchor 的最大 iou 值大于等于 0.3，那么该 gt bbox 所对应的 anchor 也是正样本
                min_pos_iou=0.3,
                match_low_quality=True,
                # 忽略 bboxes 的阈值，-1 表示不忽略
                ignore_iof_thr=-1),
            sampler=dict(
                # 随机采样
                type='RandomSampler',
                # 采样后每个mini_batch的训练样本总数，不包括忽略样本
                num=256,
                # 正样本比例
                pos_fraction=0.5,
                # 正负样本比例，用于确定负样本采样个数上界 -1表示正样本不足用负样本凑
                neg_pos_ub=-1,
                # 是否加入 gt 作为 proposals 以增加高质量正样本数
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            # nms 前每个输出层最多保留1000个预测框
            nms_pre=1000,
            # 最终输出的每张图片最多 bbox 个数
            max_per_img=1000,
            # nms 方法和 nms 阈值
            nms=dict(type='nms', iou_threshold=0.7),
            # 过滤掉的最小 bbox 尺寸
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
