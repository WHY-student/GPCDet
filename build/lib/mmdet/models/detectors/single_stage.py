# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    一阶段检测器的基类
    一阶段检测器在backbone+neck的输出上直接进行密集的边界框预测
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        """创建backbone，build函数作用与build_detecor相同"""
        self.backbone = build_backbone(backbone)

        """创建backbone，build函数作用与build_detecor相同"""
        if neck is not None:
            self.neck = build_neck(neck)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        """创建bbox_head"""
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    """特征提取"""
    def extract_feat(self, img):
        """使用backbone和neck提取特征 """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """前向传播算法  x ---> backbone+neck ---> feat ---> bbox_head ---> outs"""
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        参数:
            img (Tensor): 输入图片，形状为(N,C,H,W),一般来说应当是归一化后的图片.
            img_metas (list[dict]): 包含'image_scale','flip','filename','ori_shape'等元信息的字典列表
            gt_bboxes (list[Tensor]): 边界框真实标注，形状为(xmin,ymin,xmax,ymax).
            gt_labels (list[Tensor]): 边界框的类别
            gt_bboxes_ignore (None | list[Tensor]): 指定在计算损失时可以被忽略的边界框.

        返回值:
            dict[str, Tensor]: 包含多个损失函数的字典.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """没有测试阶段数据增强的简单测试函数
        参数:
            rescale (bool, optional): 是否将检测结果放缩到原有的大小，默认为False.
        返回值:
            list[list[np.ndarray]]: 每张图片中每一个类别的检测结果，第一个list维度代表不同的图片，第二个list维度代表不同的类别.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """与simple_test函数基本一致，只是在测试时使用了数据增强
        """

        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
